import torch
import torch.nn as nn

from models import networks_img
from models.mmcv.conv_module import ConvModule
from models import resnet
from models import networks_pc
from models import layers_common
from models import layers_pc
from kitti.options import Options
from util import pytorch_helper


class KeypointDetector(nn.Module):
    def __init__(self, opt: Options):
        super(KeypointDetector, self).__init__()
        self.opt = opt

        self.pc_encoder = networks_pc.PCEncoder(opt, Ca=64, Cb=256, Cg=512).to(self.opt.device)
        self.img_encoder = networks_img.ImageEncoder(self.opt).to(self.opt.device)

        self.H_fine_res = int(round(self.opt.img_H / self.opt.img_fine_resolution_scale))
        self.W_fine_res = int(round(self.opt.img_W / self.opt.img_fine_resolution_scale))

        self.node_b_attention_pn = layers_pc.PointNet(256+512,
                                               [256, self.H_fine_res*self.W_fine_res],
                                               activation=self.opt.activation,
                                               normalization=self.opt.normalization,
                                               norm_momentum=opt.norm_momentum,
                                               norm_act_at_last=False)

        # in_channels: node_b_features + global_feature + image_s32_feature + image_global_feature
        self.node_b_pn = layers_pc.PointNet(256+512+512+512,
                                            [1024, 512, 512],
                                            activation=self.opt.activation,
                                            normalization=self.opt.normalization,
                                            norm_momentum=opt.norm_momentum,
                                            norm_act_at_last=False)

        self.node_a_attention_pn = layers_pc.PointNet(64 + 512,
                                                      [256, int(self.H_fine_res * self.W_fine_res * 4)],
                                                      activation=self.opt.activation,
                                                      normalization=self.opt.normalization,
                                                      norm_momentum=opt.norm_momentum,
                                                      norm_act_at_last=False)

        # in_channels: node_a_features + interpolated node_b_features
        self.node_a_pn = layers_pc.PointNet(64+256+512,
                                            [512, 128, 128],
                                            activation=self.opt.activation,
                                            normalization=self.opt.normalization,
                                            norm_momentum=opt.norm_momentum,
                                            norm_act_at_last=False)

        # final network for per-point labeling
        # in_channels: second_pn_out + interpolated node_a_features
        per_point_pn_in_channels = 32 + 64 + 128 + 512
        # per_point_pn_in_channels = 32 + 64 + 512 + 512
        if self.opt.is_fine_resolution:
            self.per_point_pn = layers_pc.PointNet(per_point_pn_in_channels,
                                                   [256, 256, 2 + self.H_fine_res * self.W_fine_res],
                                                   activation=self.opt.activation,
                                                   normalization=self.opt.normalization,
                                                   norm_momentum=opt.norm_momentum,
                                                   norm_act_at_last=False,
                                                   dropout_list=[0.5, 0.5, 0])
        else:
            self.per_point_pn = layers_pc.PointNet(per_point_pn_in_channels,
                                                   [128, 128, 2],
                                                   activation=self.opt.activation,
                                                   normalization=self.opt.normalization,
                                                   norm_momentum=opt.norm_momentum,
                                                   norm_act_at_last=False,
                                                   dropout_list=[0.5, 0.5, 0])

    def gather_topk_features(self, min_k_idx, features):
        """

        :param min_k_idx: BxNxk
        :param features: BxCxM
        :return:
        """
        B, N, k = min_k_idx.size(0), min_k_idx.size(1), min_k_idx.size(2)
        C, M = features.size(1), features.size(2)

        return torch.gather(features.unsqueeze(3).expand(B, C, M, k),
                            index=min_k_idx.unsqueeze(1).expand(B, C, N, k),
                            dim=2)  # BxCxNxk

    def upsample_by_interpolation(self,
                                  interp_ab_topk_idx,
                                  node_a,
                                  node_b,
                                  up_node_b_features):
        interp_ab_topk_node_b = self.gather_topk_features(interp_ab_topk_idx, node_b)  # Bx3xMaxk
        # Bx3xMa -> Bx3xMaxk -> BxMaxk
        interp_ab_node_diff = torch.norm(node_a.unsqueeze(3) - interp_ab_topk_node_b, dim=1, p=2, keepdim=False)
        interp_ab_weight = 1 - interp_ab_node_diff / torch.sum(interp_ab_node_diff, dim=2, keepdim=True)  # BxMaxk
        interp_ab_topk_node_b_features = self.gather_topk_features(interp_ab_topk_idx, up_node_b_features)  # BxCxMaxk
        # BxCxMaxk -> BxCxMa
        interp_ab_weighted_node_b_features = torch.sum(interp_ab_weight.unsqueeze(1) * interp_ab_topk_node_b_features,
                                                       dim=3)
        return interp_ab_weighted_node_b_features

    def forward(self,
                pc, intensity, sn, node_a, node_b,
                img):
        """

        :param pc: Bx3xN
        :param intensity: Bx1xN
        :param sn: Bx3xN
        :param node: Bx3xM
        :param img: BLx3xHxW
        :return:
        """
        B, N, Ma, Mb = pc.size(0), pc.size(2), node_a.size(2), node_b.size(2)

        # point cloud detector ----------------------------------------------------
        # BxC_pointxN
        pc_center,\
        cluster_mean, \
        node_a_min_k_idx, \
        first_pn_out, \
        second_pn_out, \
        node_a_features, \
        node_b_features, \
        global_feature = self.pc_encoder(pc,
                                          intensity,
                                          sn,
                                          node_a,
                                          node_b)
        C_global = global_feature.size(1)

        # image detector ----------------------------------------------------------
        # BxC_imgxHxW, BxC_imgx1x1
        img_s16_feature_map, img_s32_feature_map, img_global_feature = self.img_encoder(img)
        C_img = img_global_feature.size(1)
        img_s16_feature_map_BCHw = img_s16_feature_map.view(B, img_s16_feature_map.size(1), -1)  # BxC_imgx(H*W)
        img_s32_feature_map_BCHw = img_s32_feature_map.view(B, img_s32_feature_map.size(1), -1)  # BxC_imgx(H*W)
        img_global_feature_BCMa = img_global_feature.squeeze(3).expand(B, C_img, Ma)  # BxC_img -> BxC_imgxMa
        img_global_feature_BCMb = img_global_feature.squeeze(3).expand(B, C_img, Mb)  # BxC_img -> BxC_imgxMb

        # assemble node_a_features, node_b_features, global_feature, img_feature, img_s32_features
        # ----------------------------------------
        # use attention method to select resnet features for each node_b_feature
        node_b_attention_score = self.node_b_attention_pn(torch.cat((node_b_features,
                                                                     img_global_feature_BCMb), dim=1))  # Bx(H*W)xMb
        node_b_weighted_img_s32_feature_map = torch.mean(img_s32_feature_map_BCHw.unsqueeze(3) * node_b_attention_score.unsqueeze(1),
                                                  dim=2)  # BxC_imgx(H*W)xMb -> BxC_imgxMb

        up_node_b_features = self.node_b_pn(torch.cat((node_b_features,
                                                       global_feature.expand(B, C_global, Mb),
                                                       node_b_weighted_img_s32_feature_map,
                                                       img_global_feature_BCMb), dim=1))  # BxCxMb

        # interpolation of pc over node_b
        pc_node_b_diff = torch.norm(pc.unsqueeze(3) - node_b.unsqueeze(2), p=2, dim=1, keepdim=False)  # BxNxMb
        # BxNxk
        _, interp_pc_node_b_topk_idx = torch.topk(pc_node_b_diff, k=self.opt.k_interp_point_b,
                                                  dim=2, largest=False, sorted=True)
        interp_pb_weighted_node_b_features = self.upsample_by_interpolation(interp_pc_node_b_topk_idx,
                                                                            pc,
                                                                            node_b,
                                                                            up_node_b_features)


        # interpolation of point over node_a  ----------------------------------------------
        # use attention method to select resnet features for each node_a_feature
        node_a_attention_score = self.node_a_attention_pn(torch.cat((node_a_features,
                                                                     img_global_feature_BCMa), dim=1))  # Bx(H*W)xMa
        node_a_weighted_img_s16_feature_map = torch.mean(
            img_s16_feature_map_BCHw.unsqueeze(3) * node_a_attention_score.unsqueeze(1),
            dim=2)  # BxC_imgx(H*W)xMa -> BxC_imgxMa
        # interpolation of node_a over node_b
        node_a_node_b_diff = torch.norm(node_a.unsqueeze(3) - node_b.unsqueeze(2), p=2, dim=1, keepdim=False)  # BxMaxMb
        _, interp_nodea_nodeb_topk_idx = torch.topk(node_a_node_b_diff, k=self.opt.k_interp_ab,
                                                    dim=2, largest=False, sorted=True)
        interp_ab_weighted_node_b_features = self.upsample_by_interpolation(interp_nodea_nodeb_topk_idx,
                                                                            node_a,
                                                                            node_b,
                                                                            up_node_b_features)

        up_node_a_features = self.node_a_pn(torch.cat((node_a_features,
                                                       interp_ab_weighted_node_b_features,
                                                       node_a_weighted_img_s16_feature_map),
                                                      dim=1))  # BxCxMa
        interp_pa_weighted_node_a_features = self.upsample_by_interpolation(node_a_min_k_idx,
                                                                            pc,
                                                                            node_a,
                                                                            up_node_a_features)

        # per-point network
        pc_label_scores = self.per_point_pn(torch.cat((interp_pa_weighted_node_a_features,
                                                       interp_pb_weighted_node_b_features,
                                                       first_pn_out,
                                                       second_pn_out), dim=1))

        # pc_label_scores = self.per_point_pn(torch.cat((img_global_feature.squeeze(2).expand(B, C_img, N),
        #                                                global_feature.expand(B, C_global, N),
        #                                                first_pn_out,
        #                                                second_pn_out), dim=1))

        coarse_scores = pc_label_scores[:, 0:2, :]  # Bx2xN

        if self.opt.is_fine_resolution:
            fine_scores = pc_label_scores[:, 2:, :]  # BxLxN
            return coarse_scores, fine_scores  # Bx2xN
        else:
            return coarse_scores



