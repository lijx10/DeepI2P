import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time

from models import operations
from models.layers_pc import *
from kitti.options import Options

import index_max


class PCEncoder(nn.Module):
    def __init__(self, opt: Options, Ca: int, Cb: int, Cg: int):
        super(PCEncoder, self).__init__()
        self.opt = opt

        # first PointNet
        self.first_pointnet = PointNet(3 + 1 + 3,
                                       [int(Ca / 2), int(Ca / 2), int(Ca / 2)],
                                       activation=self.opt.activation,
                                       normalization=self.opt.normalization,
                                       norm_momentum=opt.norm_momentum,
                                       norm_act_at_last=True)

        self.second_pointnet = PointNet(Ca, [Ca, Ca], activation=self.opt.activation,
                                        normalization=self.opt.normalization,
                                        norm_momentum=opt.norm_momentum,
                                        norm_act_at_last=True)

        self.knnlayer = GeneralKNNFusionModule(3 + Ca, (int(Cb), int(Cb)),
                                               (Cb*2, Cb),
                                               activation=self.opt.activation,
                                               normalization=self.opt.normalization,
                                               norm_momentum=opt.norm_momentum)

        self.final_pointnet = PointNet(3+Cb, [int(Cg/2), Cg], activation=self.opt.activation,
                                       normalization=self.opt.normalization,
                                       norm_momentum=opt.norm_momentum,
                                       norm_act_at_last=True)

        node_idx_list = torch.from_numpy(np.arange(self.opt.node_a_num).astype(np.int64)).to(device=self.opt.device, dtype=torch.long)  # ma LongTensor
        self.node_idx_1NMa = node_idx_list.unsqueeze(0).unsqueeze(1).expand(1, self.opt.input_pt_num, self.opt.node_a_num)  # 1xNxMa

    def forward(self, pc, intensity, sn, node_a, node_b):
        '''

        :param pc: Bx3xN Tensor
        :param intensity: Bx1xN Tensor
        :param sn: Bx3xN Tensor
        :param node_a: Bx3xMa FloatTensor
        :param node_b: Bx3xMb FloatTensor
        :param keypoint_anchor_idx: BxK IntTensor
        :return:
        '''
        B, N, Ma, Mb = pc.size(0), pc.size(2), node_a.size(2), node_b.size(2)

        # modify the pc according to the node_a, minus the center
        pc_B3NMa = pc.unsqueeze(3).expand(B, 3, N, Ma)
        node_a_B3NMa = node_a.unsqueeze(2).expand(B, 3, N, Ma)
        diff = torch.norm(pc_B3NMa - node_a_B3NMa, dim=1, p=2, keepdim=False)  # BxNxMa
        _, min_k_idx = torch.topk(diff, k=self.opt.k_interp_point_a, dim=2, largest=False, sorted=True)  # BxNxk0
        min_idx = min_k_idx[:, :, 0]  # BxN
        mask = torch.eq(min_idx.unsqueeze(2).expand(B, N, Ma),
                        self.node_idx_1NMa.to(device=min_idx.device, dtype=torch.long).expand(B, N, Ma))  # BxNxMa
        mask_row_max, _ = torch.max(mask, dim=1, keepdim=False)  # BxMa, this indicates whether the node has nearby points
        mask_row_max_B1Ma_float = mask_row_max.unsqueeze(1).to(dtype=torch.float)

        mask_B1NMa_float = mask.unsqueeze(1).to(dtype=torch.float)  # Bx1xNxMa
        mask_row_sum = torch.sum(mask_B1NMa_float, dim=2, keepdim=False)  # Bx1xMa

        # calculate the center of the cluster
        pc_masked = pc.unsqueeze(3) * mask_B1NMa_float  # BxCxNxMa
        cluster_mean = torch.sum(pc_masked, dim=2) / (mask_row_sum + 1e-5).detach()  # BxCxMa

        # assign each point with a center
        pc_centers = torch.gather(cluster_mean,
                                     index=min_idx.unsqueeze(1).expand(B, 3, N),
                                     dim=2)  # Bx3xN
        pc_decentered = (pc - pc_centers).detach()  # Bx3xN

        # go through the first PointNet
        pc_augmented = torch.cat((pc_decentered, intensity, sn), dim=1)  # Bx7xN
        first_pn_out = self.first_pointnet(pc_augmented)

        with torch.cuda.device(first_pn_out.get_device()):
            first_gather_index = index_max.forward_cuda_shared_mem(first_pn_out.detach(), min_idx.int(),
                                                                   Ma).detach().long()
        first_pn_out_masked_max = first_pn_out.gather(dim=2,
                                                      index=first_gather_index) * mask_row_max_B1Ma_float  # BxCxMa

        # scatter the masked_max back to the N points
        scattered_first_masked_max = torch.gather(first_pn_out_masked_max,
                                                  dim=2,
                                                  index=min_idx.unsqueeze(1).expand(B, first_pn_out.size(1), N))  # BxCxN
        first_pn_out_fusion = torch.cat((first_pn_out, scattered_first_masked_max), dim=1)  # Bx2CxN
        second_pn_out = self.second_pointnet(first_pn_out_fusion)

        with torch.cuda.device(second_pn_out.get_device()):
            second_gather_index = index_max.forward_cuda_shared_mem(second_pn_out, min_idx.int(), Ma).detach().long()
        node_a_features = second_pn_out.gather(dim=2,
                                               index=second_gather_index) * mask_row_max_B1Ma_float  # BxCaxMa

        # knn module, knn search on nodes: ----------------------------------
        node_b_features = self.knnlayer(query=node_b,
                                        database=cluster_mean,
                                        database_features=node_a_features,
                                        # database_features=torch.cat((cluster_mean, second_pn_out_masked_max), dim=1),
                                        K=self.opt.k_ab)  # BxCbxM

        # get global feature
        final_pn_out = self.final_pointnet(torch.cat((node_b, node_b_features), dim=1))  # BxCgxN
        global_feature, _ = torch.max(final_pn_out, dim=2, keepdim=True)  # BxCgx1

        return pc_centers,\
               cluster_mean,\
               min_k_idx, \
               first_pn_out, \
               second_pn_out, \
               node_a_features, \
               node_b_features, \
               global_feature

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
