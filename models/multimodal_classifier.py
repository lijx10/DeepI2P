import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
from collections import OrderedDict
import os
import random
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from models import networks_pc
from models import networks_img
from models import networks_united
from models import losses
from data import augmentation
from models import operations
from kitti.options import Options
from util import pytorch_helper
from util import vis_tools
from models import focal_loss


class MMClassifer():
    def __init__(self, opt: Options, writer):
        self.opt = opt
        self.writer = writer
        self.global_step = 0

        self.detector = networks_united.KeypointDetector(self.opt).to(self.opt.device)
        # self.coarse_ce_criteria = nn.CrossEntropyLoss()
        self.coarse_ce_criteria = focal_loss.FocalLoss(alpha=0.5, gamma=2, reduction='mean')
        self.fine_ce_criteria = nn.CrossEntropyLoss()

        # multi-gpu training
        if len(opt.gpu_ids) > 1:
            self.detector = nn.DataParallel(self.detector, device_ids=opt.gpu_ids)
            # self.coarse_ce_criteria = nn.DataParallel(self.coarse_ce_criteria, device_ids=opt.gpu_ids)
            # self.fine_ce_criteria = nn.DataParallel(self.fine_ce_criteria, device_ids=opt.gpu_ids)


        # learning rate_control
        self.old_lr_detector = self.opt.lr
        self.optimizer_detector = torch.optim.Adam(self.detector.parameters(),
                                                   lr=self.old_lr_detector,
                                                   betas=(0.9, 0.999),
                                                   weight_decay=0)

        # place holder for GPU tensors
        self.pc = torch.empty(self.opt.batch_size, 3, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.intensity = torch.empty(self.opt.batch_size, 1, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.sn = torch.empty(self.opt.batch_size, 3, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.node_a = torch.empty(self.opt.batch_size, 3, self.opt.node_a_num, dtype=torch.float, device=self.opt.device)
        self.node_b = torch.empty(self.opt.batch_size, 3, self.opt.node_b_num, dtype=torch.float, device=self.opt.device)
        self.P = torch.empty(self.opt.batch_size, 3, 4, dtype=torch.float, device=self.opt.device)
        self.img = torch.empty(self.opt.batch_size, 3, self.opt.img_H, self.opt.img_W, dtype=torch.float, device=self.opt.device)
        self.K = torch.empty(self.opt.batch_size, 3, 3, dtype=torch.float, device=self.opt.device)

        # record the train / test loss and accuracy
        self.train_loss_dict = {}
        self.test_loss_dict = {}

        self.train_accuracy = {}
        self.test_accuracy = {}

        # pre-cautions for shared memory usage
        # if opt.batch_size * 2 / len(opt.gpu_ids) > operations.CUDA_SHARED_MEM_DIM_X or opt.node_num > operations.CUDA_SHARED_MEM_DIM_Y:
        #     print('--- WARNING: batch_size or node_num larger than pre-defined cuda shared memory array size. '
        #           'Please modify CUDA_SHARED_MEM_DIM_X and CUDA_SHARED_MEM_DIM_Y in models/operations.py')

    def global_step_inc(self, delta):
        self.global_step += delta

    def load_model(self, model_path):
        self.detector.load_state_dict(
            pytorch_helper.model_state_dict_convert_auto(
                torch.load(
                    model_path,
                    map_location='cpu'), self.opt.gpu_ids))

    def set_input(self,
                  pc, intensity, sn, node_a, node_b,
                  P,
                  img, K):
        self.pc.resize_(pc.size()).copy_(pc).detach()
        self.intensity.resize_(intensity.size()).copy_(intensity).detach()
        self.sn.resize_(sn.size()).copy_(sn).detach()
        self.node_a.resize_(node_a.size()).copy_(node_a).detach()
        self.node_b.resize_(node_b.size()).copy_(node_b).detach()
        self.P.resize_(P.size()).copy_(P).detach()
        self.img.resize_(img.size()).copy_(img).detach()
        self.K.resize_(K.size()).copy_(K).detach()

    def forward(self,
                pc, intensity, sn, node_a, node_b,
                img):
        return self.detector(pc, intensity, sn, node_a, node_b, img)

    def inference_pass(self):
        N = self.pc.size(2)
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)

        img_W_fine_res = int(self.opt.img_W / self.opt.img_fine_resolution_scale)
        img_H_fine_res = int(self.opt.img_H / self.opt.img_fine_resolution_scale)

        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>
        # BLx2xN, BLxKxN
        coarse_scores, fine_scores = self.forward(self.pc, self.intensity, self.sn, self.node_a, self.node_b,
                                                  self.img)
        K = fine_scores.size(1)
        assert K == img_W_fine_res * img_H_fine_res
        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>

        _, coarse_prediction = torch.max(coarse_scores, dim=1, keepdim=False)  # BLxN
        _, fine_prediction = torch.max(fine_scores, dim=1, keepdim=False)  # BLxN
        return coarse_prediction, fine_prediction

    def foraward_pass(self):
        N = self.pc.size(2)
        Ma, Mb = self.opt.node_a_num, self.opt.node_b_num
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)

        img_W_fine_res = int(round(W / self.opt.img_fine_resolution_scale))
        img_H_fine_res = int(round(H / self.opt.img_fine_resolution_scale))

        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>
        # Bx2xN, BxLxN
        coarse_scores, fine_scores = self.forward(self.pc, self.intensity, self.sn, self.node_a, self.node_b,
                                                  self.img)
        L = fine_scores.size(1)
        assert L == img_W_fine_res * img_H_fine_res
        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>

        # project points onto image to get ground truth labels for both coarse and fine resolution
        pc_homo = torch.cat((self.pc,
                                torch.ones((B, 1, N), dtype=torch.float32, device=self.pc.device)),
                               dim=1)  # Bx4xN
        P_pc_homo = torch.matmul(self.P, pc_homo)  # Bx3xN
        KP_pc_homo = torch.matmul(self.K, P_pc_homo)  # Bx3xN
        KP_pc_pxpy = KP_pc_homo[:, 0:2, :] / KP_pc_homo[:, 2:3, :]  # Bx2xN

        x_inside_mask = (KP_pc_pxpy[:, 0:1, :] >= 0) \
                        & (KP_pc_pxpy[:, 0:1, :] <= W - 1)  # Bx1xN
        y_inside_mask = (KP_pc_pxpy[:, 1:2, :] >= 0) \
                        & (KP_pc_pxpy[:, 1:2, :] <= H - 1)  # Bx1xN
        z_inside_mask = KP_pc_homo[:, 2:3, :] > 0.1  # Bx1xN
        inside_mask = (x_inside_mask & y_inside_mask & z_inside_mask).squeeze(1)  # BxN

        # apply scaling to get fine resolution
        # Bx2xN
        KP_pc_pxpy_scale_int = torch.floor(KP_pc_pxpy / self.opt.img_fine_resolution_scale).to(dtype=torch.long)
        KP_pc_pxpy_index = KP_pc_pxpy_scale_int[:, 0, :] + KP_pc_pxpy_scale_int[:, 1, :] * img_W_fine_res  # BxN

        # get coarse labels
        coarse_labels = inside_mask.to(dtype=torch.long)  # BxN

        # get fine labels
        # organize everything into (B*N)x* shape
        inside_mask_Bn = inside_mask.reshape(B*N)  # BN
        inside_mask_Bn_int = inside_mask_Bn.to(dtype=torch.int32)  # BN
        insider_num = int(torch.sum(inside_mask_Bn_int).item())  # scalar
        _, inside_idx_Bn = torch.sort(inside_mask_Bn_int, descending=True)  # BN
        insider_idx = inside_idx_Bn[0: insider_num]  # B_insider
        
        KP_pc_pxpy_index_Bn = KP_pc_pxpy_index.view(B*N)  # BN in long
        KP_pc_pxpy_index_insider = torch.gather(KP_pc_pxpy_index_Bn, dim=0, index=insider_idx)  # B_insider in long
        # assure correctness
        fine_labels_min = torch.min(KP_pc_pxpy_index_insider).item()
        fine_labels_max = torch.max(KP_pc_pxpy_index_insider).item()
        assert fine_labels_min >= 0
        assert fine_labels_max <= img_W_fine_res * img_H_fine_res - 1

        # BxLxN -> BxNxL
        fine_scores_BnL = fine_scores.permute(0, 2, 1).reshape(B*N, L).contiguous()  # BNxL
        insider_idx_BinsiderL = insider_idx.unsqueeze(1).expand(insider_num, L)  # B_insiderxL
        fine_scores_insider = torch.gather(fine_scores_BnL, dim=0, index=insider_idx_BinsiderL)  # B_insiderxL

        # build loss -------------------------------------------------------------------------------------
        coarse_loss = self.coarse_ce_criteria(coarse_scores, coarse_labels) * self.opt.coarse_loss_alpha
        fine_loss = self.fine_ce_criteria(fine_scores_insider, KP_pc_pxpy_index_insider)
        loss = coarse_loss + fine_loss
        # build loss -------------------------------------------------------------------------------------

        # get accuracy
        # get predictions for visualization
        _, coarse_predictions = torch.max(coarse_scores, dim=1, keepdim=False)
        coarse_accuracy = torch.sum(torch.eq(coarse_labels, coarse_predictions).to(dtype=torch.float)) / ( B * N)

        _, fine_predictions_insider = torch.max(fine_scores_insider, dim=1, keepdim=False)
        fine_accuracy = torch.sum(torch.eq(KP_pc_pxpy_index_insider, fine_predictions_insider).to(dtype=torch.float)) / insider_num

        _, fine_predictions = torch.max(fine_scores, dim=1, keepdim=False)


        loss_dict = {'loss': loss,
                     'coarse': coarse_loss,
                     'fine': fine_loss}
        vis_dict = {'pc': P_pc_homo,
                    'coarse_labels': coarse_labels,
                    'fine_labels': KP_pc_pxpy_index,
                    'coarse_predictions': coarse_predictions,
                    'fine_predictions': fine_predictions,
                    'KP_pc_pxpy': KP_pc_pxpy}
        accuracy_dict = {'coarse_accuracy': coarse_accuracy,
                         'fine_accuracy': fine_accuracy}

        # debug
        if self.opt.is_debug:
            print(loss_dict)

        return loss_dict, vis_dict, accuracy_dict

    def optimize(self):
        self.detector.train()
        self.detector.zero_grad()
        self.train_loss_dict, self.train_visualization, self.train_accuracy = self.foraward_pass()
        self.train_loss_dict['loss'].backward()
        self.optimizer_detector.step()

    def test_model(self):
        self.detector.eval()
        with torch.no_grad():
            self.test_loss_dict, self.test_visualization, self.test_accuracy = self.foraward_pass()

    def freeze_model(self):
        print("!!! WARNING: Model freezed.")
        for p in self.detector.parameters():
            p.requires_grad = False

    def run_model(self, pc, intensity, sn, node, img):
        self.detector.eval()
        with torch.no_grad():
            raise Exception("Not implemented.")

    def tensor_dict_to_float_dict(self, tensor_dict):
        float_dict = {}
        for key, value in tensor_dict.items():
            if type(value) == torch.Tensor:
                float_dict[key] = value.item()
            else:
                float_dict[key] = value
        return float_dict

    def get_current_errors(self):
        return self.tensor_dict_to_float_dict(self.train_loss_dict), \
               self.tensor_dict_to_float_dict(self.test_loss_dict)

    def get_current_accuracy(self):
        return self.tensor_dict_to_float_dict(self.train_accuracy), \
               self.tensor_dict_to_float_dict(self.test_accuracy)

    @staticmethod
    def print_loss_dict(loss_dict, accuracy_dict=None, duration=-1):
        output = 'Per sample time: %.4f - ' % (duration)
        for key, value in loss_dict.items():
            output += '%s: %.4f, ' % (key, value)
        if accuracy_dict is not None:
            for key, value in accuracy_dict.items():
                output += '%s: %.4f, ' % (key, value)
        print(output)

    def save_network(self, network, save_filename):
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def update_learning_rate(self, ratio):
        lr_clip = 0.00001

        # detector
        lr_detector = self.old_lr_detector * ratio
        if lr_detector < lr_clip:
            lr_detector = lr_clip
        for param_group in self.optimizer_detector.param_groups:
            param_group['lr'] = lr_detector
        print('update detector learning rate: %f -> %f' % (self.old_lr_detector, lr_detector))
        self.old_lr_detector = lr_detector

    # visualization with tensorboard, pytorch 1.2
    def write_loss(self):
        train_loss_dict, test_loss_dict = self.get_current_errors()
        self.writer.add_scalars('train_loss',
                                train_loss_dict,
                                global_step=self.global_step)
        self.writer.add_scalars('test_loss',
                                test_loss_dict,
                                global_step=self.global_step)

    def write_accuracy(self):
        train_acc_dict, test_acc_dict = self.get_current_accuracy()
        self.writer.add_scalars('train_accuracy',
                                train_acc_dict,
                                global_step=self.global_step)
        self.writer.add_scalars('test_accuracy',
                                test_acc_dict,
                                global_step=self.global_step)

    def write_pc_label(self, pc, labels, title):
        """
        :param pc: Bx3xN
        :param labels: BxN
        :param title: string
        :return:
        """
        pc_np = pc.detach().cpu().numpy()  # Bx3xN
        labels_np = labels.detach().cpu().numpy()  # BxN

        B = pc_np.shape[0]

        visualization_fig_np_list = []
        vis_number = min(self.opt.vis_max_batch, B)
        for b in range(vis_number):
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(pc_np[b, 0, :], pc_np[b, 2, :],
                        c=labels_np[b, :],
                        cmap='jet',
                        marker='o',
                        s=1)
            fig.tight_layout()
            plt.axis('equal')
            fig_img_np = vis_tools.fig_to_np(fig)
            plt.close(fig)
            visualization_fig_np_list.append(fig_img_np)

        imgs_vis_grid = vis_tools.visualization_list_to_grid(visualization_fig_np_list, col=2)
        imgs_vis_grid = np.moveaxis(imgs_vis_grid, (0, 1, 2), (1, 2, 0))
        self.writer.add_image(title, imgs_vis_grid, global_step=self.global_step)

    def write_img(self):
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)
        imgs = self.img.detach().round().to(dtype=torch.uint8).cpu()  # Bx3xHxW

        vis_number = min(self.opt.vis_max_batch, B)
        imgs_vis = imgs[0:vis_number, ...]  # Bx3xHxW
        imgs_vis_grid = torchvision.utils.make_grid(imgs_vis, nrow=2)
        self.writer.add_image('image', imgs_vis_grid, global_step=self.global_step)

    def write_classification_visualization(self,
                                           pc_pxpy,
                                           coarse_predictions, fine_predictions,
                                           coarse_labels, fine_labels,
                                           t_ij):
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)
        imgs = self.img.detach().round().to(dtype=torch.uint8).cpu()  # Bx3xHxW
        imgs_np = imgs.permute(0, 2, 3, 1).numpy()  # BxHxWx3

        pc_pxpy_np = pc_pxpy.cpu().numpy()  # Bx2xN
        coarse_predictions_np = coarse_predictions.detach().cpu().numpy()  # BxN
        fine_predictions_np = fine_predictions.detach().cpu().numpy()  # BxN
        coarse_labels_np = coarse_labels.detach().cpu().numpy()  # BxN
        fine_labels_np = fine_labels.detach().cpu().numpy()  # BxN

        t_ij_np = t_ij.cpu().numpy()  # Bx3

        vis_number = min(self.opt.vis_max_batch, B)
        visualization_fig_np_list = []
        for b in range(vis_number):
            img_b = imgs_np[b, ...]  # HxWx3
            pc_pxpy_b = pc_pxpy_np[b, ...]  # 2xN
            coarse_prediction_b = coarse_predictions_np[b, :]  # N
            fine_predictions_b = fine_predictions_np[b, :]  # N
            coarse_labels_b = coarse_labels_np[b, :]  # N
            fine_labels_b = fine_labels_np[b, :]  # N

            vis_img = vis_tools.get_classification_visualization(pc_pxpy_b,
                                                                 coarse_prediction_b, fine_predictions_b,
                                                                 coarse_labels_b, fine_labels_b,
                                                                 img_b,
                                                                 img_fine_resolution_scale=self.opt.img_fine_resolution_scale,
                                                                 H_delta=100, W_delta=100,
                                                                 circle_size=1,
                                                                 t_ij_np=t_ij_np[b, :])
            visualization_fig_np_list.append(vis_img)

        imgs_vis_grid = vis_tools.visualization_list_to_grid(visualization_fig_np_list, col=2)
        imgs_vis_grid = np.moveaxis(imgs_vis_grid, (0, 1, 2), (1, 2, 0))
        self.writer.add_image("Classification Visualization", imgs_vis_grid, global_step=self.global_step)


class MMClassiferCoarse():
    def __init__(self, opt: Options, writer):
        self.opt = opt
        self.writer = writer
        self.global_step = 0

        self.detector = networks_united.KeypointDetector(self.opt).to(self.opt.device)
        # self.coarse_ce_criteria = nn.CrossEntropyLoss()
        self.coarse_ce_criteria = focal_loss.FocalLoss(alpha=0.5, gamma=2, reduction='mean')
        # self.fine_ce_criteria = nn.CrossEntropyLoss()

        # multi-gpu training
        if len(opt.gpu_ids) > 1:
            self.detector = nn.DataParallel(self.detector, device_ids=opt.gpu_ids)
            # self.coarse_ce_criteria = nn.DataParallel(self.coarse_ce_criteria, device_ids=opt.gpu_ids)
            # self.fine_ce_criteria = nn.DataParallel(self.fine_ce_criteria, device_ids=opt.gpu_ids)

        # learning rate_control
        self.old_lr_detector = self.opt.lr
        self.optimizer_detector = torch.optim.Adam(self.detector.parameters(),
                                                   lr=self.old_lr_detector,
                                                   betas=(0.9, 0.999),
                                                   weight_decay=0)

        # place holder for GPU tensors
        self.pc = torch.empty(self.opt.batch_size, 3, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.intensity = torch.empty(self.opt.batch_size, 1, self.opt.input_pt_num, dtype=torch.float,
                                     device=self.opt.device)
        self.sn = torch.empty(self.opt.batch_size, 3, self.opt.input_pt_num, dtype=torch.float, device=self.opt.device)
        self.node_a = torch.empty(self.opt.batch_size, 3, self.opt.node_a_num, dtype=torch.float,
                                  device=self.opt.device)
        self.node_b = torch.empty(self.opt.batch_size, 3, self.opt.node_b_num, dtype=torch.float,
                                  device=self.opt.device)
        self.P = torch.empty(self.opt.batch_size, 3, 4, dtype=torch.float, device=self.opt.device)
        self.img = torch.empty(self.opt.batch_size, 3, self.opt.img_H, self.opt.img_W, dtype=torch.float,
                               device=self.opt.device)
        self.K = torch.empty(self.opt.batch_size, 3, 3, dtype=torch.float, device=self.opt.device)

        # record the train / test loss and accuracy
        self.train_loss_dict = {}
        self.test_loss_dict = {}

        self.train_accuracy = {}
        self.test_accuracy = {}

        # pre-cautions for shared memory usage
        # if opt.batch_size * 2 / len(opt.gpu_ids) > operations.CUDA_SHARED_MEM_DIM_X or opt.node_num > operations.CUDA_SHARED_MEM_DIM_Y:
        #     print('--- WARNING: batch_size or node_num larger than pre-defined cuda shared memory array size. '
        #           'Please modify CUDA_SHARED_MEM_DIM_X and CUDA_SHARED_MEM_DIM_Y in models/operations.py')

    def global_step_inc(self, delta):
        self.global_step += delta

    def load_model(self, model_path):
        self.detector.load_state_dict(
            pytorch_helper.model_state_dict_convert_auto(
                torch.load(
                    model_path,
                    map_location='cpu'), self.opt.gpu_ids))

    def set_input(self,
                  pc, intensity, sn, node_a, node_b,
                  P,
                  img, K):
        self.pc.resize_(pc.size()).copy_(pc).detach()
        self.intensity.resize_(intensity.size()).copy_(intensity).detach()
        self.sn.resize_(sn.size()).copy_(sn).detach()
        self.node_a.resize_(node_a.size()).copy_(node_a).detach()
        self.node_b.resize_(node_b.size()).copy_(node_b).detach()
        self.P.resize_(P.size()).copy_(P).detach()
        self.img.resize_(img.size()).copy_(img).detach()
        self.K.resize_(K.size()).copy_(K).detach()

    def forward(self,
                pc, intensity, sn, node_a, node_b,
                img):
        return self.detector(pc, intensity, sn, node_a, node_b, img)

    def inference_pass(self):
        N = self.pc.size(2)
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)

        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>
        # BLx2xN, BLxKxN
        coarse_scores = self.forward(self.pc, self.intensity, self.sn, self.node_a, self.node_b,
                                                  self.img)
        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>

        _, coarse_prediction = torch.max(coarse_scores, dim=1, keepdim=False)  # BLxN
        return coarse_prediction

    def foraward_pass(self):
        N = self.pc.size(2)
        Ma, Mb = self.opt.node_a_num, self.opt.node_b_num
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)

        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>
        # Bx2xN, BxLxN
        coarse_scores = self.forward(self.pc, self.intensity, self.sn, self.node_a, self.node_b,
                                                  self.img)
        # =================>>>>>>>>>>>>>>>>>> network feed forward ==================>>>>>>>>>>>>>>>>>>

        # project points onto image to get ground truth labels for both coarse and fine resolution
        pc_homo = torch.cat((self.pc,
                             torch.ones((B, 1, N), dtype=torch.float32, device=self.pc.device)),
                            dim=1)  # Bx4xN
        P_pc_homo = torch.matmul(self.P, pc_homo)  # Bx3xN
        KP_pc_homo = torch.matmul(self.K, P_pc_homo)  # Bx3xN
        KP_pc_pxpy = KP_pc_homo[:, 0:2, :] / KP_pc_homo[:, 2:3, :]  # Bx2xN

        x_inside_mask = (KP_pc_pxpy[:, 0:1, :] >= 0) \
                        & (KP_pc_pxpy[:, 0:1, :] <= W - 1)  # Bx1xN
        y_inside_mask = (KP_pc_pxpy[:, 1:2, :] >= 0) \
                        & (KP_pc_pxpy[:, 1:2, :] <= H - 1)  # Bx1xN
        z_inside_mask = KP_pc_homo[:, 2:3, :] > 0.1  # Bx1xN
        inside_mask = (x_inside_mask & y_inside_mask & z_inside_mask).squeeze(1)  # BxN

        # get coarse labels
        coarse_labels = inside_mask.to(dtype=torch.long)  # BxN

        # build loss -------------------------------------------------------------------------------------
        coarse_loss = self.coarse_ce_criteria(coarse_scores, coarse_labels) * self.opt.coarse_loss_alpha
        loss = coarse_loss
        # build loss -------------------------------------------------------------------------------------

        # get accuracy
        # get predictions for visualization
        _, coarse_predictions = torch.max(coarse_scores, dim=1, keepdim=False)
        coarse_accuracy = torch.sum(torch.eq(coarse_labels, coarse_predictions).to(dtype=torch.float)) / (B * N)

        loss_dict = {'loss': loss,
                     'coarse': coarse_loss}
        vis_dict = {'pc': P_pc_homo,
                    'coarse_labels': coarse_labels,
                    'coarse_predictions': coarse_predictions,
                    'KP_pc_pxpy': KP_pc_pxpy}
        accuracy_dict = {'coarse_accuracy': coarse_accuracy}

        # debug
        if self.opt.is_debug:
            print(loss_dict)

        return loss_dict, vis_dict, accuracy_dict

    def optimize(self):
        self.detector.train()
        self.detector.zero_grad()
        self.train_loss_dict, self.train_visualization, self.train_accuracy = self.foraward_pass()
        self.train_loss_dict['loss'].backward()
        self.optimizer_detector.step()

    def test_model(self):
        self.detector.eval()
        with torch.no_grad():
            self.test_loss_dict, self.test_visualization, self.test_accuracy = self.foraward_pass()

    def freeze_model(self):
        print("!!! WARNING: Model freezed.")
        for p in self.detector.parameters():
            p.requires_grad = False

    def run_model(self, pc, intensity, sn, node, img):
        self.detector.eval()
        with torch.no_grad():
            raise Exception("Not implemented.")

    def tensor_dict_to_float_dict(self, tensor_dict):
        float_dict = {}
        for key, value in tensor_dict.items():
            if type(value) == torch.Tensor:
                float_dict[key] = value.item()
            else:
                float_dict[key] = value
        return float_dict

    def get_current_errors(self):
        return self.tensor_dict_to_float_dict(self.train_loss_dict), \
               self.tensor_dict_to_float_dict(self.test_loss_dict)

    def get_current_accuracy(self):
        return self.tensor_dict_to_float_dict(self.train_accuracy), \
               self.tensor_dict_to_float_dict(self.test_accuracy)

    @staticmethod
    def print_loss_dict(loss_dict, accuracy_dict=None, duration=-1):
        output = 'Per sample time: %.3f - ' % (duration)
        for key, value in loss_dict.items():
            output += '%s: %.2f, ' % (key, value)
        if accuracy_dict is not None:
            for key, value in accuracy_dict.items():
                output += '%s: %.2f, ' % (key, value)
        print(output)

    def save_network(self, network, save_filename):
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def update_learning_rate(self, ratio):
        lr_clip = 0.00001

        # detector
        lr_detector = self.old_lr_detector * ratio
        if lr_detector < lr_clip:
            lr_detector = lr_clip
        for param_group in self.optimizer_detector.param_groups:
            param_group['lr'] = lr_detector
        print('update detector learning rate: %f -> %f' % (self.old_lr_detector, lr_detector))
        self.old_lr_detector = lr_detector

    # visualization with tensorboard, pytorch 1.2
    def write_loss(self):
        train_loss_dict, test_loss_dict = self.get_current_errors()
        self.writer.add_scalars('train_loss',
                                train_loss_dict,
                                global_step=self.global_step)
        self.writer.add_scalars('test_loss',
                                test_loss_dict,
                                global_step=self.global_step)

    def write_accuracy(self):
        train_acc_dict, test_acc_dict = self.get_current_accuracy()
        self.writer.add_scalars('train_accuracy',
                                train_acc_dict,
                                global_step=self.global_step)
        self.writer.add_scalars('test_accuracy',
                                test_acc_dict,
                                global_step=self.global_step)

    def write_pc_label(self, pc, labels, title):
        """
        :param pc: Bx3xN
        :param labels: BxN
        :param title: string
        :return:
        """
        pc_np = pc.detach().cpu().numpy()  # Bx3xN
        labels_np = labels.detach().cpu().numpy()  # BxN

        B = pc_np.shape[0]

        visualization_fig_np_list = []
        vis_number = min(self.opt.vis_max_batch, B)
        for b in range(vis_number):
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(pc_np[b, 0, :], pc_np[b, 2, :],
                        c=labels_np[b, :],
                        cmap='jet',
                        marker='o',
                        s=1)
            fig.tight_layout()
            plt.axis('equal')
            fig_img_np = vis_tools.fig_to_np(fig)
            plt.close(fig)
            visualization_fig_np_list.append(fig_img_np)

        imgs_vis_grid = vis_tools.visualization_list_to_grid(visualization_fig_np_list, col=2)
        imgs_vis_grid = np.moveaxis(imgs_vis_grid, (0, 1, 2), (1, 2, 0))
        self.writer.add_image(title, imgs_vis_grid, global_step=self.global_step)

    def write_img(self):
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)
        imgs = self.img.detach().round().to(dtype=torch.uint8).cpu()  # Bx3xHxW

        vis_number = min(self.opt.vis_max_batch, B)
        imgs_vis = imgs[0:vis_number, ...]  # Bx3xHxW
        imgs_vis_grid = torchvision.utils.make_grid(imgs_vis, nrow=2)
        self.writer.add_image('image', imgs_vis_grid, global_step=self.global_step)

    def write_classification_visualization(self,
                                           pc_pxpy,
                                           coarse_predictions,
                                           coarse_labels,
                                           t_ij):
        B, H, W = self.img.size(0), self.img.size(2), self.img.size(3)
        imgs = self.img.detach().round().to(dtype=torch.uint8).cpu()  # Bx3xHxW
        imgs_np = imgs.permute(0, 2, 3, 1).numpy()  # BxHxWx3

        pc_pxpy_np = pc_pxpy.cpu().numpy()  # Bx2xN
        coarse_predictions_np = coarse_predictions.detach().cpu().numpy()  # BxN
        coarse_labels_np = coarse_labels.detach().cpu().numpy()  # BxN

        t_ij_np = t_ij.cpu().numpy()  # Bx3

        vis_number = min(self.opt.vis_max_batch, B)
        visualization_fig_np_list = []
        for b in range(vis_number):
            img_b = imgs_np[b, ...]  # HxWx3
            pc_pxpy_b = pc_pxpy_np[b, ...]  # 2xN
            coarse_prediction_b = coarse_predictions_np[b, :]  # N
            coarse_labels_b = coarse_labels_np[b, :] # N

            vis_img = vis_tools.get_classification_visualization_coarse(pc_pxpy_b,
                                                                 coarse_prediction_b,
                                                                 coarse_labels_b,
                                                                 img_b,
                                                                 H_delta=100, W_delta=100,
                                                                 circle_size=1,
                                                                 t_ij_np=t_ij_np[b, :])
            visualization_fig_np_list.append(vis_img)

        imgs_vis_grid = vis_tools.visualization_list_to_grid(visualization_fig_np_list, col=2)
        imgs_vis_grid = np.moveaxis(imgs_vis_grid, (0, 1, 2), (1, 2, 0))
        self.writer.add_image("Classification Visualization", imgs_vis_grid, global_step=self.global_step)
