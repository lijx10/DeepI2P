import numpy as np
import math
import torch


class Options:
    def __init__(self):
        self.dataroot = '/ssd/jiaxin/datasets/kitti'
        # self.dataroot = '/data/personal/jiaxin/datasets/kitti'
        self.checkpoints_dir = 'checkpoints'
        self.version = '1.27'
        self.is_debug = False
        self.is_fine_resolution = True
        self.is_remove_ground = False
        self.accumulation_frame_num = 3
        self.accumulation_frame_skip = 6

        self.delta_ij_max = 40
        self.translation_max = 10.0

        self.crop_original_top_rows = 50
        self.img_scale = 0.5
        self.img_H = 160  # 320 * 0.5
        self.img_W = 512  # 1224 * 0.5
        # the fine resolution is img_H/scale x img_W/scale
        self.img_fine_resolution_scale = 32

        self.input_pt_num = 20480
        self.pc_min_range = -1.0
        self.pc_max_range = 80.0
        self.node_a_num = 128
        self.node_b_num = 128
        self.k_ab = 16
        self.k_interp_ab = 3
        self.k_interp_point_a = 3
        self.k_interp_point_b = 3

        # CAM coordinate
        self.P_tx_amplitude = 0
        self.P_ty_amplitude = 0
        self.P_tz_amplitude = 0
        self.P_Rx_amplitude = 0.0 * math.pi / 12.0
        self.P_Ry_amplitude = 2.0 * math.pi
        self.P_Rz_amplitude = 0.0 * math.pi / 12.0
        self.dataloader_threads = 10

        self.batch_size = 8
        self.gpu_ids = [1]
        self.device = torch.device('cuda', self.gpu_ids[0])
        self.normalization = 'batch'
        self.norm_momentum = 0.1
        self.activation = 'relu'
        self.lr = 0.001
        self.lr_decay_step = 20
        self.lr_decay_scale = 0.5
        self.vis_max_batch = 4
        if self.is_fine_resolution:
            self.coarse_loss_alpha = 50
        else:
            self.coarse_loss_alpha = 1




