import open3d
import time
import numpy as np
import math
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import cv2
import random

import matplotlib
matplotlib.use('TkAgg')

from models.multimodal_classifier import MMClassifer
from data.kitti_pc_img_pose_loader import KittiLoader
from kitti import options
from util import vis_tools
from data import augmentation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

import FrustumRegistration


def generate_random_transform(P_tx_amplitude, P_ty_amplitude, P_tz_amplitude,
                              P_Rx_amplitude, P_Ry_amplitude, P_Rz_amplitude):
    """

    :param pc_np: pc in NWU coordinate
    :return:
    """
    t = [random.uniform(-P_tx_amplitude, P_tx_amplitude),
         random.uniform(-P_ty_amplitude, P_ty_amplitude),
         random.uniform(-P_tz_amplitude, P_tz_amplitude)]
    angles = [random.uniform(-P_Rx_amplitude, P_Rx_amplitude),
              random.uniform(-P_Ry_amplitude, P_Ry_amplitude),
              random.uniform(-P_Rz_amplitude, P_Rz_amplitude)]

    rotation_mat = augmentation.angles2rotation_matrix(angles)
    P_random = np.identity(4, dtype=np.float)
    P_random[0:3, 0:3] = rotation_mat
    P_random[0:3, 3] = t

    return P_random


def get_inside_img_mask(points_np, P_np, K_np, H, W):
    # get the visualization based on P
    points_np_homo = np.concatenate((points_np,
                                     np.ones((1, points_np.shape[1]), dtype=points_np.dtype)),
                                    axis=0)
    P_points_np = np.dot(P_np, points_np_homo)[0:3, :]
    K_pc_np = np.dot(K_np, P_points_np)
    pc_pxpy_np = K_pc_np[0:2, :] / K_pc_np[2:3, :]  # Bx3xN -> Bx2xN

    # compute ground truth
    x_inside_mask = np.logical_and(pc_pxpy_np[0:1, :] >= 0,
                                   pc_pxpy_np[0:1, :] <= W - 1)  # Bx1xN_pc
    y_inside_mask = np.logical_and(pc_pxpy_np[1:2, :] >= 0,
                                   pc_pxpy_np[1:2, :] <= H - 1)  # Bx1xN_pc
    z_inside_mask = P_points_np[2:3, :] > 0.1  # Bx1xN_pc
    inside_mask = np.logical_and(np.logical_and(x_inside_mask, y_inside_mask),
                                 z_inside_mask)  # Bx1xN_pc
    return inside_mask[0]


if __name__=='__main__':
    root_path = '/ssd/jiaxin/point-img-feature/kitti/save/1.3-odometry'
    visualization_output_folder = 'visualization'
    visualization_output_path = os.path.join(root_path, visualization_output_folder)
    data_output_folder = 'data'
    data_output_path = os.path.join(root_path, data_output_folder)

    is_plot = False
    H = 160
    W = 512

    filename_list = [f[0:9] for f in os.listdir(data_output_path) if os.path.isfile(os.path.join(data_output_path, f))]
    filename_list = list(set(filename_list))
    filename_list.sort()
    for i, filename in enumerate(filename_list):
        if i<0:
            continue

        point_data_np = np.loadtxt(os.path.join(data_output_path, filename+'_pc_label.txt'))
        points_np = point_data_np[0:3, :].astype(np.float64)
        labels_np = point_data_np[3, :]
        K_np = np.loadtxt(os.path.join(data_output_path, filename + '_K.txt')).astype(np.float64)
        P_gt_np = np.loadtxt(os.path.join(data_output_path, filename + '_P.txt')).astype(np.float64)

        # trick to help optimization
        # mask = points_np[2, :]>0
        # points_np = points_np[:, mask]
        # labels_np = labels_np[mask]

        inside_mask = get_inside_img_mask(points_np, P_gt_np, K_np, H, W)
        inside_mask = inside_mask.astype(np.int32)
        print(inside_mask.shape)
        print(inside_mask == labels_np)
        labels_np = inside_mask

        P_random_init = generate_random_transform(0, 0, 0,
                                                  0, 0.2*math.pi, 0)
        P = FrustumRegistration.solvePGivenK(points_np,
                                             labels_np.astype(np.int32),
                                             K_np,
                                             # P_gt_np[0:3, 0:3],
                                             # P_gt_np[0:3, 3],
                                             P_random_init[0:3, 0:3],
                                             P_random_init[0:3, 3],
                                             160,
                                             512,
                                             True)
        print(P)
        print(P_gt_np)



        if is_plot:
            img_vis_np = cv2.cvtColor(cv2.imread(os.path.join(visualization_output_path, filename+'_img.png')), cv2.COLOR_BGR2RGB)
            img_vis_fine_np = cv2.cvtColor(cv2.imread(os.path.join(visualization_output_path, filename+'_prediction.png')), cv2.COLOR_BGR2RGB)

            plt.figure()
            plt.imshow(img_vis_np)
            plt.figure()
            plt.imshow(img_vis_fine_np)

            fig_prediction = plt.figure(figsize=(9, 9))
            ax_prediction = Axes3D(fig_prediction)
            ax_prediction.set_title("coarse label")
            vis_tools.plot_pc(points_np, color=labels_np, size=6, ax=ax_prediction)

            # fig_gt = plt.figure(figsize=(9, 9))
            # ax_gt = Axes3D(fig_gt)
            # ax_gt.set_title("registration")
            # vis_tools.plot_pc(pc_vis_np, color=coarse_label_vis_np, size=6, ax=ax_gt)

            plt.show()

        break