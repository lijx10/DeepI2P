import open3d
import time
import numpy as np
import math
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import cv2
import random
from scipy.spatial.transform import Rotation
import multiprocessing

import matplotlib
matplotlib.use('TkAgg')

from models.multimodal_classifier import MMClassifer
from data.kitti_pc_img_pose_loader import KittiLoader
from data.augmentation import angles2rotation_matrix
from kitti import options
from util import vis_tools
from data import augmentation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

import FrustumRegistration


def transform_pc_np(P, pc_np):
    """

    :param pc_np: 3xN
    :param P: 4x4
    :return:
    """
    pc_homo_np = np.concatenate((pc_np,
                                 np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)),
                                axis=0)
    P_pc_homo_np = np.dot(P, pc_homo_np)
    return P_pc_homo_np[0:3, :]


def enu2cam(pc_np, P):
    """

    :param pc_np: 3xN
    :param P: 4x4
    :return:
    """
    P_convert = np.asarray([[1, 0, 0, 0],
                            [0, 0, -1,0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]], dtype=P.dtype)
    return transform_pc_np(P_convert, pc_np), np.dot(P, np.linalg.inv(P_convert))


def get_P_diff(P_pred_np, P_gt_np):
    P_diff = np.dot(np.linalg.inv(P_pred_np), P_gt_np)
    t_diff = np.linalg.norm(P_diff[0:3, 3])

    r_diff = P_diff[0:3, 0:3]
    R_diff = Rotation.from_matrix(r_diff)
    angles_diff = np.sum(np.abs(R_diff.as_euler('xzy', degrees=True)))

    return t_diff, angles_diff


def wrap_in_pi(x):
    x = math.fmod(x+math.pi, math.pi*2)
    if x<0:
        x += math.pi*2
    return x - math.pi


def get_initial_guess(pc_np, coarse_predictions_np):
    """

    :param pc_np: 3xN
    :param coarse_predictions_np: N
    :return:
    """
    pc_np_masked = pc_np[:, coarse_predictions_np==1]
    pc_np_masked_mean = np.mean(pc_np_masked, axis=1)  # 3
    src_mean_point_angle_y = math.atan2(pc_np_masked_mean[2], pc_np_masked_mean[0])
    dst_mean_point_angle_y = math.pi/2
    init_y_angle = wrap_in_pi(src_mean_point_angle_y-dst_mean_point_angle_y)

    R1 = angles2rotation_matrix([0, init_y_angle, 0])
    R1_pc_np = np.dot(R1, pc_np)

    R1_pc_np_min = np.min(R1_pc_np[:, coarse_predictions_np==1], axis=1)  # 3
    front_mask = R1_pc_np[2, :] > R1_pc_np_min[2] - 10
    pc_np_front = pc_np[:, front_mask]
    coarse_predictions_np_front = coarse_predictions_np[front_mask]

    P_init = np.identity(4)
    P_init[0:3, 0:3] = R1

    return P_init, init_y_angle, pc_np_front, coarse_predictions_np_front




def main():
    # root_path = '/ssd/jiaxin/point-img-feature/kitti/save/1.30-noTranslation'
    root_path = '/ssd/jiaxin/point-img-feature/oxford/save/1.16-fine-wGround-nocrop-0.5x384x640'
    # root_path = '/ssd/jiaxin/point-img-feature/nuscenes_t/save/3.3-160x320-accu'

    visualization_output_folder = 'visualization'
    visualization_output_path = os.path.join(root_path, visualization_output_folder)
    data_output_folder = 'data'
    data_output_path = os.path.join(root_path, data_output_folder)

    is_plot = False
    is_2d = True
    H = 384  # kitti=160, oxford=288/192/384, nuscenes 160
    W = 640  # kitti=512, oxford=512/320/640, nuscenes 320
    is_enu2cam = 'nuscene' in root_path

    filename = '001851_07'

    point_data_np = np.load(os.path.join(data_output_path, filename + '_pc_label.npy'))
    pc_np = point_data_np[0:3, :].astype(np.float64)
    coarse_predictions_np = point_data_np[3, :].astype(np.int)
    coarse_labels_np = point_data_np[4, :].astype(np.int)
    fine_predictions_np = point_data_np[5, :].astype(np.int)
    fine_labels_np = point_data_np[6, :].astype(np.int)
    K_np = np.load(os.path.join(data_output_path, filename + '_K.npy')).astype(np.float64)
    P_gt_np = np.load(os.path.join(data_output_path, filename + '_P.npy')).astype(np.float64)
    if P_gt_np.shape[0] == 3:
        P_gt_np = np.concatenate((P_gt_np, np.identity(4)[3:4, :]), axis=0)
    if is_enu2cam:
        pc_np, P_gt_np = enu2cam(pc_np, P_gt_np)

    P_init, init_y_angle, _, _ = get_initial_guess(pc_np, coarse_labels_np)

    P_pred_np, final_cost, residuals = FrustumRegistration.solvePGivenK(pc_np,
                                                                        coarse_labels_np,
                                                                        K_np,
                                                                        init_y_angle - math.pi / 5,
                                                                        np.zeros(3),
                                                                        H,
                                                                        W,
                                                                        [-100, -100, -100],
                                                                        [100, 100, 100],
                                                                        500,
                                                                        True,  # is debug
                                                                        is_2d,  # is_2d
                                                                        )

    print("P_pred_np")
    print(P_pred_np)
    print("P_gt_np")
    print(P_gt_np)
    t_diff, r_diff = get_P_diff(P_pred_np, P_gt_np)
    print('%s - cost: %.1f, T: %.1f, R:%.1f' % (filename, final_cost, t_diff, r_diff))


if __name__ == '__main__':
    main()