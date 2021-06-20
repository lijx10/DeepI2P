import random
import numbers
import os
import os.path
import numpy as np
import struct
import math

import torch
import torchvision
import matplotlib.pyplot as plt


def angles2rotation_matrix(angles):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R


def rotate_pc(pc_np, angles):
    '''

    :param pc_np: numpy array of 3xN array
    :param angles: numpy array / list of 3
    :return: rotated_data: numpy array of 3xN
    '''
    R = angles2rotation_matrix(angles)
    rotated_pc_np = np.dot(R, pc_np)

    return rotated_pc_np


def jitter_point_cloud(pc_np, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          CxN array, original point clouds
        Return:
          CxN array, jittered point clouds
    """
    C, N = pc_np.shape
    assert(clip > 0)
    jittered_pc = np.clip(sigma * np.random.randn(C, N), -1*clip, clip).astype(pc_np.dtype)
    jittered_pc += pc_np
    return jittered_pc


def coordinate_cam_to_NWU(pc_np):
    assert pc_np.shape[0] == 3
    pc_nwu_np = np.copy(pc_np)
    pc_nwu_np[0, :] = pc_np[2, :]  # x <- z
    pc_nwu_np[1, :] = -pc_np[0, :]  # y <- -x
    pc_nwu_np[2, :] = -pc_np[1, :]  # z <- -y
    return pc_nwu_np


def coordinate_NWU_to_cam(pc_np):
    assert pc_np.shape[0] == 3
    pc_cam_np = np.copy(pc_np)
    pc_cam_np[0, :] = -pc_np[1, :]  # x <- -y
    pc_cam_np[1, :] = -pc_np[2, :]  # y <- -z
    pc_cam_np[2, :] = pc_np[0, :]  # z <- x
    return pc_cam_np


def coordinate_ENU_to_cam(pc_np):
    assert pc_np.shape[0] == 3
    pc_cam_np = np.copy(pc_np)
    pc_cam_np[0, :] = pc_np[0, :]  # x <- x
    pc_cam_np[1, :] = -pc_np[2, :]  # y <- -z
    pc_cam_np[2, :] = pc_np[1, :]  # z <- y
    return pc_cam_np


