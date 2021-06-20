import open3d
import numpy as np
import math
import random
from scipy.spatial.transform import Rotation
import os
import torch

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


import kitti.options
import oxford.options
import nuscenes_t.options
from data.kitti_pc_img_pose_loader import KittiLoader
from data.oxford_pc_img_pose_loader import OxfordLoader
from data.nuscenes_pc_img_pose_loader import nuScenesLoader

from util import vis_tools


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


if __name__ == '__main__':
    # root_path = '/ssd/jiaxin/point-img-feature/kitti/save/1.17-accu-node128'
    root_path = '/ssd/jiaxin/point-img-feature/oxford/save/1.16-fine-wGround-nocrop-0.5x384x640'
    # root_path = '/ssd/jiaxin/point-img-feature/nuscenes_t/save/3.3-160x320-accu'
    visualization_output_folder = 'visualization'
    visualization_output_path = os.path.join(root_path, visualization_output_folder)
    data_output_folder = 'data'
    data_output_path = os.path.join(root_path, data_output_folder)

    is_plot = False
    batch_size = 8
    H = 160  # kitti=160, oxford=288/192/384, nuscenes 160
    W = 320  # kitti=512, oxford=512/320/640, nuscenes 320
    fine_resolution_scale = 1 / 32.0

    batch_id = 605
    item_id = 4
    filename = '%06d_%02d' % (batch_id, item_id)

    if 'kitti' in root_path:
        dataset = 'kitti'
        opt = kitti.options.Options()
    elif 'oxford' in root_path:
        dataset = 'oxford'
        opt = oxford.options.Options()
    elif 'nuscenes' in root_path:
        dataset = 'nuscenes'
        opt = nuscenes_t.options.Options()
    opt.batch_size = batch_size

    if dataset == 'kitti':
        testset = KittiLoader(opt.dataroot, 'val', opt)
    elif dataset == 'oxford':
        testset = OxfordLoader(opt.dataroot, 'val', opt)
    elif dataset == 'nuscenes':
        testset = nuScenesLoader(opt.dataroot, 'val', opt)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False,
    #                                          num_workers=opt.dataloader_threads, pin_memory=True)
    print('#testing point clouds = %d' % len(testset))



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

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.transpose(pc_np[0:3, :]))

    # height coloring
    # pc_np_y_min, pc_np_y_max = np.min(pc_np[1, :]), np.max(pc_np[1, :])
    # pc_np_y_interval = pc_np_y_max - pc_np_y_min
    # height_normalized = (pc_np[1, :] - pc_np_y_min) / pc_np_y_interval
    # height_normalized = np.clip(height_normalized, 0, 1)
    # height_colors = plt.cm.jet(height_normalized)
    # print(height_colors.shape)

    # label coloring
    tp_mask = np.logical_and(coarse_predictions_np == coarse_labels_np, coarse_predictions_np==1)  # green
    fp_mask = np.logical_and(coarse_predictions_np==1, coarse_labels_np==0)  # blue
    fn_mask = np.logical_and(coarse_predictions_np==0, coarse_labels_np==1)  # red
    other_mask = np.logical_not(np.logical_or(np.logical_or(tp_mask, fp_mask), fn_mask))


    color_np = np.zeros((pc_np.shape[1], 3))
    color_np[tp_mask, 1] = 1
    color_np[fp_mask, 2] = 1
    color_np[fn_mask, 0] = 1
    color_np[other_mask, :] = np.asarray([0.5, 0.5, 0.5])
    pcd.colors = open3d.utility.Vector3dVector(color_np)

    open3d.visualization.draw_geometries([pcd])


    # fig_gt = plt.figure(figsize=(9, 9))
    # plt.axis('off')
    # plt.grid(b=None)
    # ax_gt = Axes3D(fig_gt)
    # ax_gt.set_title("registration")
    # ax_gt.axis('off')
    # ax_gt.grid(b=None)
    # vis_tools.plot_pc(pc_np, color=coarse_labels_np, size=6, ax=ax_gt,
    #                   elev=0, azim=-90)
    #
    # plt.show()




