import numpy as np
import os
import math
import open3d
import bisect
import multiprocessing

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from util import vis_tools
from data import kitti_helper


def downsample_with_intensity_sn(pointcloud, intensity, sn, voxel_grid_downsample_size):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.transpose(pointcloud[0:3, :]))
    intensity_max = np.max(intensity)

    fake_colors = np.zeros((pointcloud.shape[1], 3))
    fake_colors[:, 0:1] = np.transpose(intensity) / intensity_max

    pcd.colors = open3d.utility.Vector3dVector(fake_colors)
    pcd.normals = open3d.utility.Vector3dVector(np.transpose(sn))

    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_grid_downsample_size)
    down_pcd_points = np.transpose(np.asarray(down_pcd.points))  # 3xN
    pointcloud = down_pcd_points

    intensity = np.transpose(np.asarray(down_pcd.colors)[:, 0:1]) * intensity_max
    sn = np.transpose(np.asarray(down_pcd.normals))

    return pointcloud, intensity, sn


def transform_pc(pc, P):
    """

    :param pc: 3xN
    :param P: 4x4
    :return: P_pc: 3xN
    """
    pc_homo = np.concatenate((pc,
                              np.ones((1, pc.shape[1]), dtype=pc.dtype)),
                             axis=0)
    P_pc_homo = np.dot(P, pc_homo)
    return P_pc_homo[0:3, :]  # 3xN

def accumulate_sequence(root_path, seq, Tr,
                        frame_stride_dist=4, accumulate_radius=50, voxel_size=0.2,
                        is_plot=True):
    np_folder = 'voxel0.1-SNr0.6'
    output_folder = 'stride%d-acc%d-voxel%.1f' % (frame_stride_dist, accumulate_radius,voxel_size)
    output_folder_path = os.path.join(root_path, 'data_odometry_velodyne_NWU', 'sequences',
                                      '%02d' % seq, output_folder)
    if os.path.exists(output_folder_path):
        print('%s exists, skip' % output_folder_path)
        return
    else:
        os.mkdir(output_folder_path)

    pc_nwu_folder = os.path.join(root_path, 'data_odometry_velodyne_NWU', 'sequences', '%02d' % seq, np_folder)
    pose_folder = os.path.join(root_path, 'poses', '%02d' % seq)
    sample_num = round(len(os.listdir(pc_nwu_folder)))

    accumulate_frames = int(round(accumulate_radius / frame_stride_dist))
    Tr_inv = np.linalg.inv(Tr)

    # build per frame_stride_dist poses
    stride_list = [0]
    P_prev = np.load(os.path.join(pose_folder, '%06d.npz' % 0))['pose']  # 4x4
    for i in range(1, sample_num):
        P_i = np.load(os.path.join(pose_folder, '%06d.npz' % i))['pose']  # 4x4
        P_prev_i = np.dot(np.linalg.inv(P_prev), P_i)  # 4x4
        t_prev_i_norm = np.linalg.norm(P_prev_i[0:3, 3])  # scalar
        if t_prev_i_norm > frame_stride_dist:
            stride_list.append(i)
            P_prev = P_i

    print('%02d stride_list length: %d' % (seq, len(stride_list)))


    for i in range(0, sample_num):
        Pi = np.load(os.path.join(pose_folder, '%06d.npz' % i))['pose']  # 4x4
        Pi_inv = np.linalg.inv(Pi)

        stride_list_nearest_idx = bisect.bisect_left(stride_list, i)
        stride_list_left_idx = max(0, stride_list_nearest_idx-accumulate_frames)
        stride_list_right_idx = min(len(stride_list)-1, stride_list_nearest_idx+accumulate_frames)

        pc_nwu_list = []
        intensity_list = []
        sn_list = []
        for stride in stride_list[stride_list_left_idx:stride_list_right_idx+1]:
            Pj = np.load(os.path.join(pose_folder, '%06d.npz' % stride))['pose']  # 4x4
            data_nwu = np.load(os.path.join(pc_nwu_folder, '%06d.npy' % stride))

            # remove point falls on egocar
            x_inside = np.logical_and(data_nwu[0, :] < 2.7, data_nwu[0, :] > -2.7 * 0.75)
            y_inside = np.logical_and(data_nwu[1, :] < 0.8, data_nwu[1, :] > -0.8)
            inside_mask = np.logical_and(x_inside, y_inside)
            outside_mask = np.logical_not(inside_mask)
            data_nwu = data_nwu[:, outside_mask]

            pc_nwu = data_nwu[0:3, :]  # 3xN
            intensity = data_nwu[3:4, :]  # 1xN
            sn = data_nwu[4:7, :]  # 3xN

            Pij = np.dot(Pi_inv, Pj)
            P_transform = np.dot(Tr_inv, np.dot(Pij, Tr))

            pc_nwu_list.append(transform_pc(pc_nwu, P_transform))
            intensity_list.append(intensity)
            sn_list.append(transform_pc(sn, P_transform))

            # assemble the final point cloud at frame i
        pc_nwu = np.concatenate(pc_nwu_list, axis=1)
        intensity = np.concatenate(intensity_list, axis=1)
        sn = np.concatenate(sn_list, axis=1)

        pc_nwu, intensity, sn = downsample_with_intensity_sn(pc_nwu, intensity, sn,
                                                             voxel_grid_downsample_size=voxel_size)
        range_nwu = np.linalg.norm(pc_nwu[0:2, :], axis=0)
        range_mask = range_nwu < accumulate_radius
        pc_nwu = pc_nwu[:, range_mask]
        intensity = intensity[:, range_mask]
        sn = sn[:, range_mask]

        output_np = np.concatenate((pc_nwu, intensity, sn), axis=0).astype(np.float32)
        output_np_path = os.path.join(output_folder_path, '%06d.npy' % i)
        np.save(output_np_path,
                output_np)
        print('%s: %dx%d' % (output_np_path, output_np.shape[0], output_np.shape[1]))

        if is_plot:
            vis_tools.plot_pc(pc_nwu)
            plt.show()

def main():
    root_path = '/ssd/jiaxin/datasets/kitti'
    calib_helper = kitti_helper.KittiCalibHelper(root_path)

    is_plot = False
    frame_stride_dist = 4
    accumulate_radius = 50
    voxel_size = 0.4


    threads = []
    for seq in range(0, 11):
        Tr = calib_helper.get_matrix(seq, 'Tr')
        threads.append(multiprocessing.Process(target=accumulate_sequence,
                                               args=(root_path,
                                                     seq,
                                                     Tr,
                                                     frame_stride_dist,
                                                     accumulate_radius,
                                                     voxel_size,
                                                     is_plot)))

    for thread in threads:
        thread.start()

    for i, thread in enumerate(threads):
        thread.join()




if __name__ == '__main__':
    main()