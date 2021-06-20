import numpy as np
import open3d
import os
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial import cKDTree
import struct
from multiprocessing import Process
import time
import math

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from util import vis_tools
from data.kitti_helper import *


def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*4
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2], point[3]])
    return np.asarray(pc_list, dtype=np.float32).T


def process_kitti(root_path,
                  pc_bin_root_path,
                  seq_list,
                  downsample_voxel_size,
                  sn_radius,
                  sn_max_nn):
    calib_helper = KittiCalibHelper(root_path)

    for seq in seq_list:
        pc_bin_folder = os.path.join(pc_bin_root_path, 'sequences', '%02d' % seq, 'velodyne')

        sample_num = round(len(os.listdir(pc_bin_folder)))
        for i in range(sample_num):
            # show progress
            print('sequence %d: %d/%d' % (seq, i, sample_num))

            data_np = read_velodyne_bin(os.path.join(pc_bin_folder, '%06d.bin' % i))
            pc_np = data_np[0:3, :]
            intensity_np = data_np[3:, :]

            # compute surface normal and downsample -----------------------------------
            # convert to Open3D point cloud datastructure
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(pc_np.T)
            downpcd = open3d.geometry.voxel_down_sample(pcd, voxel_size=downsample_voxel_size)

            # surface normal computation
            open3d.geometry.estimate_normals(downpcd,
                                             search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=sn_radius,
                                                                                                  max_nn=sn_max_nn))
            open3d.geometry.orient_normals_to_align_with_direction(downpcd, [0, 0, 1])
            # open3d.visualization.draw_geometries([downpcd])

            # get numpy array from pcd
            pc_down_np = np.asarray(downpcd.points).T
            pc_down_sn_np = np.asarray(downpcd.normals).T

            # get intensity through 1-NN between downsampled pc and original pc
            kdtree = cKDTree(pc_np.T)
            D, I = kdtree.query(pc_down_np.T, k=1)
            intensity_down_np = intensity_np[:, I]
            # compute surface normal and downsample -----------------------------------

            # covert to image frame
            for img_key in ['image_2', 'image_3']:
                if img_key == 'image_2':
                    img_key_P = 'P2'
                elif img_key == 'image_3':
                    img_key_P = 'P3'
                else:
                    assert False

                # convert pc_down_np, intensity_down_np, pc_down_sn_np into image frame, and save to disk
                pc_npy_folder = os.path.join(root_path,
                                             'data_odometry_velodyne_%s_npy' % img_key_P,
                                             'sequences', '%02d' % seq,
                                             'voxel%.1f-SNr%.1f' % (downsample_voxel_size, sn_radius))
                if not os.path.isdir(pc_npy_folder):
                    os.makedirs(pc_npy_folder)
                pc_down_img_np = calib_helper.transform_pc_vel_to_img(pc_down_np, seq, img_key_P)
                pc_down_sn_img_np = coordinate_NWU_to_cam(pc_down_sn_np)
                pc_output_np = np.concatenate((pc_down_img_np,
                                               intensity_down_np,
                                               pc_down_sn_img_np), axis=0).astype(np.float32)
                np.save(os.path.join(pc_npy_folder, '%06d.npy' % i), pc_output_np)

                # debug
                # vis_tools.plot_pc(pc_down_np, size=1, color=intensity_down_np[0, :])
                # plt.show()
                # plt.waitforbuttonpress()

                # project pc_np onto image to get the depth map & dist_to_nearest_lidar_point_map, float32
                # ------------------------------
                img_folder = os.path.join(root_path, 'data_odometry_color_npy', 'sequences', '%02d' % seq, img_key)
                K = calib_helper.get_matrix(seq, img_key_P + '_K')  # 3x3

                # load image
                img_path = os.path.join(img_folder, '%06d.npy' % i)
                img = np.load(img_path)
                H, W = img.shape[0], img.shape[1]
                img_depth = np.zeros((H, W), dtype=np.float32)
                img_depth_dist_to_nn_pc = np.zeros((H, W), dtype=np.float32) + H*W  # large init distance
                img_depth_mask = np.zeros((H, W), dtype=np.bool)

                P_pc_homo = calib_helper.transform_pc_vel_to_img(pc_np, seq, img_key_P)
                KP_pc_homo = np.dot(K, P_pc_homo)  # 3xM
                KP_pc_pxpy = KP_pc_homo[0:2, :] / KP_pc_homo[2:3, :]  # 2xN

                for j in range(KP_pc_pxpy.shape[1]):
                    px = KP_pc_pxpy[0, j]
                    py = KP_pc_pxpy[1, j]
                    px_round = int(round(px))
                    py_round = int(round(py))
                    pz = P_pc_homo[2, j]

                    if pz>0.1 and px_round >= 0 and px_round <= W-1 and py_round >= 0 and py_round <= H-1:
                        # can be projected into image plane for NN search
                        dx = px - float(px_round)
                        dy = py - float(py_round)
                        dist_to_nn_pc = math.sqrt(dx*dx + dy*dy)
                        if dist_to_nn_pc < img_depth_dist_to_nn_pc[py_round, px_round]:
                            img_depth[py_round, px_round] = pz
                            img_depth_dist_to_nn_pc[py_round, px_round] = dist_to_nn_pc
                            img_depth_mask[py_round, px_round] = True

                # edt for pixels that are not filled with depth
                img_depth_mask_inv = np.logical_not(img_depth_mask)
                edt_val, edt_idx = distance_transform_edt(img_depth_mask_inv, return_indices=True, return_distances=True)

                img_depth_dist_to_nn_pc = img_depth_mask * img_depth_dist_to_nn_pc + img_depth_mask_inv * edt_val
                img_depth = img_depth[edt_idx[0], edt_idx[1]]

                img_depth_output = np.stack((img_depth, img_depth_dist_to_nn_pc), axis=0).astype(np.float32)  # 2xHxW

                # save to disk
                img_depth_folder = os.path.join(root_path, 'data_odometry_color_depth_npy', 'sequences', '%02d' % seq, img_key)
                if not os.path.isdir(img_depth_folder):
                    os.makedirs(img_depth_folder)
                np.save(os.path.join(img_depth_folder, '%06d.npy' % i), img_depth_output)

                # debug
                # print(img_depth)
                # depth_vis_threshold = 40
                # img_depth_vis_mask = img_depth < depth_vis_threshold
                # img_depth_vis = img_depth * img_depth_vis_mask + depth_vis_threshold * np.logical_not(img_depth_vis_mask)
                # img_depth_vis = img_depth_vis * (255 / depth_vis_threshold)
                # img_depth_vis = img_depth_vis.astype(np.uint8)
                # plt.figure()
                # plt.imshow(img_depth_vis, cmap='gray', vmin=0, vmax=255)
                #
                # print(img_depth_dist_to_nn_pc)
                # depth_nn_vis_threshold = 60
                # img_depth_nn_vis_mask = img_depth_dist_to_nn_pc < depth_nn_vis_threshold
                # img_depth_nn_vis = img_depth_dist_to_nn_pc * img_depth_nn_vis_mask + depth_nn_vis_threshold * np.logical_not(img_depth_nn_vis_mask)
                # img_depth_nn_vis = img_depth_nn_vis * (255 / depth_nn_vis_threshold)
                # img_depth_nn_vis = img_depth_nn_vis.astype(np.uint8)
                # plt.figure()
                # plt.imshow(img_depth_nn_vis, cmap='gray', vmin=0, vmax=255)
                #
                # plt.figure()
                # plt.imshow(img)
                # plt.show()

                # ------------------------------
            # for i in range(sample_num):
            # break
        # for seq in seq_list:


if __name__ == '__main__':
    root_path = '/ssd/jiaxin/datasets/kitti'
    pc_bin_root_path = '/data/datasets/data_odometry_velodyne/dataset'
    seq_list = list(range(22))

    downsample_voxel_size = 0.1
    sn_radius = 0.6
    sn_max_nn = 30

    # process_kitti(root_path,
    #               pc_bin_root_path,
    #               seq_list,
    #               downsample_voxel_size,
    #               sn_radius,
    #               sn_max_nn)

    thread_num = 22  # One thread for one folder
    kitti_threads = []
    for i in range(thread_num):
        thread_seq_list = [i]
        kitti_threads.append(Process(target=process_kitti,
                                     args=(root_path,
                                           pc_bin_root_path,
                                           thread_seq_list,
                                           downsample_voxel_size,
                                           sn_radius,
                                           sn_max_nn)))

    for thread in kitti_threads:
        thread.start()

    for thread in kitti_threads:
        thread.join()

