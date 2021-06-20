import open3d
import torch.utils.data as data
import random
import numbers
import os
import os.path
import numpy as np
import struct
import math
import torch
import torchvision
import cv2
from PIL import Image
from torchvision import transforms

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from data import augmentation
from util import vis_tools
from kitti import options
from data.kitti_helper import *


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


def clamp(n, smallest, largest): 
    return max(smallest, min(n, largest))


def make_kitti_dataset(root_path, mode, opt):
    dataset = []

    if mode == 'train':
        seq_list = list(range(9))
    elif 'val' in mode:
        seq_list = [9, 10]
    else:
        raise Exception('Invalid mode.')

    np_folder = 'voxel0.1-SNr0.6'
    # np_folder = 'stride4-acc50-voxel0.4'
    skip_start_end = 40
    for seq in seq_list:
        pc_nwu_folder = os.path.join(root_path, 'data_odometry_velodyne_NWU', 'sequences', '%02d' % seq, np_folder)
        img2_folder = os.path.join(root_path, 'data_odometry_color_npy', 'sequences', '%02d' % seq, 'image_2')
        img3_folder = os.path.join(root_path, 'data_odometry_color_npy', 'sequences', '%02d' % seq, 'image_3')
        sample_num = round(len(os.listdir(img2_folder)))

        for i in range(skip_start_end, sample_num-skip_start_end):
            dataset.append((pc_nwu_folder, img2_folder, seq, i, sample_num, 'P2'))
            dataset.append((pc_nwu_folder, img3_folder, seq, i, sample_num, 'P3'))

    return dataset


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


class KittiLoader(data.Dataset):
    def __init__(self, root, mode, opt: options.Options):
        super(KittiLoader, self).__init__()
        self.root = root
        self.opt = opt
        self.mode = mode

        # farthest point sample
        self.farthest_sampler = FarthestSampler(dim=3)

        # store the calibration matrix for each sequence
        self.calib_helper = KittiCalibHelper(root)
        # print(self.calib_helper.calib_matrix_dict)

        # list of (pc_path, img_path, seq, i, img_key)
        self.dataset = make_kitti_dataset(root, mode, opt)

    def augment_pc(self, pc_np, intensity_np, sn_np):
        """

        :param pc_np: 3xN, np.ndarray
        :param intensity_np: 3xN, np.ndarray
        :return:
        """
        # add Gaussian noise
        pc_np = augmentation.jitter_point_cloud(pc_np, sigma=0.01, clip=0.05)
        sn_np = augmentation.jitter_point_cloud(sn_np, sigma=0.01, clip=0.05)
        return pc_np, intensity_np, sn_np

    def augment_img(self, img_np):
        """

        :param img: HxWx3, np.ndarray
        :return:
        """
        # color perturbation
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter.get_params(brightness, contrast, saturation, hue)
        img_color_aug_np = np.array(color_aug(Image.fromarray(img_np)))

        return img_color_aug_np

    def generate_random_transform(self,
                                  P_tx_amplitude, P_ty_amplitude, P_tz_amplitude,
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

    def downsample_np(self, pc_np, intensity_np, sn_np):
        if pc_np.shape[1] >= self.opt.input_pt_num:
            choice_idx = np.random.choice(pc_np.shape[1], self.opt.input_pt_num, replace=False)
        else:
            fix_idx = np.asarray(range(pc_np.shape[1]))
            while pc_np.shape[1] + fix_idx.shape[0] < self.opt.input_pt_num:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
            random_idx = np.random.choice(pc_np.shape[1], self.opt.input_pt_num - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        pc_np = pc_np[:, choice_idx]
        intensity_np = intensity_np[:, choice_idx]
        sn_np = sn_np[:, choice_idx]

        return pc_np, intensity_np, sn_np

    def get_sequence_j(self, seq_sample_num, seq_i, seq_pose_folder,
                       delta_ij_max, translation_max):
        # get the max and min of possible j
        seq_j_min = max(seq_i - delta_ij_max, 0)
        seq_j_max = min(seq_i + delta_ij_max, seq_sample_num - 1)

        # pose of i
        Pi = np.load(os.path.join(seq_pose_folder, '%06d.npz' % seq_i))['pose'].astype(np.float32)  # 4x4


        while True:
            seq_j = random.randint(seq_j_min, seq_j_max)
            # get the pose, if the pose is too large, ignore and re-sample
            Pj = np.load(os.path.join(seq_pose_folder, '%06d.npz' % seq_j))['pose'].astype(np.float32)  # 4x4
            Pji = np.dot(np.linalg.inv(Pj), Pi)  # 4x4
            t_ji = Pji[0:3, 3]  # 3
            t_ji_norm = np.linalg.norm(t_ji)  # scalar

            if t_ji_norm < translation_max:
                break
            else:
                continue

        return seq_j, Pji, t_ji


    def search_for_accumulation(self, pc_folder, seq_pose_folder,
                                seq_i, seq_sample_num, Pc, P_oi,
                                stride):
        Pc_inv = np.linalg.inv(Pc)
        P_io = np.linalg.inv(P_oi)

        pc_np_list, intensity_np_list, sn_np_list = [], [], []

        counter = 0
        while len(pc_np_list) < self.opt.accumulation_frame_num:
            counter += 1
            seq_j = seq_i + stride * counter
            if seq_j < 0 or seq_j >= seq_sample_num:
                break

            npy_data = np.load(os.path.join(pc_folder, '%06d.npy' % seq_j)).astype(np.float32)
            pc_np = npy_data[0:3, :]  # 3xN
            intensity_np = npy_data[3:4, :]  # 1xN
            sn_np = npy_data[4:7, :]  # 3xN

            P_oj = np.load(os.path.join(seq_pose_folder, '%06d.npz' % seq_j))['pose'].astype(np.float32)  # 4x4
            P_ij = np.dot(P_io, P_oj)

            P_transform = np.dot(Pc_inv, np.dot(P_ij, Pc))
            pc_np = transform_pc_np(P_transform, pc_np)
            P_transform_rot = np.copy(P_transform)
            P_transform_rot[0:3, 3] = 0
            sn_np = transform_pc_np(P_transform_rot, sn_np)

            pc_np_list.append(pc_np)
            intensity_np_list.append(intensity_np)
            sn_np_list.append(sn_np)

        return pc_np_list, intensity_np_list, sn_np_list

    def get_accumulated_pc(self, pc_folder, seq_pose_folder, seq_i, seq_sample_num, Pc):
        pc_path = os.path.join(pc_folder, '%06d.npy' % seq_i)
        npy_data = np.load(pc_path).astype(np.float32)
        # shuffle the point cloud data, this is necessary!
        npy_data = npy_data[:, np.random.permutation(npy_data.shape[1])]
        pc_np = npy_data[0:3, :]  # 3xN
        intensity_np = npy_data[3:4, :]  # 1xN
        sn_np = npy_data[4:7, :]  # 3xN

        if self.opt.accumulation_frame_num <= 0.5:
            return pc_np, intensity_np, sn_np

        pc_np_list = [pc_np]
        intensity_np_list = [intensity_np]
        sn_np_list = [sn_np]

        # pose of i
        P_oi = np.load(os.path.join(seq_pose_folder, '%06d.npz' % seq_i))['pose'].astype(np.float32)  # 4x4

        # search for previous
        prev_pc_np_list, \
        prev_intensity_np_list, \
        prev_sn_np_list = self.search_for_accumulation(pc_folder,
                                                       seq_pose_folder,
                                                       seq_i,
                                                       seq_sample_num,
                                                       Pc,
                                                       P_oi,
                                                       -self.opt.accumulation_frame_skip)
        # search for next
        next_pc_np_list, \
        next_intensity_np_list, \
        next_sn_np_list = self.search_for_accumulation(pc_folder,
                                                       seq_pose_folder,
                                                       seq_i,
                                                       seq_sample_num,
                                                       Pc,
                                                       P_oi,
                                                       self.opt.accumulation_frame_skip)

        pc_np_list = pc_np_list + prev_pc_np_list + next_pc_np_list
        intensity_np_list = intensity_np_list + prev_intensity_np_list + next_intensity_np_list
        sn_np_list = sn_np_list + prev_sn_np_list + next_sn_np_list

        pc_np = np.concatenate(pc_np_list, axis=1)
        intensity_np = np.concatenate(intensity_np_list, axis=1)
        sn_np = np.concatenate(sn_np_list, axis=1)

        return pc_np, intensity_np, sn_np



    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pc_folder, img_folder, seq, seq_i, seq_sample_num, img_key = self.dataset[index]
        seq_pose_folder = os.path.join(self.root, 'poses', '%02d' % seq)

        # load point cloud of seq_i
        Pc = np.dot(self.calib_helper.get_matrix(seq, img_key),
                    self.calib_helper.get_matrix(seq, 'Tr'))
        pc_np, intensity_np, sn_np = self.get_accumulated_pc(pc_folder, seq_pose_folder, seq_i, seq_sample_num, Pc)

        if pc_np.shape[1] > 2 * self.opt.input_pt_num:
            # point cloud too huge, voxel grid downsample first
            pc_np, intensity_np, sn_np = downsample_with_intensity_sn(pc_np, intensity_np, sn_np,
                                                            voxel_grid_downsample_size=0.3)
            pc_np = pc_np.astype(np.float32)
            intensity_np = intensity_np.astype(np.float32)
            sn_np = sn_np.astype(np.float32)
            # random downsample to a specific shape, pc is still in NWU coordinate
        pc_np, intensity_np, sn_np = self.downsample_np(pc_np, intensity_np, sn_np)

        # limit max_z, the pc is in NWU coordinate
        # pc_np_x_square = np.square(pc_np[0, :])
        # pc_np_y_square = np.square(pc_np[1, :])
        # pc_np_range_square = pc_np_x_square + pc_np_y_square
        # pc_mask_range = pc_np_range_square < self.opt.pc_max_range * self.opt.pc_max_range
        # pc_np = pc_np[:, pc_mask_range]
        # intensity_np = intensity_np[:, pc_mask_range]


        # load image of seq_j
        if self.opt.translation_max < 0:
            seq_j = seq_i
            Pji = np.identity(4)
            t_ji = Pji[0:3, 3]
        else:
            seq_j, Pji, t_ji = self.get_sequence_j(seq_sample_num, seq_i, seq_pose_folder,
                                                   self.opt.delta_ij_max, self.opt.translation_max)

        img_path = os.path.join(img_folder, '%06d.npy' % seq_j)
        img = np.load(img_path)
        K = self.calib_helper.get_matrix(seq, img_key + '_K')
        # crop the first few rows, original is 370x1226 now
        img = img[self.opt.crop_original_top_rows:, :, :]
        K = camera_matrix_cropping(K, dx=0, dy=self.opt.crop_original_top_rows)
        # scale
        img = cv2.resize(img,
                         (int(round(img.shape[1] * self.opt.img_scale)),
                          int(round((img.shape[0] * self.opt.img_scale)))),
                         interpolation=cv2.INTER_LINEAR)
        K = camera_matrix_scaling(K, self.opt.img_scale)

        # random crop into input size
        if 'train' == self.mode:
            img_crop_dx = random.randint(0, img.shape[1] - self.opt.img_W)
            img_crop_dy = random.randint(0, img.shape[0] - self.opt.img_H)
        else:
            img_crop_dx = int((img.shape[1] - self.opt.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.opt.img_H) / 2)
        # crop image
        img = img[img_crop_dy:img_crop_dy + self.opt.img_H,
              img_crop_dx:img_crop_dx + self.opt.img_W, :]
        K = camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)


        #  ------------- apply random transform on points under the NWU coordinate ------------
        if 'train' == self.mode:
            Pr = self.generate_random_transform(self.opt.P_tx_amplitude, self.opt.P_ty_amplitude, self.opt.P_tz_amplitude,
                                                self.opt.P_Rx_amplitude, self.opt.P_Ry_amplitude, self.opt.P_Rz_amplitude)
            Pr_inv = np.linalg.inv(Pr)

            # -------------- augmentation ----------------------
            pc_np, intensity_np, sn_np = self.augment_pc(pc_np, intensity_np, sn_np)
            img = self.augment_img(img)
            if random.random() > 0.5:
                img = np.flip(img, 1)
                P_flip = np.asarray([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=pc_np.dtype)
                Pr = np.dot(Pr, P_flip)
            Pr_inv = np.linalg.inv(Pr)
        elif 'val_random_Ry' == self.mode:
            Pr = self.generate_random_transform(0, 0, 0,
                                                0, math.pi*2, 0)
            Pr_inv = np.linalg.inv(Pr)
        else:
            Pr = np.identity(4, dtype=np.float)
            Pr_inv = np.identity(4, dtype=np.float)

        P_cam_nwu = np.asarray([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=pc_np.dtype)
        P_nwu_cam = np.linalg.inv(P_cam_nwu)

        # now the point cloud is in CAMERA coordinate
        Pr_Pcamnwu = np.dot(Pr, P_cam_nwu)
        pc_np = transform_pc_np(Pr_Pcamnwu, pc_np)
        sn_np = transform_pc_np(Pr_Pcamnwu, sn_np)

        # assemble P. P * pc will get the point cloud in the camera image coordinate
        PcPnwucamPrinv = np.dot(Pc, np.dot(P_nwu_cam, Pr_inv))
        P = np.dot(Pji, PcPnwucamPrinv)  # 4x4


        # debug for kitti odometry correctness
        # print(Pij)
        #
        # data_i = np.load(os.path.join(pc_folder, '%06d.npy' % seq_i)).astype(np.float32)
        # pc_np_i = data_i[0:3, :]
        # intensity_np_i = data_i[3:4, :]
        # pc_np_i, intensity_np_i = self.downsample_np(pc_np_i, intensity_np_i)
        #
        # pc_homo_np = np.concatenate((pc_np, np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)), axis=0)  # 4xN
        # PijPc = np.dot(Pij, Pc)
        # Pc_pc_np = np.dot(PijPc, pc_homo_np)[0:3, :]
        #
        # pc_homo_np_i = np.concatenate((pc_np_i, np.ones((1, pc_np_i.shape[1]), dtype=pc_np_i.dtype)), axis=0)  # 4xN
        # Pc_pc_np_i = np.dot(Pc, pc_homo_np_i)[0:3, :]
        #
        # ax = vis_tools.plot_pc(Pc_pc_np, color=(1, 0, 0))
        # ax = vis_tools.plot_pc(Pc_pc_np_i, color=(0, 0, 1), ax=ax)
        # plt.show()

        # visualization of random transformation & augmentation
        # pc_np_homo = np.concatenate((pc_np, np.ones((1, pc_np.shape[1]))), axis=0)  # 4xN
        # pc_np_recovered_homo = np.dot(P, pc_np_homo)
        # pc_np_recovered_vis = projection_pc_img(pc_np_recovered_homo[0:3, :], img, K, size=1)
        # plt.figure()
        # plt.imshow(pc_np_recovered_vis)
        # plt.show()

        # ------------ Farthest Point Sampling ------------------
        # node_a_np = fps_approximate(pc_np, voxel_size=4.0, node_num=self.opt.node_a_num)
        node_a_np, _ = self.farthest_sampler.sample(pc_np[:, np.random.choice(pc_np.shape[1],
                                                                              self.opt.node_a_num * 8,
                                                                              replace=False)],
                                                    k=self.opt.node_a_num)
        node_b_np, _ = self.farthest_sampler.sample(pc_np[:, np.random.choice(pc_np.shape[1],
                                                                            self.opt.node_b_num * 8,
                                                                            replace=False)],
                                                  k=self.opt.node_b_num)

        # visualize nodes
        # ax = vis_tools.plot_pc(pc_np, size=1)
        # ax = vis_tools.plot_pc(node_a_np, size=10, ax=ax)
        # plt.show()

        # -------------- convert to torch tensor ---------------------
        pc = torch.from_numpy(pc_np)  # 3xN
        intensity = torch.from_numpy(intensity_np)  # 1xN
        sn = torch.from_numpy(sn_np)  # 3xN
        node_a = torch.from_numpy(node_a_np)  # 3xMa
        node_b = torch.from_numpy(node_b_np)  # 3xMb

        P = torch.from_numpy(P[0:3, :].astype(np.float32))  # 3x4

        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).contiguous()  # 3xHxW
        K = torch.from_numpy(K.astype(np.float32))  # 3x3

        t_ji = torch.from_numpy(t_ji.astype(np.float32))  # 3

        return pc, intensity, sn, node_a, node_b, \
               P, img, K, \
               t_ji


if __name__ == '__main__':
    root_path = '/ssd/jiaxin/datasets/kitti'
    opt = options.Options()
    kittiloader = KittiLoader(root_path, 'train', opt)

    for i in range(0, len(kittiloader), 1000):
        print('--- %d ---' % i)
        data = kittiloader[i]
        for item in data:
            print(item.size())
