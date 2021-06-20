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
import pickle
from pyquaternion import Quaternion

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from data import augmentation
from util import vis_tools
from nuscenes_t import options
from data.kitti_helper import FarthestSampler, camera_matrix_cropping, camera_matrix_scaling, projection_pc_img

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.nuscenes import NuScenes


def downsample_with_reflectance(pointcloud, reflectance, voxel_grid_downsample_size):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.transpose(pointcloud[0:3, :]))
    reflectance_max = np.max(reflectance)

    fake_colors = np.zeros((pointcloud.shape[1], 3))
    fake_colors[:, 0] = reflectance / reflectance_max
    pcd.colors = open3d.utility.Vector3dVector(fake_colors)
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_grid_downsample_size)
    down_pcd_points = np.transpose(np.asarray(down_pcd.points))  # 3xN
    pointcloud = down_pcd_points
    reflectance = np.asarray(down_pcd.colors)[:, 0] * reflectance_max

    return pointcloud, reflectance



def load_dataset_info(filepath):
    with open(filepath, 'rb') as f:
        dataset_read = pickle.load(f)
    return dataset_read

def make_nuscenes_dataset(root_path):
    dataset = load_dataset_info(os.path.join(root_path, 'dataset_info.list'))
    return dataset


def get_sample_data_ego_pose_P(nusc, sample_data):
    sample_data_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    sample_data_pose_R = np.asarray(Quaternion(sample_data_pose['rotation']).rotation_matrix).astype(np.float32)
    sample_data_pose_t = np.asarray(sample_data_pose['translation']).astype(np.float32)
    sample_data_pose_P = get_P_from_Rt(sample_data_pose_R, sample_data_pose_t)
    return sample_data_pose_P


def get_calibration_P(nusc, sample_data):
    calib = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    R = np.asarray(Quaternion(calib['rotation']).rotation_matrix).astype(np.float32)
    t = np.asarray(calib['translation']).astype(np.float32)
    P = get_P_from_Rt(R, t)
    return P


def get_P_from_Rt(R, t):
    P = np.identity(4)
    P[0:3, 0:3] = R
    P[0:3, 3] = t
    return P


def get_camera_K(nusc, camera):
    calib = nusc.get('calibrated_sensor', camera['calibrated_sensor_token'])
    return np.asarray(calib['camera_intrinsic']).astype(np.float32)


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


class nuScenesLoader(data.Dataset):
    def __init__(self, root, mode, opt: options.Options):
        super(nuScenesLoader, self).__init__()
        self.root = root
        self.opt = opt
        self.mode = mode

        # farthest point sample
        self.farthest_sampler = FarthestSampler(dim=3)

        # list of (traversal, pc_timestamp, pc_timestamp_idx, traversal_pc_num)
        if mode == 'train':
            self.nuscenes_path = os.path.join(root, 'trainval')
            version = 'v1.0-trainval'
        else:
            self.nuscenes_path = os.path.join(root, 'test')
            version = 'v1.0-test'

        self.dataset = make_nuscenes_dataset(self.nuscenes_path)
        self.nusc = NuScenes(version=version, dataroot=self.nuscenes_path, verbose=True)

        self.camera_name_list = ['CAM_FRONT',
                'CAM_FRONT_LEFT',
                'CAM_FRONT_RIGHT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT']

    def augment_pc(self, pc_np, intensity_np):
        """

        :param pc_np: 3xN, np.ndarray
        :param intensity_np: 3xN, np.ndarray
        :param sn_np: 1xN, np.ndarray
        :return:
        """
        # add Gaussian noise
        pc_np = augmentation.jitter_point_cloud(pc_np, sigma=0.01, clip=0.05)
        intensity_np = augmentation.jitter_point_cloud(intensity_np, sigma=0.01, clip=0.05)
        return pc_np, intensity_np

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
        P_random = np.identity(4, dtype=np.float32)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t

        return P_random.astype(np.float32)

    def downsample_np(self, pc_np, intensity_np, k):
        if pc_np.shape[1] >= k:
            choice_idx = np.random.choice(pc_np.shape[1], k, replace=False)
        else:
            fix_idx = np.asarray(range(pc_np.shape[1]))
            while pc_np.shape[1] + fix_idx.shape[0] < k:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
            random_idx = np.random.choice(pc_np.shape[1], k - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        pc_np = pc_np[:, choice_idx]
        intensity_np = intensity_np[:, choice_idx]

        return pc_np, intensity_np


    def get_lidar_pc_intensity_by_token(self, lidar_token):
        lidar = self.nusc.get('sample_data', lidar_token)
        pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, lidar['filename']))
        pc_np = pc.points[0:3, :]
        intensity_np = pc.points[3:4, :]

        # remove point falls on egocar
        x_inside = np.logical_and(pc_np[0, :] < 0.8, pc_np[0, :] > -0.8)
        y_inside = np.logical_and(pc_np[1, :] < 2.7, pc_np[1, :] > -2.7)
        inside_mask = np.logical_and(x_inside, y_inside)
        outside_mask = np.logical_not(inside_mask)
        pc_np = pc_np[:, outside_mask]
        intensity_np = intensity_np[:, outside_mask]

        P_oi = get_sample_data_ego_pose_P(self.nusc, lidar)

        return pc_np, intensity_np, P_oi


    def lidar_frame_accumulation(self, lidar, P_io, P_lidar_vehicle, P_vehicle_lidar,
                                 direction,
                                 pc_np_list, intensity_np_list):
        counter = 1
        accumulated_counter = 0
        while accumulated_counter < self.opt.accumulation_frame_num:
            if lidar[direction] == '':
                break

            if counter % self.opt.accumulation_frame_skip != 0:
                counter += 1
                lidar = self.nusc.get('sample_data', lidar[direction])
                continue

            pc_np_j, intensity_np_j, P_oj = self.get_lidar_pc_intensity_by_token(lidar[direction])
            P_ij = np.dot(P_io, P_oj)
            P_ij_trans = np.dot(np.dot(P_lidar_vehicle, P_ij), P_vehicle_lidar)
            pc_np_j_transformed = transform_pc_np(P_ij_trans, pc_np_j)
            pc_np_list.append(pc_np_j_transformed)
            intensity_np_list.append(intensity_np_j)

            counter += 1
            lidar = self.nusc.get('sample_data', lidar[direction])
            accumulated_counter += 1

        # print('accumulation %s %d' % (direction, counter))
        return pc_np_list, intensity_np_list


    def accumulate_lidar_points(self, lidar):
        pc_np_list = []
        intensity_np_list = []
        # load itself
        pc_np_i, intensity_np_i, P_oi = self.get_lidar_pc_intensity_by_token(lidar['token'])
        pc_np_list.append(pc_np_i)
        intensity_np_list.append(intensity_np_i)
        P_io = np.linalg.inv(P_oi)

        P_vehicle_lidar = get_calibration_P(self.nusc, lidar)
        P_lidar_vehicle = np.linalg.inv(P_vehicle_lidar)

        # load next
        pc_np_list, intensity_np_list = self.lidar_frame_accumulation(lidar, P_io, P_lidar_vehicle, P_vehicle_lidar,
                                                                      'next',
                                                                      pc_np_list, intensity_np_list)

        # load prev
        pc_np_list, intensity_np_list = self.lidar_frame_accumulation(lidar, P_io, P_lidar_vehicle, P_vehicle_lidar,
                                                                      'prev',
                                                                      pc_np_list, intensity_np_list)

        pc_np = np.concatenate(pc_np_list, axis=1)
        intensity_np = np.concatenate(intensity_np_list, axis=1)

        return pc_np, intensity_np


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        lidar_token = item[0]
        nearby_cam_token_dict = item[1]

        # load point cloud
        lidar = self.nusc.get('sample_data', lidar_token)
        pc_np, intensity_np = self.accumulate_lidar_points(lidar)

        # voxel downsample
        if pc_np.shape[1] > 2 * self.opt.input_pt_num:
            # point cloud too huge, voxel grid downsample first
            pc_np, intensity_np = downsample_with_reflectance(pc_np, intensity_np[0], voxel_grid_downsample_size=0.2)
            intensity_np = np.expand_dims(intensity_np, axis=0)
            pc_np = pc_np.astype(np.float32)
            intensity_np = intensity_np.astype(np.float32)
        # random sampling
        pc_np, intensity_np = self.downsample_np(pc_np, intensity_np, self.opt.input_pt_num)

        lidar_calib_P = get_calibration_P(self.nusc, lidar)
        lidar_pose_P = get_sample_data_ego_pose_P(self.nusc, lidar)

        # load image
        # randomly select camera and nearby_idx
        camera_name = random.choice(self.camera_name_list)
        nearby_camera_token = random.choice(nearby_cam_token_dict[camera_name])
        camera = self.nusc.get('sample_data', nearby_camera_token)
        img = np.array(Image.open(os.path.join(self.nusc.dataroot, camera['filename'])))
        K = get_camera_K(self.nusc, camera)

        # crop top 100 rows
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

        camera_calib_P = get_calibration_P(self.nusc, camera)
        camera_pose_P = get_sample_data_ego_pose_P(self.nusc, camera)

        #  ------------- apply random transform on points under the NWU coordinate ------------
        if 'train' == self.mode:
            Pr = self.generate_random_transform(self.opt.P_tx_amplitude, self.opt.P_ty_amplitude,
                                                self.opt.P_tz_amplitude,
                                                self.opt.P_Rx_amplitude, self.opt.P_Ry_amplitude,
                                                self.opt.P_Rz_amplitude)
            Pr_inv = np.linalg.inv(Pr)

            # -------------- augmentation ----------------------
            pc_np, intensity_np = self.augment_pc(pc_np, intensity_np)
            if random.random() > 0.5:
                img = self.augment_img(img)
        elif 'val_random_Ry' == self.mode:
            Pr = self.generate_random_transform(0, 0, 0,
                                                0, 0, math.pi*2)
            Pr_inv = np.linalg.inv(Pr)
        else:
            Pr = np.identity(4, dtype=np.float32)
            Pr_inv = np.identity(4, dtype=np.float32)

        # random rotate pc_np
        pc_np = transform_pc_np(Pr, pc_np)


        camera_pose_P_inv = np.linalg.inv(camera_pose_P)
        camera_calib_P_inv = np.linalg.inv(camera_calib_P)
        P_cam_pc = np.dot(camera_calib_P_inv, np.dot(camera_pose_P_inv,
                                                        np.dot(lidar_pose_P, lidar_calib_P)))
        P = np.dot(P_cam_pc, Pr_inv)
        t_ij = P_cam_pc[0:3, 3]

        # debug, visualization
        # new_pc_homo_np = np.concatenate((pc_np,
        #                              np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)),
        #                             axis=0)
        # pc_np_cam = np.dot(P, new_pc_homo_np)[0:3, :]
        #
        # img_vis = projection_pc_img(pc_np_cam, img,
        #                             K,
        #                             size=2)
        # plt.figure()
        # plt.imshow(img_vis)
        # plt.show()

        # ------------ Farthest Point Sampling ------------------
        # node_a_np = fps_approximate(pc_np, voxel_size=4.0, node_num=self.opt.node_a_num)
        node_a_np, _ = self.farthest_sampler.sample(pc_np[:, np.random.choice(pc_np.shape[1],
                                                                              int(self.opt.node_a_num * 8),
                                                                              replace=False)],
                                                    k=self.opt.node_a_num)
        node_b_np, _ = self.farthest_sampler.sample(pc_np[:, np.random.choice(pc_np.shape[1],
                                                                              int(self.opt.node_b_num * 8),
                                                                              replace=False)],
                                                    k=self.opt.node_b_num)

        # visualize nodes
        # ax = vis_tools.plot_pc(pc_np, size=1)
        # ax = vis_tools.plot_pc(node_a_np, size=10, ax=ax)
        # plt.show()

        # -------------- convert to torch tensor ---------------------
        pc = torch.from_numpy(pc_np)  # 3xN
        intensity = torch.from_numpy(intensity_np)  # 1xN
        sn = torch.zeros(pc.size(), dtype=pc.dtype, device=pc.device)
        node_a = torch.from_numpy(node_a_np)  # 3xMa
        node_b = torch.from_numpy(node_b_np)  # 3xMb

        P = torch.from_numpy(P[0:3, :].astype(np.float32))  # 3x4

        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).contiguous()  # 3xHxW
        K = torch.from_numpy(K.astype(np.float32))  # 3x3

        t_ij = torch.from_numpy(t_ij.astype(np.float32))  # 3

        # print(P)
        # print(t_ij)
        # print(pc)
        # print(intensity)

        return pc, intensity, sn, node_a, node_b, \
               P, img, K, \
               t_ij


if __name__ == '__main__':
    root_path = '/extssd/jiaxin/nuscenes'
    opt = options.Options()
    nuscenesloader = nuScenesLoader(root_path, 'val', opt)

    for i in range(0, len(nuscenesloader), 1000):
        print('--- %d ---' % i)
        data = nuscenesloader[i]
        for item in data:
            print(item.size())
