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
import bisect

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from data import augmentation
from util import vis_tools
from oxford import options
from data.kitti_helper import FarthestSampler, camera_matrix_cropping, camera_matrix_scaling, projection_pc_img


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


def read_train_val_split(txt_path):
    with open(txt_path) as f:
        sets = [x.rstrip() for x in f.readlines()]
    traversal_list = list(sets)
    return traversal_list


def clamp(n, smallest, largest): 
    return max(smallest, min(n, largest))


def make_oxford_dataset(root_path, mode, opt):
    dataset = []
    pc_timestamps_list_dict = {}
    pc_poses_np_dict = {}
    camera_timestamps_list_dict = {}
    camera_poses_np_dict = {}


    if mode == 'train':
        seq_list = read_train_val_split(os.path.join(root_path, 'train.txt'))
    elif 'val' in mode:
        seq_list = read_train_val_split(os.path.join(root_path, 'val.txt'))
    else:
        raise Exception('Invalid mode.')

    for traversal in seq_list:
        P_convert = np.asarray([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        P_convert_inv = np.linalg.inv(P_convert)

        pc_timestamps_np = np.load(os.path.join(root_path, traversal, 'pc_timestamps.npy'))
        pc_timestamps_list_dict[traversal] = pc_timestamps_np.tolist()
        pc_poses_np = np.load(os.path.join(root_path, traversal, 'pc_poses.npy')).astype(np.float32)
        # convert it to camera coordinate
        for b in range(pc_poses_np.shape[0]):
            pc_poses_np[b] = np.dot(P_convert, np.dot(pc_poses_np[b], P_convert_inv))

        pc_poses_np_dict[traversal] = pc_poses_np

        img_timestamps_np = np.load(os.path.join(root_path, traversal, 'camera_timestamps.npy'))
        camera_timestamps_list_dict[traversal] = img_timestamps_np.tolist()
        img_poses_np = np.load(os.path.join(root_path, traversal, 'camera_poses.npy')).astype(np.float32)
        # convert it to camera coordinate
        for b in range(img_poses_np.shape[0]):
            img_poses_np[b] = np.dot(P_convert, np.dot(img_poses_np[b], P_convert_inv))

        camera_poses_np_dict[traversal] = img_poses_np

        for i in range(pc_timestamps_np.shape[0]):
            pc_timestamp = pc_timestamps_np[i]
            dataset.append((traversal, pc_timestamp, i, pc_timestamps_np.shape[0]))

    return dataset, \
           pc_timestamps_list_dict, pc_poses_np_dict, \
           camera_timestamps_list_dict, camera_poses_np_dict


class OxfordLoader(data.Dataset):
    def __init__(self, root, mode, opt: options.Options):
        super(OxfordLoader, self).__init__()
        self.root = root
        self.opt = opt
        self.mode = mode

        # farthest point sample
        self.farthest_sampler = FarthestSampler(dim=3)

        # list of (traversal, pc_timestamp, pc_timestamp_idx, traversal_pc_num)
        self.dataset, \
        self.pc_timestamps_list_dict, self.pc_poses_np_dict, \
        self.camera_timestamps_list_dict, self.camera_poses_np_dict = make_oxford_dataset(root, mode, opt)


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

    def get_camera_timestamp(self,
                             pc_timestamp_idx,
                             traversal_pc_num,
                             pc_timestamps_list,
                             pc_poses_np,
                             camera_timestamps_list,
                             camera_poses_np):
        if self.mode == 'train':
            translation_max = self.opt.translation_max
        else:
            translation_max = self.opt.test_translation_max
        # pc is built every opt.pc_build_interval (2m),
        # so search for the previous/nex pc_timestamp that > max_translation
        index_interval = math.ceil(translation_max / self.opt.pc_build_interval)

        previous_pc_t_idx = max(0, pc_timestamp_idx - index_interval)
        previous_pc_t = pc_timestamps_list[previous_pc_t_idx]
        next_pc_t_idx = min(traversal_pc_num-1, pc_timestamp_idx + index_interval)
        next_pc_t = pc_timestamps_list[next_pc_t_idx]

        previous_cam_t_idx = bisect.bisect_left(camera_timestamps_list, previous_pc_t)
        next_cam_t_idx = bisect.bisect_left(camera_timestamps_list, next_pc_t)

        P_o_pc = pc_poses_np[pc_timestamp_idx]
        while True:
            cam_t_idx = random.randint(previous_cam_t_idx, next_cam_t_idx)
            P_o_cam = camera_poses_np[cam_t_idx]
            P_cam_pc = np.dot(np.linalg.inv(P_o_cam), P_o_pc)
            t_norm = np.linalg.norm(P_cam_pc[0:3, 3])

            if t_norm < translation_max:
                break

        return cam_t_idx, P_cam_pc


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        K = np.asarray([[964.828979, 0, 643.788025], [0, 964.828979, 484.407990], [0, 0, 1]], dtype=np.float32)

        traversal, pc_timestamp, pc_timestamp_idx, traversal_pc_num = self.dataset[index]
        pc_timestamps_list = self.pc_timestamps_list_dict[traversal]
        pc_poses_np = self.pc_poses_np_dict[traversal]
        camera_timestamps_list = self.camera_timestamps_list_dict[traversal]
        camera_poses_np = self.camera_poses_np_dict[traversal]

        camera_timestamp_idx, P_cam_pc = self.get_camera_timestamp(pc_timestamp_idx,
                                                                   traversal_pc_num,
                                                                   pc_timestamps_list,
                                                                   pc_poses_np,
                                                                   camera_timestamps_list,
                                                                   camera_poses_np)

        camera_folder = os.path.join(self.root, traversal, 'stereo', 'centre')
        camera_timestamp = camera_timestamps_list[camera_timestamp_idx]
        img = np.array(Image.open(os.path.join(camera_folder, "%d.jpg" % camera_timestamp)))

        # ------------- load image, original size is 960x1280, bottom rows are car itself -------------
        tmp_img_H = img.shape[0]
        img = img[0:(tmp_img_H-self.opt.crop_original_bottom_rows), :, :]
        # scale
        img = cv2.resize(img,
                         (int(round(img.shape[1] * self.opt.img_scale)), int(round((img.shape[0] * self.opt.img_scale)))),
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

        # ------------- load point cloud ----------------
        if self.opt.is_remove_ground:
            lidar_name = 'lms_front_foreground'
        else:
            lidar_name = 'lms_front'
        pc_path = os.path.join(self.root, traversal, lidar_name, '%d.npy' % pc_timestamp)
        npy_data = np.load(pc_path).astype(np.float32)
        # shuffle the point cloud data, this is necessary!
        npy_data = npy_data[:, np.random.permutation(npy_data.shape[1])]
        pc_np = npy_data[0:3, :]  # 3xN
        intensity_np = npy_data[3:4, :]  # 1xN

        # limit max_z, the pc is in CAMERA coordinate
        pc_np_x_square = np.square(pc_np[0, :])
        pc_np_z_square = np.square(pc_np[2, :])
        pc_np_range_square = pc_np_x_square + pc_np_z_square
        pc_mask_range = pc_np_range_square < self.opt.pc_max_range * self.opt.pc_max_range
        pc_np = pc_np[:, pc_mask_range]
        intensity_np = intensity_np[:, pc_mask_range]

        # remove the ground points!

        if pc_np.shape[1] > 2 * self.opt.input_pt_num:
            # point cloud too huge, voxel grid downsample first
            pc_np, intensity_np = downsample_with_reflectance(pc_np, intensity_np[0], voxel_grid_downsample_size=0.2)
            intensity_np = np.expand_dims(intensity_np, axis=0)
            pc_np = pc_np.astype(np.float32)
            intensity_np = intensity_np.astype(np.float32)
        # random sampling
        pc_np, intensity_np = self.downsample_np(pc_np, intensity_np, self.opt.input_pt_num)

        #  ------------- apply random transform on points under the NWU coordinate ------------
        if 'train' == self.mode:
            Pr = self.generate_random_transform(self.opt.P_tx_amplitude, self.opt.P_ty_amplitude, self.opt.P_tz_amplitude,
                                                self.opt.P_Rx_amplitude, self.opt.P_Ry_amplitude, self.opt.P_Rz_amplitude)
            Pr_inv = np.linalg.inv(Pr)

            # -------------- augmentation ----------------------
            pc_np, intensity_np = self.augment_pc(pc_np, intensity_np)
            if random.random() > 0.5:
                img = self.augment_img(img)
        elif 'val_random_Ry' == self.mode:
            Pr = self.generate_random_transform(0, 0, 0,
                                                0, math.pi*2, 0)
            Pr_inv = np.linalg.inv(Pr)
        else:
            Pr = np.identity(4, dtype=np.float32)
            Pr_inv = np.identity(4, dtype=np.float32)

        t_ij = P_cam_pc[0:3, 3]
        P = np.dot(P_cam_pc, Pr_inv)

        # now the point cloud is in CAMERA coordinate
        pc_homo_np = np.concatenate((pc_np, np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)), axis=0)  # 4xN
        Pr_pc_homo_np = np.dot(Pr, pc_homo_np)  # 4xN
        pc_np = Pr_pc_homo_np[0:3, :]  # 3xN

        # data_i = np.load(os.path.join(pc_folder, '%06d.npy' % seq_i)).astype(np.float32)
        # pc_np_i = data_i[0:3, :]
        # intensity_np_i = data_i[3:4, :]
        # sn_np_i = data_i[4:7, :]
        # pc_np_i, intensity_np_i, sn_np_i = self.downsample_np(pc_np_i, intensity_np_i, sn_np_i)
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
                                                                              int(self.opt.node_a_num*8),
                                                                              replace=False)],
                                                    k=self.opt.node_a_num)
        node_b_np, _ = self.farthest_sampler.sample(pc_np[:, np.random.choice(pc_np.shape[1],
                                                                            int(self.opt.node_b_num*8),
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
    root_path = '/extssd/jiaxin/oxford'
    opt = options.Options()
    oxfordloader = OxfordLoader(root_path, 'train', opt)

    for i in range(0, len(oxfordloader), 10000):
        print('--- %d ---' % i)
        data = oxfordloader[i]
        for item in data:
            print(item.size())
