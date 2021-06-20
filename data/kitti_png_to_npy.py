import os
import os.path
import numpy as np
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from data import kitti_helper
from data import augmentation


def make_kitti_dataset(root_path, mode, opt):
    dataset = []

    if mode == 'train':
        seq_list = list(range(9))
    elif mode == 'val':
        seq_list = [9, 10]
    else:
        raise Exception('Invalid mode.')

    np_folder = 'voxel0.1-SNr0.6'
    for seq in seq_list:
        pc_folder = os.path.join(root_path, 'data_odometry_velodyne', 'sequences', '%02d' % seq, np_folder)
        img2_folder = os.path.join(root_path, 'data_odometry_color', 'sequences', '%02d' % seq, 'image_2')
        img3_folder = os.path.join(root_path, 'data_odometry_color', 'sequences', '%02d' % seq, 'image_3')
        sample_num = round(len(os.listdir(pc_folder)))

        for i in range(sample_num):
            pc_path = os.path.join(pc_folder, '%06d.npy' % i)
            img2_path = os.path.join(img2_folder, '%06d.png' % i)
            img3_path = os.path.join(img3_folder, '%06d.png' % i)
            dataset.append((pc_path, img2_path, seq, i, 'P2'))
            dataset.append((pc_path, img3_path, seq, i, 'P3'))

    return dataset


if __name__ == '__main__':
    root_path = '/ssd/jiaxin/datasets/kitti'
    seq_list = range(0, 22)
    np_folder = 'voxel0.1-SNr0.6'

    is_save_img = False
    is_save_pc = True

    calib_helper = kitti_helper.KittiCalibHelper(root_path)

    for seq in seq_list:
        pc_folder = os.path.join(root_path, 'data_odometry_velodyne', 'sequences', '%02d' % seq, np_folder)
        img2_folder = os.path.join(root_path, 'data_odometry_color', 'sequences', '%02d' % seq, 'image_2')
        img3_folder = os.path.join(root_path, 'data_odometry_color', 'sequences', '%02d' % seq, 'image_3')
        sample_num = round(len(os.listdir(img2_folder)))

        img2_folder_npy = os.path.join(root_path, 'data_odometry_color_npy', 'sequences', '%02d' % seq, 'image_2')
        img3_folder_npy = os.path.join(root_path, 'data_odometry_color_npy', 'sequences', '%02d' % seq, 'image_3')
        if os.path.isdir(img2_folder_npy) == False:
            os.makedirs(img2_folder_npy)
        if os.path.isdir(img3_folder_npy) == False:
            os.makedirs(img3_folder_npy)

        pc_img2_folder_npy = os.path.join(root_path, 'data_odometry_velodyne_P2_npy', 'sequences', '%02d' % seq, np_folder)
        pc_img3_folder_npy = os.path.join(root_path, 'data_odometry_velodyne_P3_npy', 'sequences', '%02d' % seq, np_folder)
        if os.path.isdir(pc_img2_folder_npy) == False:
            os.makedirs(pc_img2_folder_npy)
        if os.path.isdir(pc_img3_folder_npy) == False:
            os.makedirs(pc_img3_folder_npy)

        for i in range(sample_num):
            print('working on seq %d - image %d' % (seq, i))

            # ----------- png -------------
            img2_path = os.path.join(img2_folder, '%06d.png' % i)
            img3_path = os.path.join(img3_folder, '%06d.png' % i)

            img2 = cv2.imread(img2_path)
            img2 = img2[:, :, ::-1]  # HxWx3

            img3 = cv2.imread(img3_path)
            img3 = img3[:, :, ::-1]  # HxWx3

            if is_save_img:
                np.save(os.path.join(img2_folder_npy, '%06d.npy' % i), img2)
                np.save(os.path.join(img3_folder_npy, '%06d.npy' % i), img3)

            if not is_save_pc:
                continue

            # ------------- point cloud ----------------
            pc_path = os.path.join(pc_folder, '%06d.npy' % i)
            npy_data = np.load(pc_path)
            pc_np = npy_data[0:3, :]  # 3xN
            intensity_np = npy_data[3:4, :]  # 1xN
            sn_np = npy_data[4:7, :]  # 3xN

            # crop the pc, remove x<1.0 in NWU coordinate
            pc_mask_z = pc_np[0, :] > 1.0
            pc_np = pc_np[:, pc_mask_z]
            intensity_np = intensity_np[:, pc_mask_z]
            sn_np = sn_np[:, pc_mask_z]

            # ------------- transform into CAM coordinate -------------------
            for img_key in ['P2', 'P3']:
                K = calib_helper.get_matrix(seq, img_key + '_K')
                Pi = calib_helper.get_matrix(seq, img_key)
                Tr = calib_helper.get_matrix(seq, 'Tr')
                pc_img_np = calib_helper.transform_pc_vel_to_img(pc_np,
                                                                  Pi=Pi,
                                                                  Tr=Tr)
                intensity_img_np = intensity_np
                sn_img_np = augmentation.coordinate_NWU_to_cam(sn_np)


                if 'P2' == img_key:
                    img = img2
                    pc_img_save_path = os.path.join(pc_img2_folder_npy, '%06d.npy' % i)
                elif 'P3' == img_key:
                    img = img3
                    pc_img_save_path = os.path.join(pc_img3_folder_npy, '%06d.npy' % i)
                else:
                    assert False

                # project the points onto image as visualization
                # pc_np_vis = kitti_helper.projection_pc_img(pc_img_np, img, K)
                # plt.figure()
                # plt.imshow(pc_np_vis)

                # crop point cloud, get those overlapped with image only
                pc_img_np, intensity_img_np, sn_img_np = kitti_helper.crop_pc_with_img(pc_img_np,
                                                                                       intensity_img_np,
                                                                                       sn_img_np,
                                                                                       img,
                                                                                       K)
                npy_out = np.asarray(np.concatenate((pc_img_np,
                                                     intensity_img_np,
                                                     sn_img_np), axis=0),
                                     dtype=np.float32)
                np.save(pc_img_save_path, npy_out)

                # plt.show()

