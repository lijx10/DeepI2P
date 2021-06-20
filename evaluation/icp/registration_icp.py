import copy
import math
import os
import random

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from data import augmentation


def generate_uniform_random_transform(P_tx_amplitude, P_ty_amplitude, P_tz_amplitude,
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


def get_inside_img_mask(pc_np, P_np, K_np, H, W):
    # get the visualization based on P
    pc_np_homo = np.concatenate((pc_np,
                                     np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)),
                                    axis=0)
    P_points_np = np.dot(P_np, pc_np_homo)[0:3, :]
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
    return P_points_np, inside_mask[0]


def wrap_in_pi(x):
    x = math.fmod(x+math.pi, math.pi*2)
    if x<0:
        x += math.pi*2
    return x - math.pi


def get_P_diff(P_pred_np, P_gt_np):
    P_diff = np.dot(np.linalg.inv(P_pred_np), P_gt_np)
    t_diff = np.linalg.norm(P_diff[0:3, 3])

    r_diff = P_diff[0:3, 0:3]
    R_diff = Rotation.from_matrix(r_diff)
    angles_diff = np.sum(np.abs(R_diff.as_euler('xzy', degrees=True)))

    return t_diff, angles_diff


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


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])



def icp_random_init(pc_np, pc_monodepth_np, num_iterations, is_plot):
    P_tx_amplitude, P_ty_amplitude, P_tz_amplitude = 5, 0, 10
    P_Rx_amplitude, P_Ry_amplitude, P_Rz_amplitude = 0, math.pi*2, 0

    max_fitness = 0.001
    P_pred_np = np.eye(4, dtype=np.float)
    for i in range(num_iterations):
        P_init = generate_uniform_random_transform(P_tx_amplitude, P_ty_amplitude, P_tz_amplitude,
                                                   P_Rx_amplitude, P_Ry_amplitude, P_Rz_amplitude)

        P_tmp, fitness_tmp = icp_wrapper(pc_np, pc_monodepth_np, P_init, is_plot)

        # in 2D case, force to be only one unknown
        P_tmp = np.copy(P_tmp)
        P_tmp[0, 1] = 0
        P_tmp[1, 0] = 0
        P_tmp[1, 1] = 1
        P_tmp[1, 2] = 0
        P_tmp[2, 1] = 0

        if fitness_tmp > max_fitness:
            max_fitness = fitness_tmp
            P_pred_np = P_tmp

    return P_pred_np, max_fitness


def icp_wrapper(pc_np, pc_monodepth_np, P_init, is_plot):
    # run ICP tp get P_predict_np
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(np.transpose(pc_np))
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(np.transpose(pc_monodepth_np))
    threshold = 1.0
    trans_init = P_init
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    if is_plot:
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        print("GT:")
        print(P_init)
        draw_registration_result(source_pcd, target_pcd, P_init)

    return reg_p2p.transformation, reg_p2p.fitness


def main():
    root_path = '/home/tohar/repos/point-img-feature/oxford/workspace/640x384-noCrop-monodepth'

    visualization_folder = os.path.join(root_path, 'visualization')
    data_folder = os.path.join(root_path, 'data')
    monodepth_folder = os.path.join(root_path, 'monodepth')

    is_plot = False
    is_2d = True
    H = 384  # kitti=160, oxford=288/192/384, nuscenes 160
    W = 640  # kitti=512, oxford=512/320/640, nuscenes 320
    is_enu2cam = 'nuscene' in root_path

    t_diff_avg = 0
    r_diff_avg = 0
    counter = 0

    filename_list = [f[0:9] for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
    filename_list = list(set(filename_list))
    filename_list.sort()

    random_permute_idx = np.random.permutation(len(filename_list))
    filename_list = [filename_list[i] for i in random_permute_idx]

    # save to disk
    P_pred_all_np = np.zeros((len(filename_list), 4, 4))
    P_gt_all_np = np.zeros((len(filename_list), 4, 4))
    cost_all_np = np.zeros((len(filename_list)))
    for i in range(0, len(filename_list), 100):
        filename = filename_list[i]
        counter += 1

        # if filename != '000400_00':
        #     continue

        point_data_np = np.load(os.path.join(data_folder, filename + '_pc_label.npy'))
        pc_np = point_data_np[0:3, :].astype(np.float64)
        coarse_predictions_np = point_data_np[3, :].astype(np.int)
        coarse_labels_np = point_data_np[4, :].astype(np.int)
        fine_predictions_np = point_data_np[5, :].astype(np.int)
        fine_labels_np = point_data_np[6, :].astype(np.int)
        K_np = np.load(os.path.join(data_folder, filename + '_K.npy')).astype(np.float64)
        P_gt_np = np.load(os.path.join(data_folder, filename + '_P.npy')).astype(np.float64)

        pc_monodepth_np = np.load(os.path.join(monodepth_folder, filename+'_pc.npy'))
        if P_gt_np.shape[0] == 3:
            P_gt_np = np.concatenate((P_gt_np, np.identity(4)[3:4, :]), axis=0)
        if is_enu2cam:
            pc_np, P_gt_np = enu2cam(pc_np, P_gt_np)

        # transform to camera frame, calibrate depth
        pc_cam_np, valid_mask = get_inside_img_mask(pc_np, P_gt_np, K_np, H, W)
        scale_factor = np.mean(pc_cam_np[2, valid_mask]) / np.mean(pc_monodepth_np[2, :])
        pc_monodepth_np *= scale_factor
        print("scale_factor=", scale_factor)

        P_pred_np, fitness = icp_random_init(pc_np, pc_monodepth_np, 60, is_plot)

        t_diff, r_diff = get_P_diff(P_pred_np, P_gt_np)
        if r_diff > 180:
            r_diff = 360 - r_diff
        t_diff_avg += t_diff
        r_diff_avg += r_diff

        print('%s - fitness: %.1f, T: %.1f, R:%.1f' % (filename, fitness, t_diff, r_diff))
        P_pred_all_np[i, :, :] = P_pred_np
        P_gt_all_np[i, :, :] = P_gt_np
        cost_all_np[i] = fitness

        current_t_diff_avg = t_diff_avg / counter
        current_r_diff_avg = r_diff_avg / counter
        print("%d frame average translation / rotation error: [%.2f, %.2f]" % (counter, current_t_diff_avg, current_r_diff_avg))


    t_diff_avg = t_diff_avg / counter
    r_diff_avg = r_diff_avg / counter
    print("%d frame average translation / rotation error: [%.2f, %.2f]" % (counter, t_diff_avg, r_diff_avg))

    np.save(os.path.join(root_path, 'P_pred_all_np.npy'), P_pred_all_np)
    np.save(os.path.join(root_path, 'P_gt_all_np.npy'), P_gt_all_np)
    np.save(os.path.join(root_path, 'cost_all_np.npy'), cost_all_np)




if __name__ == '__main__':
    main()