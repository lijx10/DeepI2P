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


def generate_gaussian_random_transform(tx_sigma, ty_sigma, tz_sigma,
                                       rx_sigma, ry_sigma, rz_sigma):
    t = [random.gauss(0, tx_sigma),
         random.gauss(0, ty_sigma),
         random.gauss(0, tz_sigma)]
    angles = [random.gauss(0, rx_sigma),
              random.gauss(0, ry_sigma),
              random.gauss(0, rz_sigma)]
    rotation_mat = augmentation.angles2rotation_matrix(angles)
    P_random = np.identity(4, dtype=np.float)
    P_random[0:3, 0:3] = rotation_mat
    P_random[0:3, 3] = t
    return P_random


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
    return inside_mask[0]


def get_P_diff(P_pred_np, P_gt_np):
    P_diff = np.dot(np.linalg.inv(P_pred_np), P_gt_np)
    t_diff = np.linalg.norm(P_diff[0:3, 3])

    r_diff = P_diff[0:3, 0:3]
    R_diff = Rotation.from_matrix(r_diff)
    angles_diff = np.sum(np.abs(R_diff.as_euler('xzy', degrees=True)))

    return t_diff, angles_diff


def solve_P_random_init(pc_np, coarse_predictions_np, K_np, H, W,
                        iteration_num, is_2d):
    min_cost = 1e10
    P_final = None
    residuals_final = None
    for i in range(iteration_num):
        P_random_init = generate_uniform_random_transform(0, 0, 0,
                                                          0, 2 * math.pi, 0)

        P_pred_np, final_cost, residuals = FrustumRegistration.solvePGivenK(pc_np,
                                                                            coarse_predictions_np,
                                                                            K_np,
                                                                            P_random_init[0:3, 0:3],
                                                                            P_random_init[0:3, 3],
                                                                            H,
                                                                            W,
                                                                            [-10, -1, -10],
                                                                            [10, 1, 10],
                                                                            500,
                                                                            False,  # is_debug
                                                                            is_2d  # is_2d
                                                                            )
        if final_cost < min_cost:
            P_final = P_pred_np
            residuals_final = residuals
            min_cost = final_cost
    return P_final, min_cost, residuals_final


def solver_wrapper(pc_np, coarse_predictions_np, K_np,
                   R_init, t_init, H, W,
                   t_xyz_lower_bound, t_xyz_upper_bound, max_iter,
                   is_debug, is_2d,
                   final_dict):
    P_pred_np, final_cost, residuals = FrustumRegistration.solvePGivenK(pc_np, coarse_predictions_np, K_np,
                                                                        R_init, t_init, H, W,
                                                                        t_xyz_lower_bound, t_xyz_upper_bound, max_iter,
                                                                        is_debug, is_2d)
    if final_cost < final_dict['cost']:
        final_dict['P'] = P_pred_np
        final_dict['residuals'] = residuals
        final_dict['cost'] = final_cost


def solve_P_random_perturb(pc_np, coarse_predictions_np, K_np, H, W,
                           init_t_amplitude, init_y_angle, ry_sigma, t_lowerbound, t_upperbound, iteration_num,
                           is_2d,
                           thread_num):

    manager = multiprocessing.Manager()
    final_dict = manager.dict()
    final_dict['P'] = None
    final_dict['cost'] = 1e20
    final_dict['residuals'] = None

    batch_num = math.ceil(iteration_num / thread_num)
    last_batch_element_num = iteration_num - thread_num * math.floor(iteration_num / thread_num)
    for b in range(batch_num):
        threads = []
        if b!= batch_num-1:
            thread_num_in_batch = thread_num
        else:
            thread_num_in_batch = last_batch_element_num

        for i in range(thread_num_in_batch):
            ry_init = init_y_angle + random.gauss(0, ry_sigma)
            t_init = np.asarray([0, 0, random.uniform(-init_t_amplitude, init_t_amplitude)])

            threads.append(multiprocessing.Process(target=solver_wrapper,
                                                   args=(pc_np,
                                                         coarse_predictions_np,
                                                         K_np,
                                                         ry_init,
                                                         t_init,
                                                         H,
                                                         W,
                                                         t_lowerbound,
                                                         t_upperbound,
                                                         500,
                                                         False,  # is debug
                                                         is_2d,  # is_2d
                                                         final_dict)))
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    return final_dict['P'], final_dict['cost'], final_dict['residuals']


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


if __name__=='__main__':
    # root_path = '/ssd/jiaxin/point-img-feature/kitti/save/1.30-noTranslation'
    # root_path = '/ssd/jiaxin/point-img-feature/oxford/save/1.16-fine-wGround-nocrop-0.5x384x640'
    # root_path = '/ssd/jiaxin/point-img-feature/nuscenes_t/save/3.3-160x320-accu'

    root_path = '/home/tohar/repos/point-img-feature/oxford/workspace/640x384-noCrop'

    visualization_output_folder = 'visualization'
    visualization_output_path = os.path.join(root_path, visualization_output_folder)
    data_output_folder = 'data'
    data_output_path = os.path.join(root_path, data_output_folder)

    is_plot = False
    is_2d = True
    H = 384  # kitti=160, oxford=288/192/384, nuscenes 160
    W = 640  # kitti=512, oxford=512/320/640, nuscenes 320
    is_enu2cam = 'nuscene' in root_path

    t_diff_avg = 0
    r_diff_avg = 0
    counter = 0

    filename_list = [f[0:9] for f in os.listdir(data_output_path) if os.path.isfile(os.path.join(data_output_path, f))]
    filename_list = list(set(filename_list))
    filename_list.sort()

    random_permute_idx = np.random.permutation(len(filename_list))
    filename_list = [filename_list[i] for i in random_permute_idx]

    # save to disk
    P_pred_all_np = np.zeros((len(filename_list), 4, 4))
    P_gt_all_np = np.zeros((len(filename_list), 4, 4))
    cost_all_np = np.zeros((len(filename_list)))
    for i in range(0, len(filename_list), 30):
        filename = filename_list[i]
        counter += 1

        # if filename != '000400_00':
        #     continue

        point_data_np = np.load(os.path.join(data_output_path, filename+'_pc_label.npy'))
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

        # debug code to ensure that the label is correct
        # inside_mask = get_inside_img_mask(points_np, P_gt_np, K_np, H, W)
        # inside_mask = inside_mask.astype(np.int32)
        # assert 0 == np.sum((inside_mask != coarse_labels_np).astype(np.int))

        R_gt = Rotation.from_matrix(P_gt_np[0:3, 0:3])
        angles_gt = R_gt.as_euler('yxz', degrees=False)
        ry_gt = angles_gt[0]

        # P_pred_np, final_cost, residuals = FrustumRegistration.solvePGivenK(pc_np,
        #                                                                     # coarse_predictions_np,
        #                                                                     coarse_labels_np,
        #                                                                     K_np,
        #                                                                     ry_gt,
        #                                                                     P_gt_np[0:3, 3],
        #                                                                     H,
        #                                                                     W,
        #                                                                     [-100, -100, -100],
        #                                                                     [100, 100, 100],
        #                                                                     500,
        #                                                                     True,
        #                                                                     is_2d)

        # P_pred_np, final_cost, residuals = solve_P_random_init(pc_np, coarse_predictions_np, K_np, H, W, 10, is_2d)

        if np.sum((coarse_predictions_np==1).astype(np.int)) == 0:
            P_pred_np = np.identity(4)
            final_cost = 1e4
            residuals = 0
        else:
            P_init, init_y_angle, pc_np, coarse_predictions_np = get_initial_guess(pc_np, coarse_predictions_np)
            P_pred_np, \
            final_cost, \
            residuals = solve_P_random_perturb(pc_np, coarse_predictions_np, K_np, H, W,
                                               init_t_amplitude=10,
                                               init_y_angle=init_y_angle,
                                               ry_sigma=10*math.pi/180,
                                               t_lowerbound=[-5, -0.1, -10], t_upperbound=[5, 0.1, 10],
                                               iteration_num=60, is_2d=is_2d,
                                               thread_num=8)

        # print(ry_gt)
        # print(init_y_angle)

        t_diff, r_diff = get_P_diff(P_pred_np, P_gt_np)
        t_diff_avg += t_diff
        r_diff_avg += r_diff

        print('%s - cost: %.1f, T: %.1f, R:%.1f' % (filename, final_cost, t_diff, r_diff))
        P_pred_all_np[i, :, :] = P_pred_np
        P_gt_all_np[i, :, :] = P_gt_np
        cost_all_np[i] = final_cost

        if is_plot:
            print("P_pred_np")
            print(P_pred_np)
            print("P_gt_np")
            print(P_gt_np)

            img_vis_np = cv2.cvtColor(cv2.imread(os.path.join(visualization_output_path, filename+'_img.png')), cv2.COLOR_BGR2RGB)
            img_vis_fine_np = cv2.cvtColor(cv2.imread(os.path.join(visualization_output_path, filename+'_prediction.png')), cv2.COLOR_BGR2RGB)

            img_reg_np = vis_tools.get_registration_visualization(pc_np,
                                                               P_pred_np,
                                                               K_np,
                                                               coarse_predictions_np,
                                                               img_vis_np)

            plt.figure()
            plt.imshow(img_vis_np)
            plt.figure()
            plt.imshow(img_vis_fine_np)
            plt.figure()
            plt.imshow(img_reg_np)

            fig_prediction = plt.figure(figsize=(9, 9))
            ax_prediction = Axes3D(fig_prediction)
            ax_prediction.set_title("coarse label")
            vis_tools.plot_pc(pc_np, color=coarse_predictions_np, size=6, ax=ax_prediction)

            # fig_gt = plt.figure(figsize=(9, 9))
            # ax_gt = Axes3D(fig_gt)
            # ax_gt.set_title("registration")
            # vis_tools.plot_pc(pc_vis_np, color=coarse_label_vis_np, size=6, ax=ax_gt)

            plt.show()

        current_t_diff_avg = t_diff_avg / counter
        current_r_diff_avg = r_diff_avg / counter
        print("%d frame average translation / rotation error: [%.2f, %.2f]" % (counter, current_t_diff_avg, current_r_diff_avg))

    t_diff_avg = t_diff_avg / counter
    r_diff_avg = r_diff_avg / counter
    print("%d frame average translation / rotation error: [%.2f, %.2f]" % (counter, t_diff_avg, r_diff_avg))

    np.save(os.path.join(root_path, 'P_pred_all_np.npy'), P_pred_all_np)
    np.save(os.path.join(root_path, 'P_gt_all_np.npy'), P_gt_all_np)
    np.save(os.path.join(root_path, 'cost_all_np.npy'), cost_all_np)
