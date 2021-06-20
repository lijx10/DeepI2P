import numpy as np
import os
import math

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from evaluation.registration_pnp import get_P_diff

if __name__ == '__main__':
    # registration_result_folder = '/ssd/jiaxin/point-img-feature/kitti/save/1.12-networkOK/registration_results/lsq-2d'
    registration_result_folder = '/ssd/jiaxin/point-img-feature/oxford/save/1.16-fine-wGround-nocrop-0.5x384x640/registration_results/lsq-2d'
    # registration_result_folder = '/ssd/jiaxin/point-img-feature/nuscenes_t/save/3.3-160x320-accu/registration_results/random'

    P_pred_all_np = np.load(os.path.join(registration_result_folder, 'P_pred_all_np.npy'))
    P_gt_all_np = np.load(os.path.join(registration_result_folder, 'P_gt_all_np.npy'))
    cost_all_np = np.load(os.path.join(registration_result_folder, 'cost_all_np.npy'))

    valid_mask = cost_all_np > 1e-6
    P_pred_all_np = P_pred_all_np[valid_mask, ...]
    P_gt_all_np = P_gt_all_np[valid_mask, ...]
    cost_all_np = cost_all_np[valid_mask]

    reg_num = P_pred_all_np.shape[0]
    t_diff_all_np = np.zeros(reg_num)
    r_diff_all_np = np.zeros(reg_num)

    for i in range(reg_num):
        t_diff, r_diff = get_P_diff(P_pred_all_np[i], P_gt_all_np[i])
        t_diff_all_np[i] = t_diff
        r_diff_all_np[i] = r_diff


    success_mask = np.logical_and(t_diff_all_np < 2, r_diff_all_np < 5)
    success_rate = np.mean(success_mask.astype(np.float))

    rte_sigma = math.sqrt(np.var(t_diff_all_np))
    rre_sigma = math.sqrt(np.var(r_diff_all_np))
    # print result
    print('RTE %.2f +- %.2f, RRE %.2f +- %.2f, success rate %.2f' % (np.mean(t_diff_all_np),
                                                                     rte_sigma,
                                                                     np.mean(r_diff_all_np),
                                                                     rre_sigma,
                                                                     success_rate*100))

    plt.figure()
    plt.title('cost vs translation-error')
    plt.plot(cost_all_np, t_diff_all_np, 'bo', markersize=1)

    plt.figure()
    plt.title('cost vs rotation-error')
    plt.plot(cost_all_np, r_diff_all_np, 'bo', markersize=1)

    plt.figure(figsize=(3, 3))
    plt.title('Oxford RTE Histogram')
    plt.hist(t_diff_all_np, bins='auto', range=[0, 15], density=True, cumulative=False)

    plt.figure(figsize=(3, 3))
    plt.title('Oxford RRE Histogram')
    plt.hist(r_diff_all_np, bins='auto', range=[0, 30], density=True, cumulative=False)

    plt.show()

