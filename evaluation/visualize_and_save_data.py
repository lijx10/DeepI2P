import open3d
import time
import numpy as np
import math
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import cv2

import matplotlib
matplotlib.use('TkAgg')

from models.multimodal_classifier import MMClassifer, MMClassiferCoarse
from data.kitti_pc_img_pose_loader import KittiLoader
from data.oxford_pc_img_pose_loader import OxfordLoader
from data.nuscenes_pc_img_pose_loader import nuScenesLoader
from util import vis_tools
import kitti.options
import oxford.options
import nuscenes_t.options
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


if __name__ == "__main__":
    # root_path = '/ssd/jiaxin/point-img-feature/kitti/save/1.30-noTranslation'
    root_path = '/home/tohar/repos/point-img-feature/oxford/workspace/640x384-noCrop'
    # root_path = '/ssd/jiaxin/point-img-feature/nuscenes_t/save/3.3-160x320-accu'

    dataset = 'oxford'

    if dataset == 'kitti':
        opt = kitti.options.Options()
    elif dataset == 'oxford':
        opt = oxford.options.Options()
    elif dataset == 'nuscenes':
        opt = nuscenes_t.options.Options()

    opt.gpu_ids = [0]
    opt.device = torch.device('cuda', opt.gpu_ids[0])


    visualization_output_folder = 'visualization'
    visualization_output_path = os.path.join(root_path, visualization_output_folder)
    if not os.path.exists(visualization_output_path):
        os.mkdir(visualization_output_path)
    data_output_folder = 'data'
    data_output_path = os.path.join(root_path, data_output_folder)
    if not os.path.exists(data_output_path):
        os.mkdir(data_output_path)

    is_plot = False
    is_save_visualization = True
    is_save_data = True
    iter_max = 1e9
    circle_size = 1

    if dataset == 'kitti':
        testset = KittiLoader(opt.dataroot, 'val_random_Ry', opt)
    elif dataset == 'oxford':
        testset = OxfordLoader(opt.dataroot, 'val_random_Ry', opt)
    elif dataset == 'nuscenes':
        testset = nuScenesLoader(opt.dataroot, 'val_random_Ry', opt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False,
                                             num_workers=opt.dataloader_threads, pin_memory=True)
    print('#testing point clouds = %d' % len(testset))

    if opt.is_fine_resolution:
        model = MMClassifer(opt, writer=None)
    else:
        model = MMClassiferCoarse(opt, writer=None)
    model_path = os.path.join(root_path, 'checkpoints/best.pth')
    print(model_path)
    model.load_model(model_path)
    model.detector.eval()

    counter = 0
    coarse_accuracy_sum = 0
    fine_accuracy_sum = 0
    for i, data in enumerate(testloader):
        pc, intensity, sn, node_a, node_b, \
        P, img, K, t_ij = data

        B, H, W = img.size(0), img.size(2), img.size(3)
        N = pc.size(2)
        H_fine = int(round(H / opt.img_fine_resolution_scale))
        W_fine = int(round(W / opt.img_fine_resolution_scale))

        model.set_input(pc, intensity, sn, node_a, node_b,
                        P, img, K)

        # BxN, BxN
        if opt.is_fine_resolution:
            coarse_prediction, fine_prediction = model.inference_pass()
        else:
            coarse_prediction = model.inference_pass()
            fine_prediction = coarse_prediction

        # transform and project point cloud
        pc_homo = torch.cat((pc,
                             torch.ones((B, 1, N), dtype=pc.dtype)),
                            dim=1)  # Bx4xN
        P_pc_homo = torch.matmul(P, pc_homo)  # Bx4x4 * Bx4xN -> Bx4xN
        P_pc = P_pc_homo[:, 0:3, :]  # Bx3xN
        KP_pc = torch.matmul(K, P_pc)
        KP_pc_pxpy = KP_pc[:, 0:2, :] / KP_pc[:, 2:3, :]  # Bx3xN -> Bx2xN

        # compute ground truth
        x_inside_mask = (KP_pc_pxpy[:, 0:1, :] >= 0) \
                        & (KP_pc_pxpy[:, 0:1, :] <= W - 1)  # Bx1xN_pc
        y_inside_mask = (KP_pc_pxpy[:, 1:2, :] >= 0) \
                        & (KP_pc_pxpy[:, 1:2, :] <= H - 1)  # Bx1xN_pc
        z_inside_mask = P_pc_homo[:, 2:3, :] > 0.1  # Bx1xN_pc
        inside_mask = x_inside_mask & y_inside_mask & z_inside_mask  # Bx1xN_pc
        coarse_label_np = inside_mask.squeeze(1).to(dtype=torch.long).cpu().numpy()

        coarse_prediction_np = coarse_prediction.cpu().numpy()
        fine_prediction_np = fine_prediction.cpu().numpy()
        pc_np = pc.cpu().numpy()  # Bx3xN
        P_pc_np = P_pc.cpu().numpy()  # Bx3xN
        KP_pc_pxpy_np = KP_pc_pxpy.cpu().numpy()  # Bx2xN
        imgs_np = img.detach().round()\
            .to(dtype=torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()  # Bx3xHxW -> BxHxWx3
        t_ij_np = t_ij.cpu().numpy()  # Bx3

        for b in range(B):
            pc_vis_np = pc_np[b, :, :]  # 3xN
            P_pc_vis_np = P_pc_np[b, :, :]  # 3xN
            img_vis_np = imgs_np[b, :, :, :]  # HxWx3
            coarse_prediction_vis_np = coarse_prediction_np[b, :]  # N
            fine_prediction_vis_np = fine_prediction_np[b, :]  # N
            KP_pc_pxpy_vis_np = KP_pc_pxpy_np[b, :, :]  # 2xN

            coarse_label_vis_np = coarse_label_np[b, :]  # N

            # compute fine gt
            pc_pxpy_vis_np_scaled_int = np.floor(KP_pc_pxpy_vis_np / opt.img_fine_resolution_scale).astype(np.int)  # 2xN
            fine_labels_vis_np = pc_pxpy_vis_np_scaled_int[0, :] + pc_pxpy_vis_np_scaled_int[1, :] * W_fine  # N

            # print accuracy
            current_coarse_accuracy = np.mean((coarse_prediction_vis_np == coarse_label_vis_np).astype(np.float))
            gt_in_img_mask = coarse_label_vis_np == 1
            current_fine_accuracy = np.mean((fine_prediction_vis_np[gt_in_img_mask] == fine_labels_vis_np[gt_in_img_mask]).astype(np.float))
            print('%d coarse accuracy %.4f, fine accuracy %.4f' % (counter, current_coarse_accuracy, current_fine_accuracy))
            coarse_accuracy_sum += current_coarse_accuracy
            fine_accuracy_sum += current_fine_accuracy
            counter += 1

            if opt.is_fine_resolution:
                img_vis_fine_np = vis_tools.get_classification_visualization(KP_pc_pxpy_vis_np,
                                                                             coarse_prediction_vis_np, fine_prediction_vis_np,
                                                                             coarse_label_vis_np, fine_labels_vis_np,
                                                                             img_vis_np,
                                                                             opt.img_fine_resolution_scale,
                                                                             H_delta=100, W_delta=100,
                                                                             circle_size=1,
                                                                             t_ij_np=t_ij_np[b, :])
            else:
                img_vis_fine_np = vis_tools.get_classification_visualization_coarse(KP_pc_pxpy_vis_np,
                                                                             coarse_prediction_vis_np,
                                                                             coarse_label_vis_np,
                                                                             img_vis_np,
                                                                             H_delta=100, W_delta=100,
                                                                             circle_size=1,
                                                                             t_ij_np=t_ij_np[b, :])

            if is_save_visualization:
                cv2.imwrite(os.path.join(visualization_output_path, '%06d_%02d_img.png' % (i, b)),
                            cv2.cvtColor(img_vis_np, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(visualization_output_path, '%06d_%02d_prediction.png' % (i, b)),
                            cv2.cvtColor(img_vis_fine_np, cv2.COLOR_RGB2BGR))

            if is_save_data:
                output_np = np.concatenate((pc_vis_np,
                                            np.expand_dims(coarse_prediction_vis_np, axis=0),
                                            np.expand_dims(coarse_label_vis_np, axis=0),
                                            np.expand_dims(fine_prediction_vis_np, axis=0),
                                            np.expand_dims(fine_labels_vis_np, axis=0)),
                                           axis=0)
                np.save(os.path.join(data_output_path, '%06d_%02d_pc_label.npy' % (i, b)),
                        output_np)
                np.save(os.path.join(data_output_path, '%06d_%02d_K.npy' % (i, b)),
                        K.cpu().numpy()[b, ...])
                np.save(os.path.join(data_output_path, '%06d_%02d_P.npy' % (i, b)),
                        P.cpu().numpy()[b, ...])


            # draw visualizations
            if is_plot:
                plt.figure()
                plt.imshow(img_vis_np)
                plt.figure()
                plt.imshow(img_vis_fine_np)

                fig_prediction = plt.figure(figsize=(9, 9))
                ax_prediction = Axes3D(fig_prediction)
                ax_prediction.set_title("coarse prediction")
                vis_tools.plot_pc(P_pc_vis_np, color=coarse_prediction_vis_np, size=6, ax=ax_prediction)

                fig_gt = plt.figure(figsize=(9, 9))
                ax_gt = Axes3D(fig_gt)
                ax_gt.set_title("coarse label")
                vis_tools.plot_pc(P_pc_vis_np, color=coarse_label_vis_np, size=6, ax=ax_gt)

                # modify fine_labels for plt.plot
                # inside_mask = (coarse_prediction_vis_np == 1).astype(np.float32)
                # fine_prediction_vis_np = (fine_prediction_vis_np + 1) * inside_mask
                # vis_tools.plot_pc(pc_vis_np, color=fine_prediction_vis_np, size=6)

                plt.show()

        if i >= iter_max:
            break

    print('Overall coarse accuracy %.4f, fine accuracy %.4f' % (coarse_accuracy_sum/counter,
                                                                fine_accuracy_sum/counter))









