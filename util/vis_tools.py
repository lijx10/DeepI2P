import random
import numbers
from PIL import Image, ImageMath
import os
import os.path
import numpy as np
import struct
import math
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def plot_pc(pc_np, birds_view=False, color=None, size=1.0, ax=None, cmap=cm.jet, is_equal_axes=True,
            elev=-45, azim=-90):
    """

    :param pc_np: 3xN
    :param birds_view:
    :param color:
    :param size:
    :param ax:
    :param cmap:
    :param is_equal_axes:
    :return:
    """
    if ax is None:
        fig = plt.figure(figsize=(9, 9))
        ax = Axes3D(fig)
    if type(color) == np.ndarray:
        ax.scatter(pc_np[0, :], pc_np[1, :], pc_np[2, :], s=size, c=color, cmap=cmap, edgecolors='none')
    else:
        ax.scatter(pc_np[0, :], pc_np[1, :], pc_np[2, :], s=size, c=color, edgecolors='none')

    if is_equal_axes:
        axisEqual3D(ax)
    if True == birds_view:
        ax.view_init(elev=0, azim=-90)
    else:
        ax.view_init(elev=elev, azim=azim)
    # ax.invert_yaxis()

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return ax


def fig_to_np(fig):
    # draw the figure
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def visualization_list_to_grid(visualization_fig_np_list, col=2):
    """
    Put a list of np.uint8 array into a grid for visualization
    :param visualization_fig_np_list: List: [np.ndarray], list of HxWx3 np.uint8 array
    :param col: number of column in the grid
    :return:
    """
    # handle exceptions
    if len(visualization_fig_np_list) == 0:
        return np.zeros((3, 3), dtype=np.uint8)

    row = int(math.ceil(len(visualization_fig_np_list) / col))
    img_h, img_w, img_c = visualization_fig_np_list[0].shape

    visualization_fig_np = np.full((row*img_h, col*img_w, img_c), fill_value=255, dtype=np.uint8)
    for i in range(row):
        for j in range(col):
            idx = i * col + j
            if idx >= len(visualization_fig_np_list):
                continue
            else:
                visualization_fig_np[i*img_h:(i+1)*img_h, j*img_w:(j+1)*img_w, :] = visualization_fig_np_list[idx]

    return visualization_fig_np


def get_registration_visualization(pc_np,
                                   P_np,
                                   K_np,
                                   coarse_predictions_np,
                                   img_vis_np,
                                   H_delta=100, W_delta=100,
                                   circle_size=1):
    # projection
    pc_np_homo = np.concatenate((pc_np,
                                 np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)),
                                axis=0)
    P_points_np = np.dot(P_np, pc_np_homo)[0:3, :]
    K_pc_np = np.dot(K_np, P_points_np)
    pc_pxpy_np = K_pc_np[0:2, :] / K_pc_np[2:3, :]  # Bx3xN -> Bx2xN

    # prepare image
    N = coarse_predictions_np.shape[0]
    H, W = img_vis_np.shape[0], img_vis_np.shape[1]
    H_large = H + int(H_delta * 2)
    W_large = W + int(W_delta * 2)
    img_vis_fine_np = np.zeros((H_large, W_large, 3), dtype=np.uint8) + 255
    img_vis_fine_np[H_delta:H_delta + H, W_delta:W_delta + W] = img_vis_np

    # draw points
    for n in range(N):
        px = pc_pxpy_np[0, n]
        py = pc_pxpy_np[1, n]
        if math.isinf(px) or math.isinf(py) or math.isnan(px) or math.isnan(py):
            continue
        px = int(round(px))
        py = int(round(py))
        px_offset = int(px + W_delta)
        py_offset = int(py + H_delta)
        if px_offset < 0 or px_offset >= W_large - 1 or py_offset < 0 or py_offset >= H_large - 1\
                or K_pc_np[2, n] < 0:
            continue

        if coarse_predictions_np[n] == 1:
            cv2.circle(img_vis_fine_np,
                       (px_offset, py_offset),
                       circle_size,
                       (255, 0, 0),
                       -1)
        else:
            cv2.circle(img_vis_fine_np,
                       (px_offset, py_offset),
                       circle_size,
                       (0, 0, 255),
                       -1)
    return img_vis_fine_np

def get_classification_visualization_coarse(pc_pxpy_np,
                                     coarse_predictions_np,
                                     coarse_labels_np,
                                     img_vis_np,
                                     H_delta=100, W_delta=100,
                                     circle_size=1,
                                     t_ij_np=None):
    """
    :param pc_pxpy_np: 2xN
    :param coarse_predictions_np: N
    :param coarse_labels_np: N
    :param img_vis_np: HxWx3
    :param H_delta: scalar
    :param W_delta: scalar
    :param circle_size: scalar
    :param t_ij_np: B, np.ndarray
    :return:
    """
    # project points onto image for better fine resolution visualization
    # prepare images for visualization
    # enlarge the image plane to visualize false positive

    N = coarse_predictions_np.shape[0]
    H, W = img_vis_np.shape[0], img_vis_np.shape[1]

    H_large = H + int(H_delta * 2)
    W_large = W + int(W_delta * 2)
    img_vis_fine_np = np.zeros((H_large, W_large, 3), dtype=np.uint8) + 255
    img_vis_fine_np[H_delta:H_delta + H, W_delta:W_delta + W] = img_vis_np

    # draw coarse result
    for n in range(N):
        px = pc_pxpy_np[0, n]
        py = pc_pxpy_np[1, n]
        if math.isinf(px) or math.isinf(py) or math.isnan(px) or math.isnan(py):
            continue
        px = int(round(px))
        py = int(round(py))
        px_offset = int(px + W_delta)
        py_offset = int(py + H_delta)
        if px_offset < 0 or px_offset >= W_large - 1 or py_offset < 0 or py_offset >= H_large - 1:
            continue

        coarse_prediction_bn = coarse_predictions_np[n]
        coarse_label_bn = coarse_labels_np[n]

        if coarse_prediction_bn == 1 and coarse_label_bn == 1:
            # draw color based on the fine prediction and label,
            # green: correct fine-prediction
            # yellow: wrong fine-prediction
            cv2.circle(img_vis_fine_np,
                       (px_offset, py_offset),
                       circle_size,
                       (0, 255, 0),
                       -1)

        elif coarse_prediction_bn == 0 and coarse_label_bn == 1:
            # red: false negative
            cv2.circle(img_vis_fine_np,
                       (px_offset, py_offset),
                       circle_size,
                       (255, 0, 0),
                       -1)
        elif coarse_prediction_bn == 1 and coarse_label_bn == 0:
            # blue: false positive
            cv2.circle(img_vis_fine_np,
                       (px_offset, py_offset),
                       circle_size,
                       (0, 0, 255),
                       -1)

    # put additional text
    if t_ij_np is not None:
        cv2.putText(img_vis_fine_np,
                    'T_ij: [%.2f, %.2f, %.2f]' % (t_ij_np[0], t_ij_np[1], t_ij_np[2]),
                    org=(10, int(H_delta*0.5)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 0),
                    thickness=2)

    return img_vis_fine_np


def get_classification_visualization(pc_pxpy_np,
                                     coarse_predictions_np, fine_predictions_np,
                                     coarse_labels_np, fine_labels_np,
                                     img_vis_np,
                                     img_fine_resolution_scale,
                                     H_delta=100, W_delta=100,
                                     circle_size=1,
                                     t_ij_np=None):
    """
    :param pc_pxpy_np: 2xN
    :param coarse_predictions_np: N
    :param fine_predictions_np: N
    :param coarse_labels_np: N
    :param fine_labels_np: N
    :param img_vis_np: HxWx3
    :param H_delta: scalar
    :param W_delta: scalar
    :param circle_size: scalar
    :param t_ij_np: B, np.ndarray
    :return:
    """
    # project points onto image for better fine resolution visualization
    # prepare images for visualization
    # enlarge the image plane to visualize false positive

    N = coarse_predictions_np.shape[0]
    H, W = img_vis_np.shape[0], img_vis_np.shape[1]
    H_fine = int(round(H / img_fine_resolution_scale))
    W_fine = int(round(W / img_fine_resolution_scale))

    H_large = H + int(H_delta * 2)
    W_large = W + int(W_delta * 2)
    img_vis_fine_np = np.zeros((H_large, W_large, 3), dtype=np.uint8) + 255
    img_vis_fine_np[H_delta:H_delta + H, W_delta:W_delta + W] = img_vis_np

    # draw grid representing the fine resolution
    for h in range(1, H_fine, 1):
        cv2.line(img_vis_fine_np,
                 (0 + W_delta, h * img_fine_resolution_scale + H_delta),
                 (W - 1 + W_delta, h * img_fine_resolution_scale + H_delta),
                 (255, 255, 255),
                 1)
    for w in range(1, W_fine, 1):
        cv2.line(img_vis_fine_np,
                 (w * img_fine_resolution_scale + W_delta, 0 + H_delta),
                 (w * img_fine_resolution_scale + W_delta, H - 1 + H_delta),
                 (255, 255, 255),
                 1)

    # draw coarse result
    for n in range(N):
        px = pc_pxpy_np[0, n]
        py = pc_pxpy_np[1, n]
        if math.isinf(px) or math.isinf(py) or math.isnan(px) or math.isnan(py):
            continue
        px = int(round(px))
        py = int(round(py))
        px_offset = int(px + W_delta)
        py_offset = int(py + H_delta)
        if px_offset < 0 or px_offset >= W_large - 1 or py_offset < 0 or py_offset >= H_large - 1:
            continue

        coarse_prediction_bn = coarse_predictions_np[n]
        coarse_label_bn = coarse_labels_np[n]
        fine_prediction_bn = fine_predictions_np[n]
        fine_label_bn = fine_labels_np[n]

        if coarse_prediction_bn == 1 and coarse_label_bn == 1:
            # draw color based on the fine prediction and label,
            # green: correct fine-prediction
            # yellow: wrong fine-prediction
            if fine_prediction_bn == fine_label_bn:
                cv2.circle(img_vis_fine_np,
                           (px_offset, py_offset),
                           circle_size,
                           (0, 255, 0),
                           -1)
            else:
                cv2.circle(img_vis_fine_np,
                           (px_offset, py_offset),
                           circle_size,
                           (255, 255, 0),
                           -1)
        elif coarse_prediction_bn == 0 and coarse_label_bn == 1:
            # red: false negative
            cv2.circle(img_vis_fine_np,
                       (px_offset, py_offset),
                       circle_size,
                       (255, 0, 0),
                       -1)
        elif coarse_prediction_bn == 1 and coarse_label_bn == 0:
            # blue: false positive
            cv2.circle(img_vis_fine_np,
                       (px_offset, py_offset),
                       circle_size,
                       (0, 0, 255),
                       -1)

    # put additional text
    if t_ij_np is not None:
        cv2.putText(img_vis_fine_np,
                    'T_ij: [%.2f, %.2f, %.2f]' % (t_ij_np[0], t_ij_np[1], t_ij_np[2]),
                    org=(10, int(H_delta*0.5)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 0),
                    thickness=2)

    return img_vis_fine_np
