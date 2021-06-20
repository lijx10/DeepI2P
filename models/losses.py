import torch
import torch.nn as nn
from torch.nn import functional as F
import numbers
import numpy as np
import math

from kitti.options import Options
from data import kitti_helper

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from util import vis_tools

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


class HeatMapLoss(nn.Module):
    def __init__(self, opt):
        super(HeatMapLoss, self).__init__()
        self.opt = opt
        self.gaussian_filter = GaussianSmoothing(channels=1,
                                                 kernel_size=self.opt.img_heatmap_nms_size,
                                                 sigma=self.opt.img_heatmap_nms_size / 6,
                                                 dim=2).to(opt.device)

    def forward(self, heatmap, pc_keypoints_pxpy):
        """

        :param heatmap: Bx1xHxW
        :param pc_keypoints_pxpy: Bx2xM
        :return:
        """
        # ---------- Euclidean loss over the whole heatmap
        pc_keypoints_pxpy = torch.round(pc_keypoints_pxpy).to(dtype=torch.long)
        B, H, W = heatmap.size(0), heatmap.size(2), heatmap.size(3)
        N = H * W
        M = pc_keypoints_pxpy.size(2)
        pc_keypoints_idx = pc_keypoints_pxpy[:, 1:2, :] * W + pc_keypoints_pxpy[:, 0:1, :]  # Bx1xM
        pc_keypoints_idx = torch.clamp(pc_keypoints_idx, min=0, max=N-1)  # Bx1xM
        pc_keypoints_score = torch.ones_like(pc_keypoints_idx).to(dtype=torch.float).fill_(1)  # Bx1xM

        pc_heatmap = torch.zeros_like(heatmap).view(B, 1, N).contiguous()  # Bx1xHxW -> Bx1xN
        pc_heatmap.scatter_(dim=2, index=pc_keypoints_idx, src=pc_keypoints_score)  # Bx1xN
        pc_heatmap = pc_heatmap.view(B, 1, H, W).contiguous()

        padding_size = int(math.floor(self.opt.img_heatmap_nms_size / 2))
        pc_heatmap = F.pad(pc_heatmap, (padding_size, padding_size, padding_size, padding_size), mode='reflect')
        pc_heatmap = self.gaussian_filter(pc_heatmap)  # Bx1xHxW

        heatmap_diff = heatmap - pc_heatmap
        heatmap_loss = torch.mean(torch.abs(heatmap_diff))

        return heatmap_loss, pc_heatmap


# ============== detector loss ============== begin ======
def get_chamfer_loss(pc_src_input, pc_dst_input):
    '''
            :param pc_src_input: BxDxM Tensor in GPU
            :param pc_dst_input: BxDxN Tensor in GPU
            :return:
            '''

    B, D, M = pc_src_input.size()[0], pc_src_input.size()[1], pc_src_input.size()[2]
    N = pc_dst_input.size()[2]

    pc_src_input_expanded = pc_src_input.unsqueeze(3).expand(B, D, M, N)
    pc_dst_input_expanded = pc_dst_input.unsqueeze(2).expand(B, D, M, N)

    # the gradient of norm is set to 0 at zero-input. There is no need to use custom norm anymore.
    diff = torch.norm(pc_src_input_expanded - pc_dst_input_expanded, dim=1, keepdim=False)  # BxMxN

    # pc_src vs selected pc_dst, M
    src_dst_min_dist, _ = torch.min(diff, dim=2, keepdim=False)  # BxM
    forward_loss = src_dst_min_dist.mean()

    # pc_dst vs selected pc_src, N
    dst_src_min_dist, _ = torch.min(diff, dim=1, keepdim=False)  # BxN
    backward_loss = dst_src_min_dist.mean()

    chamfer_pure = forward_loss + backward_loss
    chamfer_weighted = chamfer_pure

    return forward_loss + backward_loss, chamfer_pure, chamfer_weighted


def get_chamfer_loss_prob(pc_src_input, pc_dst_input, sigma_src=None, sigma_dst=None):
    '''
            :param pc_src_input: BxDxM Tensor in GPU
            :param pc_dst_input: BxDxN Tensor in GPU
            :param sigma_src: BxM Tensor in GPU
            :param sigma_dst: BxN Tensor in GPU
            :return:
            '''
    B, D, M = pc_src_input.size()[0], pc_src_input.size()[1], pc_src_input.size()[2]
    N = pc_dst_input.size()[2]

    pc_src_input_expanded = pc_src_input.unsqueeze(3).expand(B, D, M, N)
    pc_dst_input_expanded = pc_dst_input.unsqueeze(2).expand(B, D, M, N)

    # the gradient of norm is set to 0 at zero-input. There is no need to use custom norm anymore.
    diff = torch.norm(pc_src_input_expanded - pc_dst_input_expanded, dim=1, keepdim=False)  # BxMxN

    if sigma_src is None or sigma_dst is None:
        # pc_src vs selected pc_dst, M
        src_dst_min_dist, _ = torch.min(diff, dim=2, keepdim=False)  # BxM
        forward_loss = src_dst_min_dist.mean()

        # pc_dst vs selected pc_src, N
        dst_src_min_dist, _ = torch.min(diff, dim=1, keepdim=False)  # BxN
        backward_loss = dst_src_min_dist.mean()

        chamfer_pure = forward_loss + backward_loss
        chamfer_weighted = chamfer_pure
    else:
        # pc_src vs selected pc_dst, M
        src_dst_min_dist, src_dst_I = torch.min(diff, dim=2, keepdim=False)  # BxM, BxM
        selected_sigma_dst = torch.gather(sigma_dst, dim=1, index=src_dst_I)  # BxN -> BxM
        # sigma_src_dst = (sigma_src + selected_sigma_dst) / 2
        sigma_src_dst = selected_sigma_dst
        forward_loss = (torch.log(sigma_src_dst) + src_dst_min_dist / sigma_src_dst).mean()

        # pc_dst vs selected pc_src, N
        dst_src_min_dist, dst_src_I = torch.min(diff, dim=1, keepdim=False)  # BxN, BxN
        selected_sigma_src = torch.gather(sigma_src, dim=1, index=dst_src_I)  # BxM -> BxN
        # sigma_dst_src = (sigma_dst + selected_sigma_src) / 2
        sigma_dst_src = selected_sigma_src
        backward_loss = (torch.log(sigma_dst_src) + dst_src_min_dist / sigma_dst_src).mean()

        # loss that do not involve in optimization
        chamfer_pure = (src_dst_min_dist.mean() + dst_src_min_dist.mean()).detach()
        weight_src_dst = (1.0 / sigma_src_dst) / torch.mean(1.0 / sigma_src_dst)
        weight_dst_src = (1.0 / sigma_dst_src) / torch.mean(1.0 / sigma_dst_src)
        chamfer_weighted = ((weight_src_dst * src_dst_min_dist).mean() +
                            (weight_dst_src * dst_src_min_dist).mean()).detach()

    return forward_loss + backward_loss, chamfer_pure, chamfer_weighted


def get_keypoint_on_pc_loss(pc_src_input, pc_dst_input):
    '''
    :param pc_src_input: BxDxM Variable in GPU
    :param pc_dst_input: BxDxN Variable in GPU
    :return:
    '''

    B, D, M = pc_src_input.size()[0], pc_src_input.size()[1], pc_src_input.size()[2]
    N = pc_dst_input.size()[2]

    pc_src_input_expanded = pc_src_input.unsqueeze(3).expand(B, D, M, N)
    pc_dst_input_expanded = pc_dst_input.unsqueeze(2).expand(B, D, M, N)

    diff = torch.norm(pc_src_input_expanded - pc_dst_input_expanded, dim=1, keepdim=False)  # BxMxN

    # pc_src vs selected pc_dst, M
    src_dst_min_dist, _ = torch.min(diff, dim=2, keepdim=False)  # BxM

    return torch.mean(src_dst_min_dist)


def get_img_keypoint_offset_norm_loss(img_keypoints_raw, box_dx, box_dy):
    img_keypoints_raw_abs = torch.abs(img_keypoints_raw)  # Bx2xM
    apply_mask_x = img_keypoints_raw_abs[:, 0, :] > box_dx  # BxM
    apply_mask_y = img_keypoints_raw_abs[:, 1, :] > box_dy  # BxM
    apply_mask = (apply_mask_x | apply_mask_y).to(dtype=torch.float32)  # BxM

    img_keypoints_raw_norm = torch.norm(img_keypoints_raw, p=2, dim=1, keepdim=False)  # BxM
    return torch.mean(apply_mask * img_keypoints_raw_norm)


# ============== detector loss ============== end ======
