from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist

from deepi2p.vis_tools import plot_pc


def load_model():

    model_name = 'mono+stereo_640x192'

    device = torch.device("cuda")

    model_path = os.path.join("models", model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    return feed_width, feed_height, encoder, depth_decoder, device


def get_img_depth(image_path, feed_width, feed_height, encoder, depth_decoder, device):
    with torch.no_grad():

        # Load image and preprocess
        input_image = pil.open(image_path).convert('RGB')
        original_width, original_height = input_image.size
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)

        depth = torch.reciprocal(disp_resized)
        depth_np = depth[0, 0, :, :].unsqueeze(2).cpu().numpy()

    return depth_np


def grid_generation(H, W):
    x = np.linspace(0, W-1, W)
    y = np.linspace(0, H-1, H)
    xv, yv = np.meshgrid(x, y)  # HxW
    meshgrid_np = np.stack((xv, yv, np.ones_like(xv)), axis=2)

    return meshgrid_np


def convert_depth_to_pointcloud(depth_np, K_np):
    H, W, _ = depth_np.shape
    K_inv_np = np.linalg.inv(K_np)  # 3x3
    meshgrid_np = grid_generation(H, W)  # HxWx1

    xyz_homo = depth_np * meshgrid_np
    xyz_homo = np.transpose(np.resize(xyz_homo, (H*W, 3)))
    pc_np = np.dot(K_inv_np, xyz_homo)

    return pc_np



def main():
    root_folder = '/home/tohar/repos/point-img-feature/oxford/workspace/640x384-noCrop/'

    output_folder = os.path.join(root_folder, 'monodepth')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data_folder = os.path.join(root_folder, 'data')
    filename_list = [f[0:9] for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
    filename_list = list(set(filename_list))
    filename_list.sort()

    feed_width, feed_height, encoder, depth_decoder, device = load_model()

    for sample in filename_list:
        image_name = sample + '_img.png'
        image_path = os.path.join(root_folder, 'visualization', image_name)
        K_np = np.load(os.path.join(data_folder, sample + '_K.npy'))

        depth_np = get_img_depth(image_path, feed_width, feed_height, encoder, depth_decoder, device)
        pc_np = convert_depth_to_pointcloud(depth_np, K_np)
        print(sample, pc_np.shape)

        np.save(os.path.join(output_folder, sample+'_pc.npy'), pc_np)

        # sample_idx = np.random.permutation(pc_np.shape[1])
        # pc_np_sampled = pc_np[:, sample_idx[0:5000]]
        # plot_pc(pc_np_sampled)
        # plt.show()
        # break


if __name__ == '__main__':
    main()