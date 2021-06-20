import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.mmcv.conv_module import ConvModule
from models.layers_pc import PointNet
from models import resnet
from kitti.options import Options


class ImageEncoder(nn.Module):
    def __init__(self, opt):
        super(ImageEncoder, self).__init__()
        self.opt = opt

        self.backbone = resnet.resnet34(in_channels=3, pretrained=True, progress=True)

        # image mesh grid
        input_mesh_np = np.meshgrid(np.linspace(start=0, stop=self.opt.img_W - 1, num=self.opt.img_W),
                                    np.linspace(start=0, stop=self.opt.img_H - 1, num=self.opt.img_H))
        input_mesh = torch.from_numpy(np.stack(input_mesh_np, axis=0).astype(np.float32)).to(self.opt.device)  # 2xHxW
        self.input_mesh = input_mesh.unsqueeze(0).expand(self.opt.batch_size, 2, self.opt.img_H,
                                                         self.opt.img_W)  # Bx2xHxW

    def forward(self, x):
        resnet_out = self.backbone(x)
        return resnet_out[3], resnet_out[4], resnet_out[5]
