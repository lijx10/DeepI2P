import torch
import torch.nn as nn
import math
from typing import Tuple, List


from . import operations


class Swish(nn.Module):
    def __init__(self):
        """
        Swish activation function
        """
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Swish activation. Apply element-wise.
        :param x: torch.Tensor
        :return: torch.Tensor
        """
        return x * torch.sigmoid(x)


class MyLinear(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 normalization: str='batch',
                 norm_momentum: float = 0.1,
                 activation: str = 'relu',
                 dropout_rate: float=None):
        """
        Customized Linear module that integrates pytorch Linear, normalization and activation functions
        :param in_channels: C of input tensor
        :param out_channels: C of output tensor
        :param normalization: normalization method, 'batch', 'instance'
        :param norm_momentum: momentum in normalization layer
        :param activation: activation method, 'relu', 'elu', 'swish', 'leakyrelu', 'selu'
        :param dropout_rate: drop percentage
        """
        super(MyLinear, self).__init__()
        self.activation = activation
        self.normalization = normalization

        if dropout_rate is not None and dropout_rate > 0 and dropout_rate < 1:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        if self.normalization == 'batch':
            self.norm = nn.BatchNorm1d(out_channels, momentum=norm_momentum, affine=True)
        elif self.normalization == 'instance':
            self.norm = nn.InstanceNorm1d(out_channels, momentum=norm_momentum, affine=True)
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif 'elu' == activation:
            self.act = nn.ELU(alpha=1.0)
        elif 'swish' == self.activation:
            self.act = Swish()
        elif 'leakyrelu' == self.activation:
            self.act = nn.LeakyReLU(0.01)
        elif 'selu' == self.activation:
            self.act = nn.SELU()

        self.weight_init()

    def weight_init(self):
        """
        Weight initialization
        :return: None
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.in_features
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm2d) \
                        or isinstance(m, nn.BatchNorm1d) \
                        or isinstance(m, nn.BatchNorm3d) \
                        or isinstance(m, nn.InstanceNorm2d) \
                        or isinstance(m, nn.InstanceNorm1d) \
                        or isinstance(m, nn.InstanceNorm3d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Linear -> normalization -> activation -> dropout
        :param x: <torch.FloatTensor, BxC> Input pytorch tensor
        :return: torch.Tensor, BxC
        """
        x = self.linear(x)

        if self.normalization is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.act(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class MyConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int or Tuple,
                 stride: int=1,
                 padding: int=0,
                 bias: bool=True,
                 normalization: str = 'batch' or None,
                 norm_momentum: float = 0.1,
                 activation: str = 'relu' or None):
        """
        Customized nn.Conv2d module that integrates pytorch Conv2d, normalization and activation functions
        :param in_channels: C of input tensor
        :param out_channels: C of output tensor
        :param kernel_size: kernel size of 2d convolution, int or Tuple[int, int]
        :param stride: stride of 2d convolution, int or Tuple[int, int]
        :param padding: padding, int or Tuple[int, int]
        :param bias: whether to perform bias
        :param normalization: normalization method, 'batch', 'instance'
        :param norm_momentum: momentum in normazliation layer
        :param activation: activation method, 'relu', 'elu', 'swish', 'leakyrelu', 'selu'
        """
        super(MyConv2d, self).__init__()
        self.activation = activation
        self.normalization = normalization

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if self.normalization == 'batch':
            self.norm = nn.BatchNorm2d(out_channels, momentum=norm_momentum, affine=True)
        elif self.normalization == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels, momentum=norm_momentum, affine=True)

        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'elu':
            self.act = nn.ELU(alpha=1.0)
        elif 'swish' == self.activation:
            self.act = Swish()
        elif 'leakyrelu' == self.activation:
            self.act = nn.LeakyReLU(0.01)
        elif 'selu' == self.activation:
            self.act = nn.SELU()

        self.weight_init()

    def weight_init(self):
        """
        Weight initialization
        :return: None
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d) \
                    or isinstance(m, nn.BatchNorm1d) \
                    or isinstance(m, nn.BatchNorm3d) \
                    or isinstance(m, nn.InstanceNorm2d) \
                    or isinstance(m, nn.InstanceNorm1d) \
                    or isinstance(m, nn.InstanceNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Conv2d -> normalization -> activation
        :param x: <torch.FloatTensor, BxCxHxW>
        :return: <torch.FloatTensor, BxCxHxW>
        """
        x = self.conv(x)

        if self.normalization is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.act(x)
        return x


class UpConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 scale_factor: float=2.0,
                 mode: str='bilinear',
                 kernel_size: int=3,
                 stride: int=1,
                 padding: int=1,
                 normalization: str=None,
                 activation: str=None):
        """
        This is a upsampling module. Instead of transposed convolution, we use Upsampling + Conv2d.
        Note that the kernel_size, stride, padding should be tuned to acquire correct output size
        :param in_channels: C of input tensor
        :param out_channels: C of output tensor
        :param scale_factor: upsampling scale factor
        :param mode:  the upsampling algorithm: one of 'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'.
        :param kernel_size: kernel size of conv2d
        :param stride: stride of conv2d
        :param padding: padding of conv2d
        :param normalization: normalization method, 'batch', 'instance'
        :param activation: activation method, 'relu', 'elu', 'swish', 'leakyrelu', 'selu'
        """
        super(UpConv, self).__init__()
        self.activation = activation
        self.normalization = normalization

        self.up_sample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.conv = MyConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True,
                             normalization=normalization, activation=activation)

        self.weight_init()

    def weight_init(self):
        """
        Weight initialization
        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0.001)
            elif isinstance(m, nn.BatchNorm2d) \
                    or isinstance(m, nn.BatchNorm1d) \
                    or isinstance(m, nn.BatchNorm3d) \
                    or isinstance(m, nn.InstanceNorm2d) \
                    or isinstance(m, nn.InstanceNorm1d) \
                    or isinstance(m, nn.InstanceNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """
        nn.Upsample -> MyConv2d
        :param x:
        :return:
        """
        x = self.up_sample(x)
        x = self.conv(x)

        return x


class EquivariantLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 normalization: str = 'batch',
                 norm_momentum: float=0.1,
                 activation: str = 'relu',
                 dropout_rate: float = None):
        """
        This is the building block of PointNet, i.e., kernel size 1 Conv1d
        :param in_channels: C of input tensor
        :param out_channels: C of output tensor
        :param normalization: normalization method, 'batch', 'instance'
        :param norm_momentum: momentum in normazliation layer
        :param activation: activation method, 'relu', 'elu', 'swish', 'leakyrelu', 'selu'
        """
        super(EquivariantLayer, self).__init__()
        self.activation = activation
        self.normalization = normalization

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        if 'batch' == self.normalization:
            self.norm = nn.BatchNorm1d(out_channels, momentum=norm_momentum, affine=True)
        elif 'instance' == self.normalization:
            self.norm = nn.InstanceNorm1d(out_channels, momentum=norm_momentum, affine=True)

        if 'relu' == self.activation:
            self.act = nn.ReLU()
        elif 'elu' == self.activation:
            self.act = nn.ELU(alpha=1.0)
        elif 'swish' == self.activation:
            self.act = Swish()
        elif 'leakyrelu' == self.activation:
            self.act = nn.LeakyReLU(0.01)
        elif 'selu' == self.activation:
            self.act = nn.SELU()

        if dropout_rate is not None and dropout_rate > 0 and dropout_rate < 1:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

        self.weight_init()

    def weight_init(self):
        """
        Weight initialization
        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm2d) \
                        or isinstance(m, nn.BatchNorm1d) \
                        or isinstance(m, nn.BatchNorm3d) \
                        or isinstance(m, nn.InstanceNorm2d) \
                        or isinstance(m, nn.InstanceNorm1d) \
                        or isinstance(m, nn.InstanceNorm3d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This is the building block of PointNet, i.e., kernel size 1 Conv1d, followed by normalization and activation
        :param x: <torch.FloatTensor, BxCxL>
        :return: <torch.FloatTensor, BxCxL>
        """

        x = self.conv(x)

        if self.normalization is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.act(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class PointNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels_list: List[int],
                 normalization: str='batch',
                 norm_momentum: float=0.1,
                 activation: str='relu',
                 output_init_radius: float=None,
                 norm_act_at_last: bool=False,
                 dropout_list: List[float]=None):
        """
        PointNet, i.e., a series of EquivariantLayer
        :param in_channels: C in input tensors
        :param out_channels_list: A list of intermediate and final output channels
        :param normalization: normalization method, 'batch', 'instance'
        :param norm_momentum: momentum in normazliation layer
        :param activation: activation method, 'relu', 'elu', 'swish', 'leakyrelu', 'selu'
        :param output_init_radius: The output tensor value range at initialization
        """
        super(PointNet, self).__init__()

        if dropout_list is None:
            dropout_list = [-1] * len(out_channels_list)

        self.layers = nn.ModuleList()
        previous_out_channels = in_channels
        for i, c_out in enumerate(out_channels_list):
            if(i == len(out_channels_list)-1):
                if False == norm_act_at_last:
                    self.layers.append(EquivariantLayer(previous_out_channels,
                                                        c_out,
                                                        normalization=None,
                                                        norm_momentum=None,
                                                        activation=None,
                                                        dropout_rate=dropout_list[i]))
                else:
                    self.layers.append(EquivariantLayer(previous_out_channels,
                                                        c_out,
                                                        normalization=normalization,
                                                        norm_momentum=norm_momentum,
                                                        activation=activation,
                                                        dropout_rate=dropout_list[i]))
            else:
                self.layers.append(EquivariantLayer(previous_out_channels,
                                                    c_out,
                                                    normalization=normalization,
                                                    norm_momentum=norm_momentum,
                                                    activation=activation,
                                                    dropout_rate=dropout_list[i]))
            previous_out_channels = c_out

        # initialize the last layer to satisfy output_init_radius
        if output_init_radius is not None:
            self.layers[len(out_channels_list)-1].conv.bias.data.uniform_(-1*output_init_radius, output_init_radius)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        PointNet
        :param x: <torch.FloatTensor, BxCxN>
        :return: <torch.FloatTensor, BxCxN>
        """
        for layer in self.layers:
            x = layer(x)
        return x


class PointNetConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels_list: List[int],
                 normalization: str='batch',
                 norm_momentum: float=0.1,
                 activation: str='relu',
                 output_init_radius: float=None):
        """
        PointNet, i.e., a series of EquivariantLayer
        :param in_channels: C in input tensors
        :param out_channels_list: A list of intermediate and final output channels
        :param normalization: normalization method, 'batch', 'instance'
        :param norm_momentum: momentum in normazliation layer
        :param activation: activation method, 'relu', 'elu', 'swish', 'leakyrelu', 'selu'
        :param output_init_radius: The output tensor value range at initialization
        """
        super(PointNetConv2d, self).__init__()

        self.layers = nn.ModuleList()
        previous_out_channels = in_channels
        for i, c_out in enumerate(out_channels_list):
            self.layers.append(MyConv2d(previous_out_channels,
                                        c_out,
                                        kernel_size=(1, 1),
                                        stride=1,
                                        padding=0,
                                        bias=True,
                                        normalization=normalization,
                                        norm_momentum=norm_momentum,
                                        activation=activation))
            previous_out_channels = c_out

        # initialize the last layer to satisfy output_init_radius
        if output_init_radius is not None:
            self.layers[len(out_channels_list)-1].conv.bias.data.uniform_(-1*output_init_radius, output_init_radius)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        PointNet
        :param x: <torch.FloatTensor, BxCxMxN>
        :return: <torch.FloatTensor, BxCxMxN>
        """
        for layer in self.layers:
            x = layer(x)
        return x


class PointResNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels_list: List[int],
                 normalization: str='batch',
                 norm_momentum: float=0.1,
                 activation: str='relu'):
        """
        PointNet with skip connection
        in -> out[0]
        out[0] -> out[1]             ----
        out[1] -> out[2]                |
             ... ...                    |
        out[k-2]+out[1] -> out[k-1]  <---
        :param in_channels: C of input tensor
        :param out_channels_list: List of channels of PointNet
        :param normalization: normalization method, 'batch', 'instance'
        :param norm_momentum: momentum in normazliation layer
        :param activation: activation method, 'relu', 'elu', 'swish', 'leakyrelu', 'selu'
        """
        super(PointResNet, self).__init__()
        self.out_channels_list = out_channels_list

        self.layers = nn.ModuleList()
        previous_out_channels = in_channels
        for i, c_out in enumerate(out_channels_list):
            self.layers.append(EquivariantLayer(previous_out_channels,
                                                c_out,
                                                norm_momentum=norm_momentum,
                                                normalization=normalization,
                                                activation=activation))
            previous_out_channels = c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        PointNet with skip connection
        in -> out[0]
        out[0] -> out[1]             ----
        out[1] -> out[2]                |
             ... ...                    |
        out[k-2]+out[1] -> out[k-1]  <---
        :param x: <torch.FloatTensor, BxCxN>
        :return: <torch.FloatTensor, BxCxN>
        """
        layer0_out = self.layers[0](x)  # BxCxN
        for l in range(1, len(self.out_channels_list)-1):
            if l == 1:
                x_tmp = self.layers[l](layer0_out)
            else:
                x_tmp = self.layers[l](x_tmp)
        layer_final_out = self.layers[len(self.out_channels_list)-1](torch.cat((layer0_out, x_tmp), dim=1))
        return layer_final_out


class PointNetFusion(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels_list_before: List[int],
                 out_channels_list_after: List[int],
                 normalization: str='batch',
                 norm_momentum: float=0.1,
                 activation: str='relu',
                 act_norm_at_endof_pn1=True):
        """
        This is a modified PointNet. The maxpool output of the first PN is expanded and concatenate to
        the output (before maxpool) of the first PN. The concatenated features are forwarded into a second PN.
        :param in_channels: C of input tensor
        :param out_channels_list_before: List of channels in first PN
        :param out_channels_list_after: List of channels in second PN
        :param normalization: normalization method, 'batch', 'instance'
        :param norm_momentum: momentum in normazliation layer
        :param activation: activation method, 'relu', 'elu', 'swish', 'leakyrelu', 'selu'
        :param act_norm_at_endof_pn1: whether to apply activation and normalization at the last layer of first PointNet
        """
        super(PointNetFusion, self).__init__()

        self.layers_before = nn.ModuleList()
        previous_out_channels = in_channels
        for i, c_out in enumerate(out_channels_list_before):
            if act_norm_at_endof_pn1 or (i != len(out_channels_list_before)-1):
                self.layers_before.append(EquivariantLayer(previous_out_channels,
                                                           c_out,
                                                           normalization=normalization,
                                                           norm_momentum=norm_momentum,
                                                           activation=activation))
            else:
                self.layers_before.append(EquivariantLayer(previous_out_channels,
                                                           c_out,
                                                           normalization=None,
                                                           activation=None))
            previous_out_channels = c_out

        self.layers_after = nn.ModuleList()
        previous_out_channels = 2 * previous_out_channels
        for i, c_out in enumerate(out_channels_list_after):
            if i != len(out_channels_list_after)-1:
                self.layers_after.append(EquivariantLayer(previous_out_channels,
                                                          c_out,
                                                          normalization=normalization,
                                                          norm_momentum=norm_momentum,
                                                          activation=activation))
            else:
                self.layers_after.append(EquivariantLayer(previous_out_channels,
                                                          c_out,
                                                          normalization=None,
                                                          activation=None))
            previous_out_channels = c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Two PointNets
        :param x: <torch.FloatTensor BxCxN>
        :return: <torch.FloatTensor BxCxN>
        """
        for layer in self.layers_before:
            x = layer(x)

        # BxCxN -> BxCx1
        x_max, _ = torch.max(x, dim=2, keepdim=True)  # BxCx1
        x_max_expanded = x_max.expand(x.size())  # BxCxN

        # BxCxN -> Bx(C+C)xN
        y = torch.cat((x, x_max_expanded), dim=1)

        for layer in self.layers_after:
            y = layer(y)

        # BxCxN
        return y


class PointNetFusionConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels_list_before,
                 out_channels_list_after,
                 normalization='batch',
                 norm_momentum=0.1,
                 activation='relu',
                 act_norm_at_endof_pn1=True):
        """
        This is a modified PointNet. The maxpool output of the first PN is expanded and concatenated to
        the output (before maxpool) of the first PN. The concatenated features are forwarded into a second PN.
        The difference between this class and PointNetFusion is that:
            This class is implemented using Conv2d instead of EquivariantLayer,
            The input / output of this class is BxCxMxK / BxCxMx1,
            That is, M point clouds, each point cloud has K points.
        :param in_channels: C of input tensor
        :param out_channels_list_before: List of channels in first PN
        :param out_channels_list_after: List of channels in second PN
        :param normalization: normalization method, 'batch', 'instance'
        :param norm_momentum: momentum in normazliation layer
        :param activation: activation method, 'relu', 'elu', 'swish', 'leakyrelu', 'selu'
        :param act_norm_at_endof_pn1: whether to apply activation and normalization at the last layer of first PointNet
        """
        super(PointNetFusionConv2d, self).__init__()

        self.layers_before = nn.ModuleList()
        previous_out_channels = in_channels
        for i, c_out in enumerate(out_channels_list_before):
            if act_norm_at_endof_pn1 or (i != len(out_channels_list_before) - 1):
                self.layers_before.append(MyConv2d(previous_out_channels,
                                                   c_out,
                                                   kernel_size=(1, 1),
                                                   stride=1,
                                                   padding=0,
                                                   bias=True,
                                                   normalization=normalization,
                                                   norm_momentum=norm_momentum,
                                                   activation=activation))
            else:
                self.layers_before.append(MyConv2d(previous_out_channels,
                                                   c_out,
                                                   kernel_size=(1, 1),
                                                   stride=1,
                                                   padding=0,
                                                   bias=True,
                                                   normalization=None,
                                                   activation=None))
            previous_out_channels = c_out

        self.layers_after = nn.ModuleList()
        previous_out_channels = 2 * previous_out_channels
        for i, c_out in enumerate(out_channels_list_after):
            if i != len(out_channels_list_after)-1:
                self.layers_after.append(MyConv2d(previous_out_channels,
                                               c_out,
                                               kernel_size=(1, 1),
                                               stride=1,
                                               padding=0,
                                               bias=True,
                                               normalization=normalization,
                                               norm_momentum=norm_momentum,
                                               activation=activation))
            else:
                self.layers_after.append(MyConv2d(previous_out_channels,
                                                  c_out,
                                                  kernel_size=(1, 1),
                                                  stride=1,
                                                  padding=0,
                                                  bias=True,
                                                  normalization=None,
                                                  activation=None))
            previous_out_channels = c_out

    def forward(self, x) -> torch.Tensor:
        """
        PointNetFusion that works for M point clouds, each point cloud has K points.
        :param x: <torch.FloatTensor, BxCxMxK>
        :return: <torch.FloatTensor, BxCxMx1>
        """
        for layer in self.layers_before:
            x = layer(x)

        # BxCxMxK -> BxCxMx1
        x_max, _ = torch.max(x, dim=3, keepdim=True)
        x_max_expanded = x_max.expand(x.size())  # BxCxMxK

        # BxCxMxK -> Bx(C+C)xMxK
        y = torch.cat((x, x_max_expanded), dim=1)

        for layer in self.layers_after:
            y = layer(y)

        y_max, _ = torch.max(y, dim=3, keepdim=True)  # BxCxMx1
        return y_max


class KNNModule(nn.Module):
    def __init__(self, in_channels, out_channels_list, activation, normalization, norm_momentum=0.1):
        super(KNNModule, self).__init__()

        self.layers = nn.ModuleList()
        previous_out_channels = in_channels
        for c_out in out_channels_list:
            self.layers.append(MyConv2d(previous_out_channels, c_out, kernel_size=1, stride=1, padding=0, bias=True,
                                        activation=activation, normalization=normalization,
                                        norm_momentum=norm_momentum))
            previous_out_channels = c_out

    def forward(self, coordinate, x, precomputed_knn_I, K, center_type):
        '''

        :param coordinate: Bx3xM Variable
        :param x: BxCxM Variable
        :param precomputed_knn_I: BxMxK'
        :param K: K neighbors
        :param center_type: 'center' or 'avg'
        :return:
        '''
        # 0. compute knn
        # 1. for each node, calculate the center of its k neighborhood
        # 2. normalize nodes with the corresponding center
        # 3. fc for these normalized points
        # 4. maxpool for each neighborhood

        coordinate_tensor = coordinate.data  # Bx3xM
        if precomputed_knn_I is not None:
            assert precomputed_knn_I.size()[2] >= K
            knn_I = precomputed_knn_I[:, :, 0:K]
        else:
            coordinate_Mx1 = coordinate_tensor.unsqueeze(3)  # Bx3xMx1
            coordinate_1xM = coordinate_tensor.unsqueeze(2)  # Bx3x1xM
            norm = torch.sum((coordinate_Mx1 - coordinate_1xM) ** 2, dim=1)  # BxMxM, each row corresponds to each coordinate - other coordinates
            knn_D, knn_I = torch.topk(norm, k=K, dim=2, largest=False, sorted=True)  # BxMxK

        # debug
        # print(knn_D[0])
        # print(knn_I[0])
        # assert False

        neighbors = operations.knn_gather_wrapper(coordinate_tensor, knn_I)  # Bx3xMxK
        if center_type == 'avg':
            neighbors_center = torch.mean(neighbors, dim=3, keepdim=True)  # Bx3xMx1
        elif center_type == 'center':
            neighbors_center = coordinate_tensor.unsqueeze(3)  # Bx3xMx1
        else:
            neighbors_center = None
        neighbors_decentered = (neighbors - neighbors_center).detach()
        neighbors_center = neighbors_center.squeeze(3).detach()

        # debug
        # print(neighbors[0, 0])
        # print(neighbors_avg[0, 0])
        # print(neighbors_decentered[0, 0])
        # assert False

        x_neighbors = operations.knn_gather_by_indexing(x, knn_I)  # BxCxMxK
        x_augmented = torch.cat((neighbors_decentered, x_neighbors), dim=1)  # Bx(3+C)xMxK

        for layer in self.layers:
            x_augmented = layer(x_augmented)
        feature, _ = torch.max(x_augmented, dim=3, keepdim=False)

        return neighbors_center, feature


class GeneralKNNFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels_list_before, out_channels_list_after,
                 activation, normalization, norm_momentum=0.1):
        super(GeneralKNNFusionModule, self).__init__()

        self.layers_before = nn.ModuleList()
        previous_out_channels = in_channels
        for i, c_out in enumerate(out_channels_list_before):
            self.layers_before.append(
                MyConv2d(previous_out_channels, c_out, kernel_size=1, stride=1, padding=0, bias=True,
                         activation=activation, normalization=normalization,
                         norm_momentum=norm_momentum))
            previous_out_channels = c_out

        self.layers_after = nn.ModuleList()
        previous_out_channels = 2 * previous_out_channels
        for i, c_out in enumerate(out_channels_list_after):
            self.layers_after.append(
                MyConv2d(previous_out_channels, c_out, kernel_size=1, stride=1, padding=0, bias=True,
                         activation=activation, normalization=normalization,
                         norm_momentum=norm_momentum))
            previous_out_channels = c_out

    def forward(self, query, database, database_features, K):
        '''

        :param query: Bx3xM FloatTensor
        :param database: Bx3xN FloatTensor
        :param x: BxCxN FloatTensor
        :param K: K neighbors
        :return:
        '''
        # 1. compute knn, query -> database
        # 2. for each query, normalize neighbors with its coordinate
        # 3. FC for these normalized points
        # 4. maxpool for each query

        B, M, N, C = query.size()[0], query.size()[2], database.size()[2], database_features.size()[1]

        query_Mx1 = query.detach().unsqueeze(3)  # Bx3xMx1
        database_1xN = database.detach().unsqueeze(2)  # Bx3x1xN

        norm = torch.norm(query_Mx1 - database_1xN, dim=1, keepdim=False)  # Bx3xMxN -> BxMxN
        knn_D, knn_I = torch.topk(norm, k=K, dim=2, largest=False, sorted=True)  # BxMxK, BxMxK
        knn_I_3 = knn_I.unsqueeze(1).expand(B, 3, M, K).contiguous().view(B, 3, M*K)  # Bx3xMxK -> Bx3xM*K
        knn_I_C = knn_I.unsqueeze(1).expand(B, C, M, K).contiguous().view(B, C, M*K)  # BxCxMxK -> BxCxM*K

        query_neighbor_coord = torch.gather(database, dim=2, index=knn_I_3).view(B, 3, M, K)  # Bx3xMxK
        query_neighbor_feature = torch.gather(database_features, dim=2, index=knn_I_C).view(B, C, M, K)  # BxCxMxK

        query_neighbor_coord_decentered = (query_neighbor_coord - query_Mx1).detach()
        query_neighbor = torch.cat((query_neighbor_coord_decentered, query_neighbor_feature), dim=1)  # Bx(3+C)xMxK

        for layer in self.layers_before:
            query_neighbor = layer(query_neighbor)
        feature, _ = torch.max(query_neighbor, dim=3, keepdim=True)  # BxCxMx1

        y = torch.cat((feature.expand_as(query_neighbor), query_neighbor), dim=1)  # Bx2CxMxK
        for layer in self.layers_after:
            y = layer(y)
        feature, _ = torch.max(y, dim=3, keepdim=False)  # BxCxM

        return feature


class KNNFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels_list_before, out_channels_list_after,
                 activation, normalization, norm_momentum=0.1):
        super(KNNFusionModule, self).__init__()

        self.layers_before = nn.ModuleList()
        previous_out_channels = in_channels
        for i, c_out in enumerate(out_channels_list_before):
            self.layers_before.append(
                MyConv2d(previous_out_channels, c_out, kernel_size=1, stride=1, padding=0, bias=True,
                         activation=activation, normalization=normalization,
                         norm_momentum=norm_momentum))
            previous_out_channels = c_out

        self.layers_after = nn.ModuleList()
        previous_out_channels = 2 * previous_out_channels
        for i, c_out in enumerate(out_channels_list_after):
            self.layers_after.append(
                MyConv2d(previous_out_channels, c_out, kernel_size=1, stride=1, padding=0, bias=True,
                         activation=activation, normalization=normalization,
                         norm_momentum=norm_momentum))
            previous_out_channels = c_out

    def forward(self, coordinate, x, precomputed_knn_I, K, center_type):
        '''

        :param coordinate: Bx3xM Variable
        :param x: BxCxM Variable
        :param precomputed_knn_I: BxMxK'
        :param K: K neighbors
        :param center_type: 'center' or 'avg'
        :return:
        '''
        # 0. compute knn
        # 1. for each node, calculate the center of its k neighborhood
        # 2. normalize nodes with the corresponding center
        # 3. fc for these normalized points
        # 4. maxpool for each neighborhood

        coordinate_tensor = coordinate.data  # Bx3xM
        if precomputed_knn_I is not None:
            assert precomputed_knn_I.size()[2] >= K
            knn_I = precomputed_knn_I[:, :, 0:K]
        else:
            coordinate_Mx1 = coordinate_tensor.unsqueeze(3)  # Bx3xMx1
            coordinate_1xM = coordinate_tensor.unsqueeze(2)  # Bx3x1xM
            norm = torch.sum((coordinate_Mx1 - coordinate_1xM) ** 2, dim=1)  # BxMxM, each row corresponds to each coordinate - other coordinates
            knn_D, knn_I = torch.topk(norm, k=K, dim=2, largest=False, sorted=True)  # BxMxK

        neighbors = operations.knn_gather_wrapper(coordinate_tensor, knn_I)  # Bx3xMxK
        if center_type == 'avg':
            neighbors_center = torch.mean(neighbors, dim=3, keepdim=True)  # Bx3xMx1
        elif center_type == 'center':
            neighbors_center = coordinate_tensor.unsqueeze(3)  # Bx3xMx1
        neighbors_decentered = (neighbors - neighbors_center).detach()
        neighbors_center = neighbors_center.squeeze(3).detach()

        # debug
        # print(neighbors[0, 0])
        # print(neighbors_avg[0, 0])
        # print(neighbors_decentered[0, 0])
        # assert False

        x_neighbors = operations.knn_gather_by_indexing(x, knn_I)  # BxCxMxK
        x_augmented = torch.cat((neighbors_decentered, x_neighbors), dim=1)  # Bx(3+C)xMxK

        for layer in self.layers_before:
            x_augmented = layer(x_augmented)
        feature, _ = torch.max(x_augmented, dim=3, keepdim=True)  # BxCxMx1

        y = torch.cat((feature.expand_as(x_augmented), x_augmented), dim=1)  # Bx2CxMxK
        for layer in self.layers_after:
            y = layer(y)
        feature, _ = torch.max(y, dim=3, keepdim=False)  # BxCxM

        return neighbors_center, feature
