'''
Module Name: decoders.py
Author: Alice Bizeul
Ownership: ETH Zürich - ETH AI Center
Modified from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
'''
from typing import Callable, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import torch.nn.functional as F

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, output_padding: int = 0, padding: int =0) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(
        int(in_planes),
        int(out_planes),
        kernel_size=3,
        stride=stride,
        padding=dilation,
        output_padding = output_padding,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1, output_padding: int = 0) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(int(in_planes), int(out_planes), kernel_size=1, stride=stride, bias=False, output_padding = output_padding)

class Bottleneck18(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 2

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        output_padding: int = 0,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv3 = conv3x3(inplanes, planes, stride=stride, groups=groups, dilation=dilation, output_padding=output_padding)
        self.bn3 = norm_layer(planes)

        self.conv1 = conv3x3(planes,planes,stride=1) 
        self.bn1 = norm_layer(planes) 
        
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv3(x)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        if self.upsample is not None:
            identity = self.upsample(x)
            out += identity
        out = self.relu(out)

        return out


class ResNetDec18(nn.Module):
    def __init__(
        self,
        layers: List[int],
        block =Bottleneck18,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        z_dim: int =64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        size: int =64,
        nc: int =3,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.z_dim = z_dim
        self.size = size
        self.inplanes = 2048
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.bn1 = norm_layer(3)
        self.relu = nn.ReLU(inplace=True)

        if   size==32:  self.final_kernel=4
        elif size==64:  self.final_kernel=16
        elif size==256: self.final_kernel=784

        self.linear = nn.Linear(z_dim, self.final_kernel*512 ) 
        self.layer4 = self._make_layer(block, 512 if self.size == 256 else 256, layers[3], stride=2)
        self.layer3 = self._make_layer(block, 256 if self.size == 256 else 128, layers[2], stride=2)
        self.layer2 = self._make_layer(block, 128 if self.size == 256 else 64,  layers[1], stride=2)
        self.layer1 = self._make_layer(block, 64  if self.size == 256 else 64,  layers[0], output_padding = 0, expansion=False)
        self.conv1 = nn.ConvTranspose2d( 64, nc, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), output_padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if (isinstance(m, Bottleneck18) or isinstance(m, Bottleneck50)) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0) 
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0) 

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        output_padding: int = 1,
        expansion: bool =True,

    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation

        layers = []
        if expansion: self.inplanes = planes * block.expansion
        else: self.inplanes = planes

        if stride != 1 :
            upsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride, output_padding),
                norm_layer(planes),
            )
        last_block = block(
                self.inplanes, planes, stride=stride, output_padding=output_padding, upsample=upsample, groups=self.groups, base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer
            )
        
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    self.inplanes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        layers.append(last_block)
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        nc = 2048 if self.size == 256 else 512 
        field = 2 if self.size == 32 else 4

        x = x.view(x.size(0), nc, field,field)
        if self.size==256: x = F.interpolate(x, size=(7,7))
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        if self.size==256:x = F.interpolate(x, scale_factor=2)
        x = self.conv1(x)

        return x

