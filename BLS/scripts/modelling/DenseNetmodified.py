# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from monai.networks.layers.factories import Conv, Dropout, Pool
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils.module import look_up_option

__all__ = [
    "DenseNet"
]


class _DenseLayer(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        dropout_prob: float,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            growth_rate: how many filters to add each layer (k in paper).
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        out_channels = bn_size * growth_rate
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, spatial_dims]

        self.layers = nn.Sequential()

        self.layers.add_module("norm1", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels))
        self.layers.add_module("relu1", get_act_layer(name=act))
        self.layers.add_module("conv1", conv_type(in_channels, out_channels, kernel_size=1, bias=False))

        self.layers.add_module("norm2", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels))
        self.layers.add_module("relu2", get_act_layer(name=act))
        self.layers.add_module("conv2", conv_type(out_channels, growth_rate, kernel_size=3, padding=1, bias=False))

        if dropout_prob > 0:
            self.layers.add_module("dropout", dropout_type(dropout_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layers(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(
        self,
        spatial_dims: int,
        layers: int,
        in_channels: int,
        bn_size: int,
        growth_rate: int,
        dropout_prob: float,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            layers: number of layers in the block.
            in_channels: number of the input channel.
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            growth_rate: how many filters to add each layer (k in paper).
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()
        for i in range(layers):
            layer = _DenseLayer(spatial_dims, in_channels, growth_rate, bn_size, dropout_prob, act=act, norm=norm)
            in_channels += growth_rate
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            out_channels: number of the output classes.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        pool_type: Callable = Pool[Pool.AVG, spatial_dims]

        self.add_module("norm", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels))
        self.add_module("relu", get_act_layer(name=act))
        self.add_module("conv", conv_type(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module("pool", pool_type(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        bn_size: int = 4,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        pool_type: type[nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d] = Pool[Pool.MAX, spatial_dims]
        upsample_type: Callable = nn.Upsample
        deconv_type: Callable = nn.ConvTranspose2d

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", conv_type(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=init_features)),
                    ("relu0", get_act_layer(name=act)),
                    ("pool0", pool_type(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        in_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=in_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features.add_module(
                    "norm5", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
                )
            else:
                _out_channels = in_channels // 2
                trans = _Transition(
                    spatial_dims, in_channels=in_channels, out_channels=_out_channels, act=act, norm=norm
                )
                self.features.add_module(f"transition{i + 1}", trans)
                in_channels = _out_channels

        # Decoder part for segmentation with additional upsample step
        self.decoder_layers = nn.Sequential(
            OrderedDict(
                [
                    # ("upsample1", upsample_type(scale_factor=2, mode="trilinear" if spatial_dims == 3 else "bilinear", align_corners=True)),
                    # ("conv1", conv_type(in_channels, in_channels // 2, kernel_size=3, padding=1)),
                    # ("norm1", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels // 2)),
                    # ("relu1", get_act_layer(name=act)),

                    # ("upsample2", upsample_type(scale_factor=2, mode="trilinear" if spatial_dims == 3 else "bilinear", align_corners=True)),
                    # ("conv2", conv_type(in_channels // 2, in_channels // 4, kernel_size=3, padding=1)),
                    # ("norm2", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels // 4)),
                    # ("relu2", get_act_layer(name=act)),

                    # ("upsample3", upsample_type(scale_factor=2, mode="trilinear" if spatial_dims == 3 else "bilinear", align_corners=True)),
                    # ("conv3", conv_type(in_channels // 4, in_channels//8, kernel_size=1)),
                    # ("norm3", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels // 8)),
                    # ("relu3", get_act_layer(name=act)),

                    # ("upsample4", upsample_type(scale_factor=2, mode="trilinear" if spatial_dims == 3 else "bilinear", align_corners=True)),
                    # ("conv4", conv_type(in_channels // 8, in_channels // 16, kernel_size=3, padding=1)),
                    # ("norm4", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels // 16)),
                    # ("relu4", get_act_layer(name=act)),

                    # ("upsample5", upsample_type(scale_factor=2, mode="trilinear" if spatial_dims == 3 else "bilinear", align_corners=True)),
                    # ("conv5", conv_type(in_channels // 16, out_channels, kernel_size=3, padding=1)),
                    ("deconv1", deconv_type(in_channels, in_channels // 2, kernel_size=2, stride=2)),
                    ("conv1", conv_type(in_channels // 2, in_channels // 2, kernel_size=3, padding=1)),
                    ("norm1", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels // 2)),
                    ("relu1", get_act_layer(name=act)),

                    ("deconv2", deconv_type(in_channels // 2, in_channels // 4, kernel_size=2, stride=2)),
                    ("conv2", conv_type(in_channels // 4, in_channels // 4, kernel_size=3, padding=1)),
                    ("norm2", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels // 4)),
                    ("relu2", get_act_layer(name=act)),

                    ("deconv3", deconv_type(in_channels // 4, in_channels // 8, kernel_size=2, stride=2)),
                    ("conv3", conv_type(in_channels // 8, in_channels // 8, kernel_size=1)),
                    ("norm3", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels // 8)),
                    ("relu3", get_act_layer(name=act)),

                    ("deconv4", deconv_type(in_channels // 8, in_channels // 16, kernel_size=2, stride=2)),
                    ("conv4", conv_type(in_channels // 16, in_channels // 16, kernel_size=3, padding=1)),
                    ("norm4", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels // 16)),
                    ("relu4", get_act_layer(name=act)),

                    ("deconv5", deconv_type(in_channels // 16, out_channels, kernel_size=2, stride=2)),
                    ("conv5", conv_type(out_channels, out_channels, kernel_size=3, padding=1))

                ]
            )
        )

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.decoder_layers(x)
        return x