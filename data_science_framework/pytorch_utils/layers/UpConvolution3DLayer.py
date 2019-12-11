"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-10

**Project** : src

Class that implements UpConvolution3DLayer structure
"""
import torch.nn as nn
import torch
import torch.functional as F
from data_science_framework.pytorch_utils.layers.DoubleConvolution3DLayer import DoubleConvolution3DLayer


class UpConvolution3DLayer(nn.Module):
    """
    Class that implements UpConvolution3DLayer structure. This upconvimplementation only
    works with 2**n image dimension size

    :param n_channels: Number of channel of the input
    :param n_classes: Number of channel of the output
    :param kernel_size: Size of the kernel
    :param padding: Padding of the convolution
    """
    def __init__(
            self, in_channels: int, out_channels: int,
            kernel_size: int, padding: int, pool_size: int
    ):
        super(UpConvolution3DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.pool_size = pool_size

        self.up = nn.Upsample(
            scale_factor=self.pool_size
        )
        self.conv = DoubleConvolution3DLayer(
            in_channels=3 * self.in_channels, out_channels=self.out_channels,
            kernel_size=self.kernel_size, padding=self.padding
        )

    def forward(self, x_down: torch.Tensor, x_left: torch.Tensor) -> torch.Tensor:
        """
        Method that computes forward pass

        :param x_down: Tensor value copy from downscale path
        :param x_left: Tensor value copy from upscale path
        :return: Tensor value after forward pass
        """
        x_down = self.up(x_down)

        x = torch.cat([x_left, x_down], dim=1)

        return self.conv(x)
