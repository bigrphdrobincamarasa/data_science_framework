"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-10

**Project** : src

**Class that implements DownConvolution3DLayer structure**
"""
import torch.nn as nn
import torch
from data_science_framework.pytorch_utils.layers.DoubleConvolution3DLayer import DoubleConvolution3DLayer


class DownConvolution3DLayer(nn.Module):
    """
    Class that implements DownConvolution3DLayer structure

    :param in_channels: Number of channel of the input
    :param out_channels: Number of channel of the output
    :param pool_size: Size of the max pooling windows
    :param kernel_size: Size of the kernel
    :param padding: Padding of the convolution
    """
    def __init__(
            self, in_channels: int,
            out_channels: int,
            pool_size: int,
            kernel_size: int,
            padding: int
    ):
        super(DownConvolution3DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(pool_size),
            DoubleConvolution3DLayer(in_channels, out_channels, kernel_size, padding)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that computes forward pass

        :param x: Tensor value before forward pass
        :return: Tensor value after forward pass
        """
        return self.maxpool_conv(x)

