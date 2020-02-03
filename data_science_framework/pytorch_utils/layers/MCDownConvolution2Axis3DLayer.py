"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-10

**Project** : src

**Class that implements MCDownConvolution2Axis3DLayer structure**
"""
import torch.nn as nn
import torch
from data_science_framework.pytorch_utils.layers import MCDownConvolution3DLayer


class MCDownConvolution2Axis3DLayer(MCDownConvolution3DLayer):
    """
    Class that implements MCDownConvolution2Axis3DLayer structure

    :param in_channels: Number of channel of the input
    :param out_channels: Number of channel of the output
    :param pool_size: Size of the max pooling windows
    :param kernel_size: Size of the kernel
    :param padding: Padding of the convolution
    :param dropout: Proportion of the dropout values
    """
    def __init__(
            self, in_channels: int,
            out_channels: int,
            pool_size: int,
            kernel_size: int,
            padding: int,
            dropout: float
    ):
        super(MCDownConvolution2Axis3DLayer, self).__init__(
            in_channels=in_channels, out_channels=out_channels,
            pool_size=(pool_size, pool_size, 1),
            kernel_size=kernel_size, padding=padding,
            dropout=dropout
        )

