"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-10

**Project** : src

**Class that implements UpConvolution2Axis3DLayer structure**
"""
import torch.nn as nn
import torch
import torch.functional as F
from data_science_framework.pytorch_utils.layers import UpConvolution3DLayer


class UpConvolution2Axis3DLayer(UpConvolution3DLayer):
    """
    Class that implements UpConvolution2Axis3DLayer structure. This upconvimplementation only
    works with 2**n image dimension size

    :param in_channels: Number of channel of the input
    :param out_channels: Number of channel of the output
    :param kernel_size: Size of the kernel
    :param padding: Size of the padding
    :param padding: Padding of the convolution
    """
    def __init__(
            self, in_channels: int, out_channels: int,
            kernel_size: int, padding: int, pool_size: int
    ):
        super(UpConvolution2Axis3DLayer, self).__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, padding=padding,
            pool_size=(pool_size, pool_size, 1)
        )