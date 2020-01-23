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


class CustomUpSample(nn.Module):
    """
    Class that compute a custom upsample

    :param scale_factor: Size of the pooling
    """
    def __init__(self, scale_factor: int):
        super(CustomUpSample, self).__init__()
        self.vanilla_upsample = nn.Upsample(
            scale_factor=scale_factor
        )

    def forward(self, x_down: torch.Tensor) -> torch.Tensor:
        """
        Method that computes forward pass

        :param x_down: Tensor value copy from downscale path
        :return: Tensor value after forward pass
        """
        up_sampled = [
            self.vanilla_upsample(
                x_down[:, :, :, :, i]
            ).unsqueeze(-1)
            for i in range(x_down.shape[-1])
        ]
        return torch.cat(up_sampled, dim=-1)


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
            pool_size=pool_size
        )

        self.up_sample = CustomUpSample
        self.create_architecture()

    def custom_upsample(self) -> callable:
        """custom_upsample

        :return: Torch upsampling layer
        :rtype: callable
        """
        def up_sample(x_down):
                    return up_sample
