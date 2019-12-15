"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-11

**Project** : src

**Class that implements OutConvolution3DLayer structure**
"""
import torch.nn as nn
import torch


class OutConvolution3DLayer(nn.Module):
    """
    Class that implements OutConvolution3DLayer structure

    :param in_channels: Number of channel of the input
    :param out_channels: Number of channel of the output
    :param kernel_size: Size of the kernel
    :param padding: Padding of the convolution
    """

    def __init__(
            self, in_channels: int, out_channels: int,
            kernel_size: int, padding: int
    ) -> None:
        super(OutConvolution3DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=self.padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that computes forward pass

        :param x: Tensor value before forward pass
        :return: Tensor value after forward pass
        """
        return self.conv(x)