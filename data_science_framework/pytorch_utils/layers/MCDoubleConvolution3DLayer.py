"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-10

**Project** : src

**Class that implements MCDoubleConvolution3DLayer structure**
"""
import torch.nn as nn
import torch


class MCDoubleConvolution3DLayer(nn.Module):
    """
    Class that implements MCDoubleConvolution3DLayer structure

    :param in_channels: Number of channel of the input
    :param out_channels: Number of channel of the output
    :param kernel_size: Size of the convolution kernel
    :param padding: Size of the padding
    :param dropout: Probability of dropout
    """
    def __init__(
            self, in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding: int,
            dropout: float
    ):
        super(MCDoubleConvolution3DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dropout = dropout
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.Dropout3d(p=self.dropout, inplace=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.Dropout3d(self.dropout, inplace=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method that computes forward pass

        :param x: Tensor value before forward pass
        :return: Tensor value after forward pass
        """
        return self.double_conv(x)
