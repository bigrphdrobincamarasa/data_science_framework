"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-11

**Project** : src

**Class that implements ApproxMCUnet structure**

"""
import torch.nn as nn
import torch
from typing import Tuple
from data_science_framework.pytorch_utils.models import MCUnet
from data_science_framework.pytorch_utils.layers import MCDownConvolution3DLayer,\
        MCUpConvolution3DLayer

class ApproxMCUnet(MCUnet):
    """
    Class that implements ApproxMCUnet structure

    :param name: Name of the model
    :param in_channels: Number of channel of the input
    :param out_channels: Number of channel of the output
    :param depth: Depth of the network
    :param n_features: Number of features of the first layer
    :param kernel_size: Size of the convolution kernel
    :param pool_size: Scaling factor in Down Convolution layer
    :param padding: Padding of the convolution
    :param activation: Type of activation function (either 'sigmoid' or 'softmax')
    :param dropout: Value of dropout
    :param n_iter: Number of interation to compute mc dropout
    :param modality: Learnt modality (either 'mean', 'std', 'both')
    """
    def __init__(
            self, name='approx_mc_unet', in_channels: int=1, out_channels: int=1,
            depth: int=3, n_features: int=8, kernel_size: int=3,
            pool_size: int=2, padding: int=1, activation: str='softmax',
            dropout: float=0.1, n_iter: int=20, modality='mean',
            down_conv=MCDownConvolution3DLayer,
            up_conv=MCUpConvolution3DLayer
    ):
        self.modality = modality
        super(ApproxMCUnet, self).__init__(
            name=name, in_channels=in_channels, out_channels=2 * out_channels,
            depth=depth, n_features=n_features, kernel_size=kernel_size,
            pool_size=pool_size, padding=padding, activation=activation,
            n_iter=n_iter, dropout=dropout, down_conv=down_conv,
            up_conv=up_conv
        )

    def forward(self, x):
        """
        Method that computes forward pass

        :param x: Tensor value before forward pass
        :return: Tensor value after forward pass
        """
        if self.modality == 'both':
            return super().forward(x)
        if self.modality == 'mean':
            return super().forward(x)[:, :int(self.out_channels/2), :]
        if self.modality == 'std':
            return super().forward(x)[:, int(self.out_channels/2):, :]
        raise NotImplementedError
