"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-11

**Project** : src

**Class that implements Unet2Axis structure**

"""
import torch.nn as nn
import torch
from data_science_framework.pytorch_utils.models import Unet
from data_science_framework.pytorch_utils.layers import DownConvolution2Axis3DLayer,\
        UpConvolution2Axis3DLayer

class Unet2Axis(Unet):
    """
    Class that implements Unet2Axis structure

    :param name: Name of the model
    :param in_channels: Number of channel of the input
    :param out_channels: Number of channel of the output
    :param depth: Depth of the network
    :param n_features: Number of features of the first layer
    :param kernel_size: Size of the convolution kernel
    :param pool_size: Scaling factor in Down Convolution layer
    :param padding: Padding of the convolution
    :param activation: Type of activation function (either 'sigmoid' or 'softmax')
    """
    def __init__(
            self, name='unet_2_axis', in_channels: int=1, out_channels: int=1,
            depth: int=3, n_features: int=8, kernel_size: int=3,
            pool_size: int=2, padding: int=1, activation: str='softmax',
    ):
        super(Unet2Axis, self).__init__(
            name=name, in_channels=in_channels, out_channels=out_channels,
            depth=depth, n_features=n_features, kernel_size=kernel_size,
            pool_size=pool_size, padding=padding, activation=activation,
            down_conv=DownConvolution2Axis3DLayer,
            up_conv=UpConvolution2Axis3DLayer,
        )

