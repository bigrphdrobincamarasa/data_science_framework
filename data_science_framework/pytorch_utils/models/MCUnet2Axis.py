"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-11

**Project** : src

**Class that implements MCUnet2Axis structure**

"""
import torch.nn as nn
import torch
from data_science_framework.pytorch_utils.models import Unet
from data_science_framework.pytorch_utils.layers import MCDownConvolution2Axis3DLayer,\
        MCUpConvolution2Axis3DLayer

class MCUnet2Axis(Unet):
    """
    Class that implements MCUnet2Axis structure

    :param name: Name of the model
    :param in_channels: Number of channel of the input
    :param out_channels: Number of channel of the output
    :param depth: Depth of the network
    :param n_features: Number of features of the first layer
    :param kernel_size: Size of the convolution kernel
    :param pool_size: Scaling factor in Down Convolution layer
    :param padding: Padding of the convolution
    :param activation: Type of activation function (either 'sigmoid' or 'softmax')
    :param dropout: Value of the dropout
    :param dropout: Value of dropout
    :param n_iter: Number of interation to compute mc dropout
    """
    def __init__(
            self, name='mc_unet_2_axis', in_channels: int=1, out_channels: int=1,
            depth: int=3, n_features: int=8, kernel_size: int=3,
            pool_size: int=2, padding: int=1, activation: str='softmax',
            dropout: float=0.1, n_iter: int=20
    ):
        self.n_iter = n_iter
        self.dropout = dropout
        super(MCUnet2Axis, self).__init__(
            name=name, in_channels=in_channels, out_channels=out_channels,
            depth=depth, n_features=n_features, kernel_size=kernel_size,
            pool_size=pool_size, padding=padding, activation=activation,
            down_conv=lambda *args, **kwargs: MCDownConvolution2Axis3DLayer(
                dropout=self.dropout, *args, **kwargs
            ),
            up_conv=lambda *args, **kwargs: MCUpConvolution2Axis3DLayer(
                dropout=self.dropout, *args, **kwargs
            ),
        )

    def mc_forward(self, x) -> torch.Tensor:
        """
        Method that computes multiple forward passes

        :param x: Tensor value before forward pass
        :return: Tensor value after n_iter forward pass
        """
        forward_passes = torch.cat(
            [
                self.forward(x=x)
                for _ in range(self.n_iter)
            ]
        )
        return forward_passes.reshape(
            *tuple(
                [self.n_iter, int(forward_passes.shape[0]/self.n_iter)] +\
                        list(forward_passes.shape)[1:]
            )
        )

