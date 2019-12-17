"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-11

**Project** : src

**Class that implements Unet structure**

"""
import torch.nn as nn
import torch
from data_science_framework.pytorch_utils.layers.DoubleConvolution3DLayer import DoubleConvolution3DLayer
from data_science_framework.pytorch_utils.layers.DownConvolution3DLayer import DownConvolution3DLayer
from data_science_framework.pytorch_utils.layers.OutConvolution3DLayer import OutConvolution3DLayer
from data_science_framework.pytorch_utils.layers.UpConvolution3DLayer import UpConvolution3DLayer


class Unet(nn.Module):
    """
    Class that implements Unet structure

    :param in_channels: Number of channel of the input
    :param out_channels: Number of channel of the output
    :param depth: Depth of the network
    :param n_features: Number of features of the first layer
    :param kernel_size: Size of the convolution kernel
    :param pool_size: Scaling factor in Down Convolution layer
    :param padding: Padding of the convolution
    """
    def __init__(
            self, in_channels: int=1, out_channels: int=1,
            depth: int=3, n_features: int=8, kernel_size: int=3,
            pool_size: int=2, padding: int=1
    ):
        super(Unet, self).__init__()

        # Initialize attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.padding = padding
        self.depth = depth

        # Initialize the number of features in the layer
        layer_n_features_ = n_features

        # Initialize down path
        self.down_path_0 = DoubleConvolution3DLayer(
            in_channels=self.in_channels,
            out_channels=layer_n_features_,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )

        for i in range(1, self.depth + 1):
            self.__setattr__(
                'down_path_{}'.format(i),
                DownConvolution3DLayer(
                    in_channels=layer_n_features_,
                    out_channels=layer_n_features_ * self.pool_size,
                    pool_size=self.pool_size,
                    kernel_size=self.kernel_size,
                    padding=self.padding
                )
            )
            layer_n_features_ *= self.pool_size

        # Initialize up path
        for i in range(self.depth):
            layer_n_features_ = int(layer_n_features_ / self.pool_size)
            self.__setattr__(
                'up_path_{}'.format(i),
                UpConvolution3DLayer(
                    in_channels=layer_n_features_,
                    out_channels=layer_n_features_,
                    pool_size=self.pool_size,
                    kernel_size=self.kernel_size,
                    padding=self.padding
                )
            )
        self.__setattr__(
            'up_path_{}'.format(i+1),
            OutConvolution3DLayer(
                in_channels=self.n_features,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.padding
            )
        )

    def forward(self, x):
        """
        Method that computes forward pass

        :param x: Tensor value before forward pass
        :return: Tensor value after forward pass
        """
        # Initialize down path storage value list
        x = self.down_path_0(x)
        x_down_path = [x.clone()]

        # Loop to compute down path
        for i in range(1, self.depth + 1):
            x = self.__getattr__('down_path_{}'.format(i))(x)
            if i != self.depth:
                x_down_path.append(x.clone())

        # Loop to compute up path
        for i in range(self.depth):
            x = self.__getattr__('up_path_{}'.format(i))(
                x_down=x,
                x_left=x_down_path[-i-1]
            )
        x = self.__getattr__('up_path_{}'.format(self.depth))(x)

        return x
