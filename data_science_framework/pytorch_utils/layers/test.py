"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-10

**Project** : src

**File that tests codes of layers module**

"""
import torch
import numpy as np
from data_science_framework.pytorch_utils.layers.DoubleConvolution3DLayer import \
    DoubleConvolution3DLayer

from data_science_framework.pytorch_utils.layers.DownConvolution3DLayer import \
    DownConvolution3DLayer


def test_DoubleConvolution3DLayer() -> None:
    """
    Function that tests DoubleConvolution3DLayer

    :return: None
    """
    # Initialize double convolution layer
    double_convolution_3_d_layer = DoubleConvolution3DLayer(
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        padding=1
    )

    # Initialize input
    input = np.arange(2 * 3 * 4 * 5).reshape(1, 2, 3, 4, 5)
    input_torch = torch.tensor(
        input, dtype=torch.float32,
    ).to('cpu')

    # Apply forward pass
    assert double_convolution_3_d_layer(input_torch).shape == (1, 3, 3, 4, 5)


def test_DownConvolution3DLayer() -> None:
    """
    Function that tests DownConvolution3DLayer

    :return: None
    """
    # Initialize double convolution layer
    down_convolution_3_d_layer = DownConvolution3DLayer(
        in_channels=2,
        out_channels=3,
        pool_size=2,
        kernel_size=3,
        padding=1
    )

    # Initialize input
    input = np.arange(2 * 3 * 4 * 5).reshape(1, 2, 3, 4, 5)
    input_torch = torch.tensor(
        input, dtype=torch.float32,
    ).to('cpu')

    # Apply forward pass
    assert down_convolution_3_d_layer(input_torch).shape == (1, 3, 1, 2, 2)
