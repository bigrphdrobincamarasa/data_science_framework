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
from data_science_framework.pytorch_utils.layers import DoubleConvolution3DLayer,\
        OutConvolution3DLayer, DownConvolution3DLayer, UpConvolution3DLayer, \
        UpConvolution2Axis3DLayer, DownConvolution2Axis3DLayer

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
    input = np.arange(2 * 4 * 4 * 4).reshape(1, 2, 4, 4, 4)
    input_torch = torch.tensor(
        input, dtype=torch.float32,
    ).to('cpu')

    # Apply forward pass
    assert double_convolution_3_d_layer(input_torch).shape == (1, 3, 4, 4, 4)


def test_DownConvolution3DLayer() -> None:
    """
    Function that tests DownConvolution3DLayer

    :return: None
    """
    # Initialize layer
    down_convolution_3_d_layer = DownConvolution3DLayer(
        in_channels=2,
        out_channels=3,
        pool_size=2,
        kernel_size=3,
        padding=1
    )

    # Initialize input
    input = np.arange(2 * 4 * 4 * 4).reshape(1, 2, 4, 4, 4)
    input_torch = torch.tensor(
        input, dtype=torch.float32,
    ).to('cpu')

    # Apply forward pass
    assert down_convolution_3_d_layer(input_torch).shape == (1, 3, 2, 2, 2)

def test_DownConvolution2Axis3DLayer() -> None:
    """
    Function that tests DownConvolution3DLayer

    :return: None
    """
    # Initialize layer
    down_convolution_3_d_layer = DownConvolution2Axis3DLayer(
        in_channels=2,
        out_channels=3,
        pool_size=2,
        kernel_size=3,
        padding=1
    )

    # Initialize input
    input = np.arange(2 * 4 * 4 * 7).reshape(1, 2, 4, 4, 7)
    input_torch = torch.tensor(
        input, dtype=torch.float32,
    ).to('cpu')

    # Apply forward pass
    assert down_convolution_3_d_layer(input_torch).shape == (1, 3, 2, 2, 7)


def test_UpConvolution3DLayer() -> None:
    """
    Function that tests UpConvolution3DLayer

    :return: None
    """
    # Initialize layer
    up_convonvolution_3_d_layer = UpConvolution3DLayer(
        in_channels=3,
        out_channels=5,
        pool_size=2,
        kernel_size=3,
        padding=1
    )

    # Initialize input
    input_left = np.arange(3 * 8 * 8 * 8).reshape(1, 3, 8, 8, 8)
    input_down = np.arange(6 * 4 * 4 * 4).reshape(1, 6, 4, 4, 4)

    input_left_torch = torch.tensor(
        input_left, dtype=torch.float32,
    ).to('cpu')
    input_down_torch = torch.tensor(
        input_down, dtype=torch.float32,
    ).to('cpu')

    # Apply forward pass
    assert up_convonvolution_3_d_layer(
        x_down=input_down_torch,
        x_left=input_left_torch
    ).shape == (1, 5, 8, 8, 8)


def test_UpConvolution2Axis3DLayer() -> None:
    """
    Function that tests UpConvolution2Axis3DLayer

    :return: None
    """
    # Initialize layer
    up_convonvolution_3_d_layer = UpConvolution2Axis3DLayer(
        in_channels=3,
        out_channels=5,
        pool_size=2,
        kernel_size=3,
        padding=1
    )

    # Initialize input
    input_left = np.arange(3 * 8 * 8 * 5).reshape(1, 3, 8, 8, 5)
    input_down = np.arange(6 * 4 * 4 * 5).reshape(1, 6, 4, 4, 5)

    input_left_torch = torch.tensor(
        input_left, dtype=torch.float32,
    ).to('cpu')
    input_down_torch = torch.tensor(
        input_down, dtype=torch.float32,
    ).to('cpu')

    # Apply forward pass
    assert up_convonvolution_3_d_layer(
        x_down=input_down_torch,
        x_left=input_left_torch
    ).shape == (1, 5, 8, 8, 5)


def test_OutConvolution3DLayer() -> None:
    """
    Function that tests OutConvolution3DLayer

    :return: None
    """
    for activation in ['sigmoid', 'softmax']:
        # Initialize layer
        out_convolution_3_d_layer = OutConvolution3DLayer(
            in_channels=3,
            out_channels=5,
            kernel_size=3,
            padding=1,
            activation='softmax'
        )

        # Initialize input
        input = np.arange(3 * 4 * 4 * 4).reshape(1, 3, 4, 4, 4)

        input_torch = torch.tensor(
            input, dtype=torch.float32,
        ).to('cpu')

        # Apply forward pass
        assert out_convolution_3_d_layer(
            input_torch
        ).shape == (1, 5, 4, 4, 4)
