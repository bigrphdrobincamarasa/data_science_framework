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


def test_DoubleConvolution3DLayer() -> None:
    """
    Function that tests DoubleConvolution3DLayer

    :return: None
    """
    # Initialize double convolution layer
    double_convolution_3DLayer = DoubleConvolution3DLayer(
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
    assert double_convolution_3DLayer(input_torch).shape == (1, 3, 3, 4, 5)
