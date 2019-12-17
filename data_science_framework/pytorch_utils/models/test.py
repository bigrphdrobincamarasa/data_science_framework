"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-11

**Project** : src

**File that tests codes of models module**
"""
import torch
import numpy as np
from data_science_framework.pytorch_utils.models.Unet import Unet


def test_Unet() -> None:
    """
    Function that tests Unet

    :return: None
    """
    # Initialize network
    unet = Unet(
        in_channels=5,
        out_channels=3,
        depth=3,
        n_features=2,
        kernel_size=3,
        pool_size=2,
        padding=1,
    )

    # Define input
    input = np.arange(5 * 16 * 16 * 16).reshape(1, 5, 16, 16, 16)
    input_torch = torch.tensor(
        input, dtype=torch.float32,
    ).to('cpu')

    # Define unet
    assert unet(input_torch).shape == (1, 3, 16, 16, 16)
