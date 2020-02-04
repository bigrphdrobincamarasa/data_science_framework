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
from data_science_framework.pytorch_utils.models import Unet,\
        Unet2Axis, MCUnet, MCUnet2Axis, ApproxMCUnet


def test_Unet() -> None:
    """
    Function that tests Unet

    :return: None
    """
    # Initialize network
    unet = Unet(
        in_channels=5,
        out_channels=3,
        depth=4,
        n_features=2,
        kernel_size=3,
        pool_size=2,
        padding=1,
        activation='softmax'
    )

    # Define input
    input = np.arange(5 * 32 * 32 * 32).reshape(1, 5, 32, 32, 32)
    input_torch = torch.tensor(
        input, dtype=torch.float32,
    ).to('cpu')

    # Define unet
    assert unet(input_torch).shape == (1, 3, 32, 32, 32)


def test_Unet2Axis() -> None:
    """
    Function that tests Unet2Axis

    :return: None
    """
    # Initialize network
    unet = Unet2Axis(
        in_channels=5,
        out_channels=3,
        depth=4,
        n_features=2,
        kernel_size=3,
        pool_size=2,
        padding=1,
        activation='softmax'
    )

    # Define input
    input = np.arange(5 * 32 * 32 * 13).reshape(1, 5, 32, 32, 13)
    input_torch = torch.tensor(
        input, dtype=torch.float32,
    ).to('cpu')

    # Define unet
    assert unet(input_torch).shape == (1, 3, 32, 32, 13)


def test_MCUnet() -> None:
    """
    Function that tests MCUnet

    :return: None
    """
    # Initialize network
    unet = MCUnet(
        in_channels=5,
        out_channels=3,
        depth=4,
        n_features=2,
        kernel_size=3,
        pool_size=2,
        padding=1,
        activation='softmax',
        dropout=0.5,
        n_iter=5
    )

    # Define input
    input = np.arange(5 * 32 * 32 * 32).reshape(1, 5, 32, 32, 32)
    input_torch = torch.tensor(
        input, dtype=torch.float32,
    ).to('cpu')

    # Test output shape
    assert unet(input_torch).shape == (1, 3, 32, 32, 32)

    # Test stochasticity
    assert unet(input_torch).sum().item() != unet(input_torch).sum().item()

    # Test MCDropout
    assert unet.mc_forward(input_torch).shape == (5, 1, 3, 32, 32, 32)


def test_MCUnet2Axis() -> None:
    """
    Function that tests MCUnet2Axis

    :return: None
    """
    # Initialize network
    unet = MCUnet2Axis(
        in_channels=5,
        out_channels=3,
        depth=4,
        n_features=2,
        kernel_size=3,
        pool_size=2,
        padding=1,
        activation='softmax',
        dropout=0.5,
        n_iter=5
    )

    # Define input
    input = np.arange(5 * 32 * 32 * 32).reshape(1, 5, 32, 32, 32)
    input_torch = torch.tensor(
        input, dtype=torch.float32,
    ).to('cpu')

    # Test output shape
    assert unet(input_torch).shape == (1, 3, 32, 32, 32)

    # Test stochasticity
    assert unet(input_torch).sum().item() != unet(input_torch).sum().item()

    # Test MCDropout
    assert unet.mc_forward(input_torch).shape == (5, 1, 3, 32, 32, 32)


def test_MCUnet2Axis() -> None:
    """
    Function that tests MCUnet2Axis

    :return: None
    """
    # Initialize network
    unet = MCUnet2Axis(
        in_channels=5,
        out_channels=3,
        depth=4,
        n_features=2,
        kernel_size=3,
        pool_size=2,
        padding=1,
        activation='softmax',
        dropout=0.5,
        n_iter=5
    )

    # Define input
    input = np.arange(5 * 32 * 32 * 32).reshape(1, 5, 32, 32, 32)
    input_torch = torch.tensor(
        input, dtype=torch.float32,
    ).to('cpu')

    # Test output shape
    assert unet(input_torch).shape == (1, 3, 32, 32, 32)

    # Test stochasticity
    assert unet(input_torch).sum().item() != unet(input_torch).sum().item()


def test_ApproxMCUnet() -> None:
    """
    Function that tests ApproxMCUnet

    :return: None
    """
    # Initialize network
    unet = ApproxMCUnet(
        in_channels=5,
        out_channels=3,
        depth=4,
        n_features=2,
        kernel_size=3,
        pool_size=2,
        padding=1,
        activation='softmax',
        dropout=0.5,
        n_iter=5
    )

    # Define input
    input = np.arange(5 * 32 * 32 * 32).reshape(1, 5, 32, 32, 32)
    input_torch = torch.tensor(
        input, dtype=torch.float32,
    ).to('cpu')

    # Test output shape 'both' modality
    unet.modality = 'both'
    assert unet(input_torch).shape == (1, 6, 32, 32, 32)

    # Test output shape 'mean' modality
    unet.modality = 'mean'
    assert unet(input_torch).shape == (1, 3, 32, 32, 32)

    # Test output shape 'std' modality
    unet.modality = 'std'
    assert unet(input_torch).shape == (1, 3, 32, 32, 32)

