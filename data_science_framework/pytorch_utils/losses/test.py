"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-17

**Project** : baseline_unet

**File that tests codes of losses module**
"""
import torch.nn.functional as F
import numpy as np
import torch

from data_science_framework.pytorch_utils.losses import BinaryCrossEntropyLoss, DiceLoss,\
        WeightedDiceLoss, WeightedCrossEntropy, GeneralizedDiceLoss


def test_BinaryCrossEntropyLoss() -> None:
    """
    Function that tests BinaryCrossEntropyLoss

    :return: None
    """
    binary_cross_entropy_loss = BinaryCrossEntropyLoss()

    # Test get function
    output = binary_cross_entropy_loss.get_torch()

    assert type(output) == type(F.binary_cross_entropy)


def test_DiceLoss() -> None:
    """
    Function that tests DiceLoss

    :return: None
    """
    dice_loss_object = DiceLoss()

    # Test get function
    dice_loss = dice_loss_object.get_torch()

    # Test the returned loss function
    filled_array = np.arange(4 * 5 * 6).reshape(4, 5, 6)
    output = torch.rand((1, 3, 4, 5, 6))
    target = torch.from_numpy(
            np.array(
                [
                    [
                        (filled_array % 3) == 0,
                        (filled_array % 3) == 1,
                        (filled_array % 3) == 2
                    ]
                ]
            )
    )
    loss_value = dice_loss(output, target)
    assert loss_value.detach().numpy().shape == ()


def test_WeightedDiceLoss() -> None:
    """
    Function that tests WeightedDiceLoss

    :return: None
    """
    weighted_dice_loss = WeightedDiceLoss(
        weights=[0.11, 0.88, 0.11], device='cpu'
    )

    # Test get function
    weighted_dice_loss = weighted_dice_loss.get_torch()

    # Test the returned loss function
    filled_array = np.arange(4 * 5 * 6).reshape(4, 5, 6)
    output = torch.rand((1, 3, 4, 5, 6))
    target = torch.from_numpy(
            np.array(
                [
                    [
                        (filled_array % 3) == 0,
                        (filled_array % 3) == 1,
                        (filled_array % 3) == 2
                    ]
                ]
            )
    )
    loss_value = weighted_dice_loss(output, target)
    assert loss_value.detach().numpy().shape == ()

def test_WeigthedCrossEntropy() -> None:
    """
    Function that tests WeigthedCrossEntropy

    :return: None
    """
    weighted_cross_entropy = WeightedCrossEntropy(device='cpu')

    # Test get function
    weighted_cross_entropy = weighted_cross_entropy.get_torch()

    # Test the returned loss function
    filled_array = np.arange(4 * 5 * 6).reshape(4, 5, 6)
    output = torch.rand((1, 3, 4, 5, 6))
    target = torch.from_numpy(
            np.array(
                [
                    [
                        (filled_array % 3) == 0,
                        (filled_array % 3) == 1,
                        (filled_array % 3) == 2
                    ]
                ]
            )
    )
    loss_value = weighted_cross_entropy(output, target)
    assert loss_value.detach().numpy().shape == ()


def test_GeneralizedDiceLoss() -> None:
    """
    Function that tests GeneralizedDiceLoss

    :return: None
    """
    generalized_dice_loss = GeneralizedDiceLoss(device='cpu')

    # Test get function
    generalized_dice_loss = generalized_dice_loss.get_torch()

    # Test the returned loss function
    filled_array = np.arange(4 * 5 * 6).reshape(4, 5, 6)
    output = torch.rand((1, 3, 4, 5, 6))
    target = torch.from_numpy(
            np.array(
                [
                    [
                        (filled_array % 3) == 0,
                        (filled_array % 3) == 1,
                        (filled_array % 3) == 2
                    ]
                ]
            )
    )
    loss_value = generalized_dice_loss(output, target)
    assert loss_value.detach().numpy().shape == ()

