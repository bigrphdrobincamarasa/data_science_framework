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

from data_science_framework.pytorch_utils.losses import BinaryCrossEntropyLoss


def test_BinaryCrossEntropyLoss() -> None:
    """
    Function that tests BinaryCrossEntropyLoss

    :return: None
    """
    binary_cross_entropy_loss = BinaryCrossEntropyLoss()

    # Test get function
    output = binary_cross_entropy_loss.get_torch()

    assert type(output) == type(F.binary_cross_entropy)