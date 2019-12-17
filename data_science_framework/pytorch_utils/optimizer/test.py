"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-17

**Project** : baseline_unet

**File that tests codes of optimizers module**
"""
from data_science_framework.pytorch_utils.optimizer import AdadeltaOptimizer


def test_Adadelta() -> None:
    """
    Function that tests Adadelta

    :return: None
    """
    # Initialize optimizer
    adadelta_optimizer = AdadeltaOptimizer()

    # Test get_torch
