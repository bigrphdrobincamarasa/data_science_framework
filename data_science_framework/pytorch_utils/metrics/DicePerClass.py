"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-19

**Project** : libraries

**Class that implements DicePerClass**
"""
import torch
import numpy as np
from typing import Tuple, List
from data_science_framework.pytorch_utils.metrics import Metric
from sklearn.metrics import confusion_matrix


class DicePerClass(Metric):
    """
    Class that implements DicePerClass

    ;param name: Name of the metric
    """
    def __init__(self, name: str = 'dice_per_class'):
        super().__init__(name=name)

    def metric_function(
            self, output: np.ndarray,  target: np.ndarray
        ) -> float:
        """metric_function

        Compute the dice per class

        :param output: class output of the network as 1D numpy array
        :type output: np.ndarray
        :param target: class target of the network as 1D numpy array
        :type target: np.ndarray
        """
        tn, fp, fn, tp = confusion_matrix(target, output).ravel()
        return 2 * tp/(2 * tp + fp + fn)

