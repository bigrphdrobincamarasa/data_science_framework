"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-19

**Project** : libraries

**Class that implements PrecisionPerClass**
"""
import torch
import numpy as np
from typing import Tuple, List
from data_science_framework.pytorch_utils.metrics import MetricPerClass
from sklearn.metrics import confusion_matrix


class PrecisionPerClass(MetricPerClass):
    """
    Class that implements PrecisionPerClass

    ;param name: Name of the metric
    """
    def __init__(self, name: str = 'precision_per_class'):
        super().__init__(name=name)

    def metric_function(
            self, output: np.ndarray,  target: np.ndarray
        ) -> float:
        """metric_function

        Compute the precision per class

        :param output: class output of the network as 1D numpy array
        :type output: np.ndarray
        :param target: class target of the network as 1D numpy array
        :type target: np.ndarray
        """
        tn, fp, fn, tp = confusion_matrix(target, output).ravel()
        return tp/(tp + fp)
