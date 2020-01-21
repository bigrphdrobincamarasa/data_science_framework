"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-19

**Project** : libraries

**Class that implements metric per class super class**
"""
import torch
import numpy as np
from typing import Tuple, List
from data_science_framework.pytorch_utils.metrics import Metric


class MetricPerClass(Metric):
    """
    Class that implements metric per class

    ;param name: Name of the metric
    """
    def __init__(self, name: str):
        super().__init__(name=name)

    def compute(
            self, output: torch.Tensor, target: torch.Tensor
        ) -> List:
        """
        Compute the metric on output and target

        :param output: Output value of the Neural Network
        :param target: Target value of the Neural Network
        :return: Dictionnary of metrics per class
        """
        return [
                self.metric_function(
                    output[:, i, :].cpu().detach().numpy().ravel(),
                    target[:, i, :].cpu().detach().numpy().ravel()
                )
            for i in range(output.shape[1])
        ]

    def metric_function(
            self, output: np.ndarray,  target: np.ndarray
        ):
        """metric_function

        :param output: class output of the network as 1D numpy array
        :type output: np.ndarray
        :param target: class target of the network as 1D numpy array
        :type target: np.ndarray
        """
