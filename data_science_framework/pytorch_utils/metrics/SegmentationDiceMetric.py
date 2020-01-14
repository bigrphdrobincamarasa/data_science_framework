"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-01-07

**Project** : data_science_framework

**Class that implements metric super class**
"""
import torch
from data_science_framework.pytorch_utils.metrics import Metric
from data_science_framework.pytorch_utils.losses import DiceLoss
from typing import Tuple
import torch.nn.functional as F


class SegmentationDiceMetric(Metric):
    """
    Class that implements dice metric

    :param: Name of the metric
    """
    def __init__(self, name='dice'):
        self.name = name
        self.dice_function = DiceLoss().get_torch()

    def compute(
            self, output: torch.Tensor, target: torch.Tensor
        ) -> Tuple:
        """
        Compute the metric on output and target

        :param output: Output value of the Neural Network
        :param target: Target value of the Neural Network
        :return: Batch size and cumulated binary cross entropy
        """
        # Get classification version of tensors
        output_classification = output.max(1)[1]
        target_classification = target.max(1)[1]

        # Initialise accumulator
        acc = 0

        # Compute the accuracy per batch
        for i in range(output.shape[0]):
            acc += self.dice_function(
                torch.Tensor([output[0].detach().numpy()]),
                torch.Tensor([target[0].detach().numpy()])
            ).item()
        return output.shape[0], acc

