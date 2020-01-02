"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-19

**Project** : data_science_framework

**Class that implements metric super class**
"""
import torch
from data_science_framework.pytorch_utils.metrics import Metric
from typing import Tuple


class SegmentationAccuracyMetric(Metric):
    """
    Class that implements metric super class

    :param: Name of the metric
    """
    def __init__(self, name='segmentation_accuracy'):
        self.name = name


    def compute(
            self, output: torch.Tensor, target: torch.Tensor
        ) -> Tuple:
        """
        Compute the metric on output and target

        :param output: Output value of the Neural Network
        :param target: Target value of the Neural Network
        :return: Cumulated accuracy over the batch and the batchsize
        """
        # Get classification version of tensors
        output_classification = output.max(1)[1]
        target_classification = target.max(1)[1]

        # Compute the accuracy per batch
        accuracy_per_batch = output_classification.eq(
            target_classification
        ).sum(0)/(1.0 * output[0, 0, :, :].flatten().shape[0])
        return accuracy_per_batch.sum(), output.shape[0]

