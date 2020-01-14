"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-19

**Project** : data_science_framework

**File that test the module Metric**
"""
from data_science_framework.pytorch_utils.metrics import SegmentationAccuracyMetric,\
        SegmentationBCEMetric, SegmentationDiceMetric
import numpy as np
import torch


def test_SegmentationAccuracyMetric() -> None:
    """
    Function that tests SegmentationAccuracyMetric

    :return: None
    """
    # Initialize metric
    segmentation_accuracy_metric = SegmentationAccuracyMetric()

    # Initialize arrays
    output_array = np.array(
        [
            [
                np.ones((4, 5, 6)),
                np.zeros((4, 5, 6)),
                np.zeros((4, 5, 6))
            ]
            for i in range(2)
        ]
    )
    target_array = np.array(
        [
            [
                np.zeros((4, 5, 6)),
                np.ones((4, 5, 6)),
                np.zeros((4, 5, 6))
            ]
            for i in range(2)
        ]
    )
    target_array[0, 0, 0, 0, 0] = 2

    # Initialize tensors
    output_tensor = torch.tensor(
            output_array, dtype=torch.float32
    ).to('cpu')
    target_tensor = torch.tensor(
            target_array, dtype=torch.float32
    ).to('cpu')

    # Compute Metric
    batch_size, cumulated_accuracy = segmentation_accuracy_metric.compute(
            output=output_tensor,
            target=target_tensor
    )
    assert batch_size == 2
    assert cumulated_accuracy == 1/120


def test_SegmentationBCEMetric() -> None:
    """
    Function that tests SegmentationAccuracyMetric

    :return: None
    """
    # Initialize metric
    segmentation_accuracy_metric = SegmentationBCEMetric()

    # Initialize arrays
    output_array = np.array(
        [
            [
                np.ones((4, 5, 6)),
                np.zeros((4, 5, 6)),
                np.zeros((4, 5, 6))
            ]
            for i in range(2)
        ]
    )
    target_array = np.array(
        [
            [
                np.zeros((4, 5, 6)),
                np.ones((4, 5, 6)),
                np.zeros((4, 5, 6))
            ]
            for i in range(2)
        ]
    )
    target_array[0, 0, 0, 0, 0] = 2

    # Initialize tensors
    output_tensor = torch.tensor(
            output_array, dtype=torch.float32
    ).to('cpu')
    target_tensor = torch.tensor(
            target_array, dtype=torch.float32
    ).to('cpu')

    # Compute Metric
    batch_size,  cumulated_accuracy = segmentation_accuracy_metric.compute(
            output=output_tensor,
            target=target_tensor
    )
    assert batch_size == 2
    assert (cumulated_accuracy - 36.7)**2 < 0.01


def test_SegmentationDiceMetric() -> None:
    """
    Function that tests SegmentationDiceMetric

    :return: None
    """
    # Initialize metric
    segmentation_dice_metric = SegmentationDiceMetric()

    # Initialize arrays
    output_array = np.array(
        [
            [
                np.ones((4, 5, 6)),
                np.zeros((4, 5, 6)),
                np.zeros((4, 5, 6))
            ]
            for i in range(2)
        ]
    )
    target_array = np.array(
        [
            [
                np.zeros((4, 5, 6)),
                np.ones((4, 5, 6)),
                np.zeros((4, 5, 6))
            ]
            for i in range(2)
        ]
    )
    target_array[0, 0, 0, 0, 0] = 2

    # Initialize tensors
    output_tensor = torch.tensor(
            output_array, dtype=torch.float32
    ).to('cpu')
    target_tensor = torch.tensor(
            target_array, dtype=torch.float32
    ).to('cpu')

    # Compute Metric
    batch_size,  cumulated_accuracy = segmentation_dice_metric.compute(
            output=output_tensor,
            target=target_tensor
    )
    assert batch_size == 2
    assert (cumulated_accuracy - 2.)**2 < 0.01
