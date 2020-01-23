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
        SegmentationBCEMetric, SegmentationDiceMetric, MetricPerClass, AccuracyPerClass,\
        SensitivityPerClass, SpecificityPerClass, PrecisionPerClass, DicePerClass
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


def test_MetricPerClass() -> None:
    """test_MetricPerClass

    Function that tests test_MetricPerClass

    :rtype: None
    """
    # Test compute
    metric_per_class = MetricPerClass(name='test')
    metric_per_class.metric_function = lambda x, y: (x.shape + y.shape)
    output = torch.rand((2, 3, 4, 5))
    target = torch.rand((2, 3, 4, 5))
    output_ = metric_per_class.compute(output, target)
    assert len(output_) == 3
    assert output_[0] == (40, 40)


def test_AccuracyPerClass() -> None:
    """test_AccuracyPerClass

    Function that tests test_AccuracyPerClass

    :rtype: None
    """
    # Test initialisation
    metric_per_class = AccuracyPerClass()
    assert metric_per_class.name == 'accuracy_per_class'

    # Test compute
    output = metric_per_class.metric_function(
        output=np.array([1, 0, 1, 0]),
        target=np.array([1, 0, 1, 1])
    )
    assert output == 0.75


def test_SensitivityPerClass() -> None:
    """test_SensitivityPerClass

    Function that tests test_SensitivityPerClass

    :rtype: None
    """
    # Test initialisation
    metric_per_class = SensitivityPerClass()
    assert metric_per_class.name == 'sensitivity_per_class'

    # Test compute
    output = metric_per_class.metric_function(
        output=np.array([1, 0, 1, 0]),
        target=np.array([1, 0, 1, 1])
    )
    assert (output - 0.66) ** 2 < 0.0001


def test_SpecificityPerClass() -> None:
    """test_SpecificityPerClass

    Function that tests test_SpecificityPerClass

    :rtype: None
    """
    # Test initialisation
    metric_per_class = SpecificityPerClass()
    assert metric_per_class.name == 'specificity_per_class'

    # Test compute
    output = metric_per_class.metric_function(
        output=np.array([1, 0, 1, 0]),
        target=np.array([1, 0, 1, 1])
    )
    assert output == 1


def test_PrecisionPerClass() -> None:
    """test_PrecisionPerClass

    Function that tests test_PrecisionPerClass

    :rtype: None
    """
    # Test initialisation
    metric_per_class = PrecisionPerClass()
    assert metric_per_class.name == 'precision_per_class'

    # Test compute
    output = metric_per_class.metric_function(
        output=np.array([1, 0, 1, 0]),
        target=np.array([1, 0, 1, 1])
    )
    assert output == 1


def test_DicePerClass() -> None:
    """test_DicePerClass

    Function that tests test_DicePerClass

    :rtype: None
    """
    # Test initialisation
    metric_per_class = DicePerClass()
    assert metric_per_class.name == 'dice_per_class'

    # Test compute
    output = metric_per_class.metric_function(
        output=np.array([1, 0, 1, 0]),
        target=np.array([1, 0, 1, 1])
    )
    assert output == 0.8
