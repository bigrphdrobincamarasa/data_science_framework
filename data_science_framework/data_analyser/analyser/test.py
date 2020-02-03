"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-01-21

**Project** : data_science_framework

**File that tests codes of analyser module**
"""
from data_science_framework.data_analyser.analyser import MODULE,\
        ConfusionMatricesAnalyser, MetricsAnalyser, ROCAnalyser
from data_science_framework.data_analyser.plotter import ConfusionMatrixPlotter
from data_science_framework.settings import TEST_ROOT
from data_science_framework.scripting.test_manager import set_test_folders
from data_science_framework.pytorch_utils.metrics import DicePerClass,\
        AccuracyPerClass, SensitivityPerClass, PrecisionPerClass,\
        SpecificityPerClass
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import torch
import os


@set_test_folders(
    output_root=TEST_ROOT,
    current_module=MODULE
)
def test_ConfusionMatricesAnalyser(output_folder: str) -> None:
    """
    Function that tests ConfusionMatricesAnalyser

    :param output_folder: Path to the output folder
    :return: None
    """
    # Test object creation
    analyser = ConfusionMatricesAnalyser(
        writer=SummaryWriter(log_dir=output_folder),
        save_path=output_folder,
        subset_name='test'
    )
    assert analyser.subset_name == 'test'
    assert analyser.save_path == output_folder
    assert type(analyser.writer) == type(SummaryWriter(log_dir=output_folder))
    assert analyser.confusion_matrices == []
    assert analyser.nb_classes is None
    assert analyser.confusion_matrix_plotter is None

    # Test call
    output = torch.rand((2, 5, 6, 7, 8))
    target = torch.rand((2, 5, 6, 7, 8))
    analyser(output, target)
    assert analyser.nb_classes == 5
    assert type(analyser.confusion_matrix_plotter) == type(
            ConfusionMatrixPlotter(
                title='',
                nb_classes=5
            )
    )
    output = torch.rand((2, 5, 6, 7, 8))
    target = torch.rand((2, 5, 6, 7, 8))
    analyser(output, target)
    assert len(analyser.confusion_matrices) == 2

    # Test save data
    analyser.save_data()
    assert np.load(
        os.path.join(output_folder, 'confusion_matrices_test.npy')
    ).shape  == (2, 5, 5)

    # Test save tensorboard
    analyser.save_to_tensorboard()


@set_test_folders(
    output_root=TEST_ROOT,
    current_module=MODULE
)
def test_MetricsAnalyser(output_folder: str) -> None:
    """
    Function that tests MetricsAnalyser

    :param output_folder: Path to the output folder
    :return: None
    """
    # Test object creation
    analyser = MetricsAnalyser(
        writer=SummaryWriter(log_dir=output_folder),
        save_path=output_folder,
        subset_name='test',
        metrics=[
            DicePerClass(), SensitivityPerClass(),
            AccuracyPerClass(), PrecisionPerClass(),
            SpecificityPerClass()
        ]
    )
    assert analyser.subset_name == 'test'
    assert analyser.save_path == output_folder
    assert type(analyser.writer) == type(SummaryWriter(log_dir=output_folder))
    assert type(analyser.dataframe) == type(pd.DataFrame())

    # Test call
    for i in range(10):
        output = torch.rand((2, 5, 6, 7, 8))
        target = torch.rand((2, 5, 6, 7, 8)) > 0.5
        analyser(output, target, meta={'test': 'test_{}'.format(i)})

    assert analyser.dataframe.shape == (10, 26)

    # Test save data
    analyser.save_data()

    assert pd.read_csv(
        os.path.join(output_folder, 'metrics_test.csv')
    ).shape  == (10, 26)

    assert pd.read_csv(
        os.path.join(output_folder, 'metrics_human_readable_test.csv')
    ).shape  == (5, 6)

    # Test save tensorboard
    analyser.save_to_tensorboard()


@set_test_folders(
    output_root=TEST_ROOT,
    current_module=MODULE
)
def test_ROCAnalyser(output_folder: str) -> None:
    """
    Function that tests ROCAnalyser

    :param output_folder: Path to the output folder
    :return: None
    """
    # Test object creation
    analyser = ROCAnalyser(
        writer=SummaryWriter(log_dir=output_folder),
        save_path=output_folder,
        subset_name='test',
        nb_thresholds=11
    )
    assert analyser.subset_name == 'test'
    assert analyser.save_path == output_folder
    assert type(analyser.writer) == type(SummaryWriter(log_dir=output_folder))
    assert analyser.nb_thresholds == 11
    assert len(analyser.threshold_range) == 11

    # Test call
    for i in range(10):
        output = torch.rand((2, 5, 6, 7, 8))
        target = torch.rand((2, 5, 6, 7, 8))
        analyser(output, target)
    assert len(analyser.acc) == 5
    assert len(analyser.acc[0]) == 10
    assert analyser.acc[0][0].shape == (11, 4)

    # Test save data
    analyser.save_data()
    assert pd.read_csv(
        os.path.join(
            output_folder, 'roc_curve_data.csv'
        )
    ).shape == (11, 10)

    # Test save tensorboard
    analyser.save_to_tensorboard()


