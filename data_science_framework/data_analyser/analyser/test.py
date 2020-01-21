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
        ConfusionMatricesAnalyser, MetricsAnalyser
from data_science_framework.data_analyser.plotter import ConfusionMatrixPlotter
from data_science_framework.settings import TEST_ROOT
from data_science_framework.scripting.test_manager import set_test_folders
from torch.utils.tensorboard import SummaryWriter
import numpy as np
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
