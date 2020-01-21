"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-17

**Project** : baseline_unet

**File that tests codes of trainer module**
"""
from data_science_framework.data_analyser.plotter import Plotter, ConfusionMatrixPlotter,\
        MODULE
from data_science_framework.scripting.test_manager import set_test_folders
from data_science_framework.settings import TEST_ROOT
import os
import matplotlib.pyplot as plt
import numpy as np


@set_test_folders(
    output_root=TEST_ROOT,
    current_module=MODULE
)
def test_Plotter(output_folder: str) -> None:
    """
    Function that tests test_Plotter

    :param output: Path to the output folder
    :return: None
    """
    # Test object creation
    plotter = Plotter('test title')
    assert plotter.title == 'test title'

    # Test figure initialisation
    plotter.initialise_figure()
    plt.savefig(os.path.join(output_folder, 'init.png'))

    # Test clear figure
    plotter.clear_figure()
    plt.savefig(os.path.join(output_folder, 'clear.png'))


@set_test_folders(
    output_root=TEST_ROOT,
    current_module=MODULE
)
def test_ConfusionMatrixPlotter(output_folder: str) -> None:
    """
    Function that tests test_ConfusionMatrixPlotter

    :param output: Path to the output folder
    :return: None
    """
    # Test object creation
    plotter = ConfusionMatrixPlotter('test title', 5)
    assert plotter.cmap == 'jet'

    # Test figure initialisation
    plotter.initialise_figure()
    plt.savefig(os.path.join(output_folder, 'init.png'))

    # Test generate figure
    confusion_matrices = np.arange(100).reshape(4, 5, 5)
    confusion_matrices_mean = confusion_matrices.mean(axis=0)
    confusion_matrices_std = confusion_matrices.std(axis=0)
    figure = plotter.generate_figure(
        confusion_matrices_mean, confusion_matrices_std
    )
    plotter.figure.savefig(os.path.join(output_folder, 'confusion_matrix.png'))

    # Test clear figure
    plotter.clear_figure()
    plt.savefig(os.path.join(output_folder, 'clear.png'))

    # Test call
    figure = plotter(np.arange(100).reshape(4, 5, 5))
    plt.savefig(os.path.join(output_folder, 'call.png'))

