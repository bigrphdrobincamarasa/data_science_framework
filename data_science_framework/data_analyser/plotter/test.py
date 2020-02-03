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
        MODULE, BoxPlotter, ROCPlotter
from data_science_framework.scripting.test_manager import set_test_folders
from data_science_framework.settings import TEST_ROOT
import pandas as pd
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
    assert plotter.cmap == 'viridis'

    # Test figure initialisation
    plotter.initialise_figure()
    plt.savefig(os.path.join(output_folder, 'init.png'))

    # Test generate figure
    confusion_matrices = np.arange(100).reshape(4, 5, 5)
    confusion_matrices = confusion_matrices.sum(axis=0) \
            / confusion_matrices.sum()
    figure = plotter.generate_figure(
        confusion_matrices
    )
    plotter.figure.savefig(
        os.path.join(output_folder, 'confusion_matrix.png')
    )

    # Test clear figure
    plotter.clear_figure()
    plt.savefig(os.path.join(output_folder, 'clear.png'))

    # Test call
    figure = plotter(np.arange(100).reshape(4, 5, 5))
    plt.savefig(os.path.join(output_folder, 'call.png'))


@set_test_folders(
    output_root=TEST_ROOT,
    current_module=MODULE
)
def test_BoxPlotter(output_folder: str) -> None:
    """
    Function that tests test_BoxPlotter

    :param output: Path to the output folder
    :return: None
    """
    plotter = BoxPlotter('test title')

    # Test figure initialisation
    plotter.initialise_figure()
    plt.savefig(os.path.join(output_folder, 'init.png'))

    # Test generate figure
    data = {
        'a': [i for i in range(5)],
        'b': [i for i in range(5, 10)],
        'c': [i for i in range(-1, 4)],
    }

    plotter.generate_figure(data)
    plotter.figure.savefig(
        os.path.join(output_folder, 'boxplot.png')
    )

    # Test call
    figure = plotter(pd.DataFrame(data))

    plotter.figure.savefig(
        os.path.join(output_folder, 'boxplot_call.png')
    )


@set_test_folders(
    output_root=TEST_ROOT,
    current_module=MODULE
)
def test_ROCPlotter(output_folder: str) -> None:
    """
    Function that tests test_ROCPlotter

    :param output: Path to the output folder
    :return: None
    """
    plotter = ROCPlotter('test title')

    # Test figure initialisation
    plotter.initialise_figure()
    plt.savefig(os.path.join(output_folder, 'init.png'))

    # Test generate figure
    sensitivity = np.sort(
        np.array(
            [np.random.rand() for i in range(11)]
        )
    )
    one_less_specificity = np.sort(
        np.array(
            [np.random.rand() for i in range(11)]
        )
    )
    thresholds_values = np.linspace(0, 1, 11)
    plotter.generate_figure(
        sensitivity=sensitivity,
        one_less_specificity=one_less_specificity,
        thresholds_values=thresholds_values
    )
    plotter.figure.savefig(
        os.path.join(output_folder, 'roc.png')
    )

    # Test call
    data=np.arange(10 * 100 * 4).reshape(100, 10, 4)
    plotter(data=data)
    plotter.figure.savefig(
        os.path.join(output_folder, 'roc_call.png')
    )



