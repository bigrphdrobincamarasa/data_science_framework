"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-17

**Project** : baseline_unet

**File that tests codes of trainer module**
"""
from data_science_framework.data_analyser.plotter import Plotter, MODULE
from data_science_framework.scripting.test_manager import set_test_folders
from data_science_framework.settings import TEST_ROOT
import os
import matplotlib.pyplot as plt


@set_test_folders(
    output_root=TEST_ROOT,
    current_module=MODULE
)
def test_Plotter(output_folder: str) -> None:
    """
    Function that tests Tester

    :return: None
    """
    # Test object creation
    plotter = Plotter('test title')
    assert plotter.title == 'test title'

    # Test figure initialisation
    plotter.initialise_figure()
    plt.savefig('init.png')

    # Test clear figure
    plotter.clear_figure()
    plt.savefig('clear.png')

