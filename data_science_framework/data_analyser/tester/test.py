"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-17

**Project** : baseline_unet

**File that tests codes of trainer module**
"""
from data_science_framework.data_analyser.tester import Tester, MODULE, MCTester
from data_science_framework.data_analyser.analyser import Analyser
from data_science_framework.scripting.test_manager import set_test_folders
from data_science_framework.settings import TEST_ROOT
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np


@set_test_folders(
    output_root=TEST_ROOT,
    current_module=MODULE
)
def test_Tester(output_folder: str) -> None:
    """
    Function that tests Tester

    :param output: Path to the output folder
    :return: None
    """
    class FooAnalyser(Analyser):
        def __init__(self, i) -> None:
            self.data_dic = []
            self.i = i

        def __call__(
                self, output, target, **kwargs
            ) -> None:
            self.data_dic.append(
                (output, target, kwargs['meta'])
            )

        def initialize_data(self) -> None:
            self.data_dic = []

        def save_data(self) -> None:
            with open(os.path.join(
                output_folder, 'data{}.txt'.format(self.i)), 'w'
            ) as handle:
                handle.write(str(self.data_dic))

        def save_to_tensorboard(self) -> None:
             with open(os.path.join(
                output_folder, 'tensorboard{}.txt'.format(self.i)), 'w'
            ) as handle:
                handle.write(str(self.data_dic))

    # Test object creation
    tester = Tester(
        result_folder=output_folder,
        writer=SummaryWriter(log_dir=output_folder),
        dataset=[(1, 2, 3), (4, 5, 6)],
        model=lambda x: x,
        analysers=[FooAnalyser(i) for i in range(2)]
    )
    assert tester.result_folder == output_folder
    assert type(tester.writer) == type(
        SummaryWriter(log_dir=output_folder)
    )
    assert len(tester.dataset) == 2
    assert type(tester.analysers[0]) == type(FooAnalyser(3))
    assert len(tester.analysers) == 2

    # Test call
    tester()
    for analyser in tester.analysers:
        assert len(analyser.data_dic) == 2


@set_test_folders(
    output_root=TEST_ROOT,
    current_module=MODULE
)
def test_MCTester(output_folder: str) -> None:
    """
    Function that tests MCTester

    :param output: Path to the output folder
    :return: None
    """
    class FooAnalyser(Analyser):
        def __init__(self, i) -> None:
            self.data_dic = []
            self.i = i

        def __call__(
                self, output, target, **kwargs
            ) -> None:
            self.data_dic.append(
                (output, target, kwargs['meta'])
            )

        def initialize_data(self) -> None:
            self.data_dic = []

        def save_data(self) -> None:
            with open(os.path.join(
                output_folder, 'data{}.txt'.format(self.i)), 'w'
            ) as handle:
                handle.write(str(self.data_dic))

        def save_to_tensorboard(self) -> None:
             with open(os.path.join(
                output_folder, 'tensorboard{}.txt'.format(self.i)), 'w'
            ) as handle:
                handle.write(str(self.data_dic))

    class FooModel(object):
        def __call__(self, *args, **kwargs):
            return args, kwargs

        def mc_forward(self, input, *args, **kwargs):
            return input

    # Test object creation
    tester = MCTester(
        result_folder=output_folder,
        writer=SummaryWriter(log_dir=output_folder),
        dataset=[
            (np.array([[1]]), np.array([[2]]), np.array([[3]])),
            (np.array([[4]]), np.array([[5]]), np.array([[6]]))
        ],
        model=FooModel(),
        analysers=[FooAnalyser(i) for i in range(2)],
        mc_analysers=[FooAnalyser(i) for i in range(2)]
    )

    assert type(tester.mc_analysers[0]) == type(FooAnalyser(3))
    assert len(tester.mc_analysers) == 2

    # Test call
    tester()
    for mc_analyser in tester.mc_analysers:
        assert len(mc_analyser.data_dic) == 2
    for analyser in tester.analysers:
        assert len(analyser.data_dic) == 2
