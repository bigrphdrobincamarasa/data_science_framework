"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-01-16

**Project** : baseline_unet

**Class that implements Analyser**
"""
from data_science_framework.data_spy.loggers.experiment_loggers import timer
from data_science_framework.pytorch_utils.optimizer import Optimizer
from data_science_framework.data_analyser.plotter import Plotter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from typing import List
import matplotlib.pyplot as plt
import torch


class Analyser(object):
    """
    Class that implements Analyser

    :param writer: Tensorboad summary writter file
    :param plotter: Plots in use for the analyse
    :param save_path: Path to the save folder
    """
    def __init__(
            self, writer: SummaryWriter,
            save_path: str
        ) -> None:
        self.writer = writer
        self.save_path = save_path

    def __call__(
            self, output: torch.Tensor,
            target: torch.Tensor, **kwargs
        ) -> None:
        """__call__

        :param output: Output of the network
        :type output: torch.Tensor
        :param target: Targetted ground truth
        :type target: torch.Tensor
        :rtype: None
        """
        pass

    def initialize_data() -> None:
        """initialize_data

        Initialize the data

        :rtype: None
        """
        pass

    def save_data() -> None:
        """save_data

        Saves the data to log files

        :rtype: None
        """
        pass

    def save_to_tensorboard(self) -> None:
        """save_to_tensorboard

        Save to tensorboard

        :rtype: None
        """
        pass
