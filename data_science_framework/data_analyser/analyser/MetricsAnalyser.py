"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-01-16

**Project** : data_science_framework

**Class that implements MetricsAnalyser**
"""
from data_science_framework.data_analyser.analyser import Analyser
from data_science_framework.pytorch_utils.metrics import Metric
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from typing import List
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
import numpy as np


class MetricsAnalyser(Analyser):
    """
    Class that implements MetricsAnalyser

    :param writer: Tensorboad summary writter file
    :param plotter: Plots in use for the analyse
    :param save_path: Path to the save folder
    :param subset_name: Name of the datasubset
    """
    def __init__(
            self, writer: SummaryWriter,
            save_path: str, subset_name: str,
            metrics: List[Metric]
        ) -> None:
        super(Analyser, self).__init__(
            writer=writer, save_path=save_path,
            subset_name=subset_name
        )
        self.metrics = metrics

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

    def save_data(self) -> None:
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

