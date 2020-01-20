"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-01-16

**Project** : data_science_framework

**Class that implements Plotter**

"""
from data_science_framework.data_spy.loggers.experiment_loggers import timer
from data_science_framework.data_analyser.plotter import Plotter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt


class MetricPlotter(Plotter):
    """
    Class that implements Trainer
    """
    def __init__(
            self, title
        )  -> None:
        self.figure = plt.gcf()

    def initialise_figure(self) -> None:
        """initialise_figure

        Initialize matplotlib figure

        :rtype: None
        """
        plt.figure()
        plt.title(self.title)

    def clear_figure(self) -> None:
        """clear_figure

        Clear matplotlib figure

        :rtype: None
        """
        plt.clf()

    def __call__(self, data) -> plt.figure:
        """__call__

        :param data: Data on which the figure will be computed
        :rtype: plt.figure
        """
        pass

    def generate_figure(self) -> None:
        """generate_figure

        :rtype: None
        """
        pass

