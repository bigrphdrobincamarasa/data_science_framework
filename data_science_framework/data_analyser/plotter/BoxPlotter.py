"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-01-16

**Project** : data_science_framework

**Class that implements BoxPlotter**

"""
from data_science_framework.data_spy.loggers.experiment_loggers import timer
from data_science_framework.data_analyser.plotter import Plotter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from typing import Dict, List
import torch
import matplotlib.pyplot as plt
import pandas as pd

class BoxPlotter(Plotter):
    """
    Class that implements BoxPlotter
    """

    def __call__(
            self, data: pd.DataFrame
        ) -> plt.figure:
        """__call__

        :param data: Dataframe that contains the metris values
        :type data: pd.DataGrame
        :rtype: Dict[str, plt.figure]
        """
        self.initialise_figure()
        self.generate_figure(
            data={
                column: [value for value in column_data.values()]
                for column, column_data in data.to_dict().items()
            }
        )
        return self.figure

    def generate_figure(self, data: Dict[str, float]) -> None:
        """generate_figure

        Generate the boxplot figure

        :param data: Contains the values to plot
        :type data: Dict[str, float]
        :rtype: None
        """
        plt.boxplot(data.values())
        plt.xticks(range(1, len(data) + 1), data.keys())
        self.figure = plt.gcf()

