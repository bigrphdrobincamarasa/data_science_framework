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
import numpy as np


class ConfusionMatrixPlotter(Plotter):
    """
    Class that implements Trainer

    :param title: Title of the figure
    :param cmap: Colorbar settings
    """
    def __init__(
            self, title: str,
            cmap='jet'
        )  -> None:
        super(Plotter, self).__init__(
            
        )
        self.cmap = cmap

    def initialise_figure(self) -> None:
        """initialise_figure

        Initialize matplotlib figure

        :rtype: None
        """
        plt.figure(figsize=(22, 16), dpi=65)
        plt.xticks(range(nb_classes))
        plt.yticks(range(nb_classes -1, -1, -1))
        plt.title(self.title)

    def __call__(self, confusion_matrices: np.ndarray) -> plt.figure:
        """__call__

        :param confusion_matrices: Matrices that will be plotted
        :rtype: plt.figure
        """
        self.initialise_figure()
        self.generate_figure(
            confusion_matrix_mean=confusion_matrices.mean(axis=0),
            confusion_matrices_std= confusion_matrices.std(axis=0)

        )
        self.clear_figure()
        return self.figure

    def generate_figure(
            self, confusion_matrix_mean: np.ndarray,
            confusion_matrices_std: np.array
        ) -> None:
        """generate_figure

        :param confusion_matrices: Confusion matrices over the dataset
        :type confusion_matrices: np.ndarray
        :param confusion_matrices_std: Confusion matrices standard deviation
        :type confusion_matrices_std: np.array
        :rtype: None
        """
        plt.imshow(confusion_matrix_mean, cmap=self.cmap)
        for i in range(nb_classes):
            for j in range(nb_classes):
                plt.text(
                    i, j,
                    '{}\n±\n{}'.format(
                        confusion_matrix_mean[i, j].__round__(3),
                        confusion_matrix_std[i, j].__round__(3),
                    ),
                )
        plt.colorbar()
        self.figure = plt.gcf()

