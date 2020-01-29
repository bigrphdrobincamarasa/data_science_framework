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
import matplotlib.pyplot as plt


class ConfusionMatrixPlotter(Plotter):
    """
    Class that implements Trainer

    :param title: Title of the figure
    :param nb_classes: Number of classes studied
    :param cmap: Colorbar settings
    """
    def __init__(
            self, title: str,
            nb_classes,
            cmap='viridis'
        )  -> None:
        super().__init__(title)
        self.nb_classes = nb_classes
        self.cmap = cmap

    def initialise_figure(self) -> None:
        """initialise_figure

        Initialize matplotlib figure

        :rtype: None
        """
        plt.figure(figsize=(22, 16), dpi=65)
        plt.xticks(range(self.nb_classes))
        plt.yticks(range(self.nb_classes -1, -1, -1))
        plt.title(self.title)

    def __call__(self, confusion_matrices: np.ndarray) -> plt.figure:
        """__call__

        :param confusion_matrices: Matrices that will be plotted
        :rtype: plt.figure
        """
        self.initialise_figure()

        # Sum values confusion matrices
        summed_confusion_matrices = confusion_matrices\
                .sum(axis=0).astype(float)

        # Divide by class cardinal
        for i in range(
            summed_confusion_matrices.shape[1]
        ):
            summed_confusion_matrices[i, :] = summed_confusion_matrices[i, :] \
                    / summed_confusion_matrices[i, :].sum()

        self.generate_figure(
            confusion_matrix_mean=summed_confusion_matrices,
        )
        return self.figure

    def generate_figure(
            self, confusion_matrix_mean: np.ndarray
        ) -> None:
        """generate_figure

        :param confusion_matrices: Confusion matrices over the dataset
        :type confusion_matrices: np.ndarray
        :rtype: None
        """
        plt.imshow(confusion_matrix_mean, cmap=self.cmap)
        for i in range(self.nb_classes):
            for j in range(self.nb_classes):
                plt.text(
                    i, j,
                    '{}'.format(
                        confusion_matrix_mean[j, i].__round__(3),
                    )
                )
        plt.colorbar()
        self.figure = plt.gcf()

