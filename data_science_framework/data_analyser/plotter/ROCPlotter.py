"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-01-16

**Project** : data_science_framework

**Class that implements ROCPlotter**

"""
from data_science_framework.data_spy.loggers.experiment_loggers import timer
from data_science_framework.data_analyser.plotter import Plotter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from typing import Dict, List
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class ROCPlotter(Plotter):
    """
    Class that implements ROCPlotter
    """
    def __call__(
            self, data: np.ndarray
        ) -> plt.figure:
        """__call__

        :param data: Array of shape (m, n, 4) m samples, n values of thresholds, 4 (tn, fp, fn, tp)
        :type data: np.ndarray
        :rtype: plt.figure
        """
        self.initialise_figure()

        # Compute metrics
        tn = np.sum(data[:, :, 0], axis=0)
        fp = np.sum(data[:, :, 1], axis=0)
        fn = np.sum(data[:, :, 2], axis=0)
        tp = np.sum(data[:, :, 3], axis=0)

        # Format data
        thresholds = np.linspace(0, 1, data.shape[0])
        sensitivity = tp/(tp + fn)
        one_less_specificity = 1 - (tn/(tn + fp))

        # Generate
        self.generate_figure(
            one_less_specificity=one_less_specificity,
            sensitivity=sensitivity,
            thresholds_values=thresholds
        )
        return self.figure

    def generate_figure(
            self, one_less_specificity: np.ndarray, sensitivity: np.ndarray,
            thresholds_values: np.ndarray
        ) -> None:
        """generate_figure

        Generate the boxplot figure

        :param one_less_specificity: Array of (m,) containing one less specificity values
        :type one_less_specificity: np.ndarray
        :param sensitivity: Array of (m,) containing sensitivity values
        :type sensitivity: np.ndarray
        :param thresholds_values: Array of shape (m,) containing the thresholds_values
        :type thresholds_values: np.ndarray
        :rtype: None
        """
        # Plot curve
        roc = plt.plot(
            one_less_specificity, sensitivity, marker='o',
            label='ROC curve'
        )

        # Plot random behaviour
        random = plt.plot(
            np.linspace(0, 1, 11),
            np.linspace(0, 1, 11),
            ls='--', label='Random'
        )

        # Compute distances to the up right corner
        best_point =np.argmin(
            np.sqrt(
                (1 - sensitivity)**2 + one_less_specificity**2
            )
        )

        # Plot best point distance
        line = plt.plot(
            np.linspace(
                0, one_less_specificity[best_point], 11
            ),
            np.linspace(
                1, sensitivity[best_point], 11
            ),
            label='Line to the optimal point'
        )

        # Display best point thresholds annotation
        plt.annotate(
            str(
                'Threshold : {}'.format(
                    thresholds_values[best_point]\
                            .__round__(3)
                )
            ),
            (
                one_less_specificity[best_point],
                sensitivity[best_point]
            )
        )
        plt.legend()
        plt.xticks(np.linspace(0, 1, 11))
        plt.yticks(np.linspace(0, 1, 11))
        self.figure = plt.gcf()

