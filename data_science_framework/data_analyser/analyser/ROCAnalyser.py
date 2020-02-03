"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-01-16

**Project** : data_science_framework

**Class that implements ROCAnalyser**
"""
from sklearn.metrics import confusion_matrix
from data_science_framework.data_analyser.plotter import ROCPlotter
from data_science_framework.data_analyser.analyser import Analyser
from data_science_framework.pytorch_utils.metrics import MetricPerClass
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from typing import List
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import os


class ROCAnalyser(Analyser):
    """
    Class that implements ROCAnalyser

    :param writer: Tensorboad summary writter file
    :param save_path: Path to the save folder
    :param subset_name: Name of the datasubset
    :param nb_thresholds: Number of thresholds
    """
    def __init__(
            self, writer: SummaryWriter,
            save_path: str, subset_name: str,
            nb_thresholds=11
        ) -> None:
        super().__init__(
            writer=writer, save_path=save_path,
            subset_name=subset_name
        )
        # Initialize attributes
        self.plotters = None
        self.acc = None
        self.nb_thresholds = nb_thresholds
        self.threshold_range = np.linspace(
            0, 1, nb_thresholds
        )

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
        if self.acc is None:
            self.acc = [[] for i in range(target.shape[1])]
            self.plotters = [
                ROCPlotter(title='ROC curve for class {}'.format(i))
                for i in range(target.shape[1])
            ]

        # Compute metrics for each class
        for i in range(target.shape[1]):
            acc_ = np.zeros((self.nb_thresholds, 4))
            for j, threshold in enumerate(self.threshold_range):
                tn, fp, fn, tp = confusion_matrix(
                    (target.max(1)[1] == i).detach().numpy().ravel(),
                    (
                        output[:, i, :] > threshold
                    ).detach().numpy().ravel(),
                ).ravel()
                acc_[j, :] = np.array([tn, fp, fn, tp])
            self.acc[i].append(acc_)

    def save_data(self) -> None:
        """save_data

        Saves the data to log files

        :rtype: None
        """
        # Transform accumulated data to numpy array
        acc_data = np.array(self.acc)

        # Compute sensitivity and specificity metric
        tn = acc_data[:, :, :, 0].sum(axis=1)
        fp = acc_data[:, :, :, 1].sum(axis=1)
        fn = acc_data[:, :, :, 2].sum(axis=1)
        tp = acc_data[:, :, :, 3].sum(axis=1)
        sensitivity = (tp/(tp + fn))
        sensitivity[np.isnan(sensitivity)] = 1
        specificity = tn/(tn + fp)
        specificity[np.isnan(specificity)] = 0

        # Construct dataframe
        dataframe = pd.DataFrame()

        for k, threshold in enumerate(
                self.threshold_range
            ):
            row_ = {}
            for i in range(sensitivity.shape[0]):
                row_.update(
                    **{
                        'class_{}_sensitivity'.format(i):\
                        sensitivity[i, k].__round__(3),
                        'class_{}_specificity'.format(i):\
                        specificity[i, k].__round__(3)
                    }
                )
            dataframe = dataframe.append(
                row_, ignore_index=True
            )
        dataframe.to_csv(
            os.path.join(
                self.save_path,
                'roc_curve_data.csv'
            ), index=False
        )


    def save_to_tensorboard(self) -> None:
        """save_to_tensorboard

        Save to tensorboard

        :rtype: None
        """
        for acc, plotter in zip(
            self.acc, self.plotters
        ):
            import ipdb; ipdb.set_trace() ###!!!BREAKPOINT!!!
            figure_ = plotter(np.array(acc))

            # Save figure
            self.writer.add_figure(
                'test/{}'.format(plotter.title),
                figure_
            )
