"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-01-16

**Project** : data_science_framework

**Class that implements ConfusionMatricesAnalyser**
"""
from data_science_framework.data_analyser.plotter import ConfusionMatrixPlotter
from data_science_framework.data_analyser.analyser import Analyser
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from typing import List
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
import numpy as np


class MetricsAnalyser(Analyser):
    """
    Class that implements ConfusionMatricesAnalyser

    :param writer: Tensorboad summary writter file
    :param plotter: Plots in use for the analyse
    :param save_path: Path to the save folder
    :param subset_name: Name of the datasubset
    """
    def __init__(
            self, writer: SummaryWriter,
            save_path: str, subset_name: str,
            MetricsAnalyser
        ) -> None:
        super(Analyser, self).__init__(
            writer=writer, save_path=save_path
        )
        self.subset_name = subset_name
        self.confusion_matrix_plotter = ConfusionMatrixPlotter(
                title='Confusion matrix per image on {} dataset'.format(
                    self.subset_name
                )
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
        # Get nb classes
        nb_classes = output.shape[1]

        # Get output classification 
        output_classification = output.max(1)[1]\
                .cpu().detach().numpy().ravel()

        # Get classification version of the target tensor
        target_classification = target.max(1)[1]\
                .cpu().detach().numpy().ravel()

        # Compute confusion matrix
        confusion_matrix_ = confusion_matrix(
            target_classification, output_classification,
            normalize='true', labels=list(range(nb_classes))
        )

        # Append confusion matrix
        self.confusion_matrices.append(confusion_matrix)

    def save_data(self) -> None:
        """save_data

        Saves the data to log files

        :rtype: None
        """
        # Save data
        np.save(
            os.path.join(
                self.save_path, '{}.npy'.format(self.subset_name)
            ),
            np.array(self.confusion_matrix)
        )

    def save_to_tensorboard(self) -> None:
        """save_to_tensorboard

        Save to tensorboard

        :rtype: None
        """
        self.writer.add_figure(
            'test/confusion_matrix_{}'.format(subset_name),
            self.confusion_matrix_plotter(
                self.confusion_matrix_plotter
            )
        )

