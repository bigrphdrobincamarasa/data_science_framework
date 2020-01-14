"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-01-07

**Project** : data_science_framework

**Module that contains the codes that implements  callback**
"""
from data_science_framework.pytorch_utils.metrics import SegmentationAccuracyMetric
from torch.utils.tensorboard import SummaryWriter
from data_science_framework.pytorch_utils.callbacks import Callback
import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class ConfusionMatrixCallback(Callback):
    """
    Class that implements ConfusionMatrixCallback

    :param writer: Tensorboad summary writter file
    :param nb_classes: Number of classes of the confusion_matrix
    """
    def __init__(
            self, writer: SummaryWriter,
        ) -> None:
        super(ConfusionMatrixCallback, self).__init__(writer=writer)
        self.training_confusion_matrices = []
        self.validation_confusion_matrices = []

    def on_epoch_start(self, epoch: int, model: nn.Module):
        """
        Method called on epoch start

        :param epoch: Epoch value
        :param model: Model under study
        """
        self.training_confusion_matrices = []
        self.validation_confusion_matrices = []

    def __call__(
            self, output: torch.Tensor, target: torch.Tensor,
            training: bool = True
        ) -> None:
        """
        Method call

        :param training: True if metric is computed on test
        """
        # Get output classification 
        output_classification = output.max(1)[1]\
                .cpu().detach().numpy().ravel()

        # Get classification version of the target tensor
        target_classification = target.max(1)[1]\
                .cpu().detach().numpy().ravel()

        # Compute confusion matrix
        confusion_matrix_ = confusion_matrix(
            target_classification, output_classification,
            normalize='true'
        )

        # Sum confusion matrix
        if training:
            self.training_confusion_matrices.append(
                np.copy(confusion_matrix_)
            )
        else:
            self.validation_confusion_matrices.append(
                np.copy(confusion_matrix_)
            )

    def on_epoch_end(
            self, epoch: int, model: nn.Module
        ) -> None:
        """
        Method called on epoch start

        :param epoch: Epoch value
        :param model: Model under study
        """
        for subset_, confusion_matrices_,  in [
            ('training', self.training_confusion_matrices),
            ('validation', self.validation_confusion_matrices),
        ]:
            self.writer.add_figure(
                'confusion_matrix/{}'.format(subset_),
                self.generate_plot(
                    title='Confusion matrix on {} at epoch {}'.format(
                        subset_, epoch
                    ),
                        confusion_matrices=np.array(confusion_matrices_)
                    ),
                epoch
            )

    def generate_plot(
            self, title: str, confusion_matrices: np.ndarray
        ) -> plt.figure:
        """
        Method that generate confusion_matrix plot

        :param title: Title of the graph
        :param confusion_matrix: Confusion matrix value
        """
        # Define title font
        fontdict = {
            'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 25,
        }

        # Define the number of classes
        nb_classes = confusion_matrices.shape[-1]

        # Clear matplotlib
        plt.figure(figsize=(22, 16), dpi=65)

        # Display heatmap
        confusion_matrix_std = confusion_matrices.std(axis=0)
        confusion_matrix_mean = confusion_matrices.mean(axis=0)
        plt.imshow(confusion_matrix_mean, cmap='jet')

        plt.xticks(range(nb_classes))
        plt.yticks(range(nb_classes -1, -1, -1))
        plt.title(title, fontdict=fontdict)
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

        # Get figure 
        figure = plt.gcf()

        # Return figure
        return figure

