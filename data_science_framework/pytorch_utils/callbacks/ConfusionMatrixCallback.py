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
from data_science_framework.data_analyser.plotter import ConfusionMatrixPlotter
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
    """
    def __init__(
            self, writer: SummaryWriter,
        ) -> None:
        super(ConfusionMatrixCallback, self).__init__(writer=writer)
        self.nb_classes=None

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
        # Get nb classes
        if self.nb_classes == None:
            self.nb_classes = output.shape[1]

        # Get output classification 
        output_classification = output.max(1)[1]\
                .cpu().detach().numpy().ravel()

        # Get classification version of the target tensor
        target_classification = target.max(1)[1]\
                .cpu().detach().numpy().ravel()

        # Compute confusion matrix
        confusion_matrix_ = confusion_matrix(
            target_classification, output_classification,
            normalize='true', labels=list(range(self.nb_classes))
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
            plotter = ConfusionMatrixPlotter(
                title='Confusion matrix on {} at epoch {}'.format(subset_, epoch),
                nb_classes=self.nb_classes
            )
            self.writer.add_figure(
                'confusion_matrix/{}'.format(subset_),
                plotter(np.array(confusion_matrices_)),
                epoch
            )

