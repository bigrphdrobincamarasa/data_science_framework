"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-01-02

**Project** : data_science_framework

**Module that contains the codes that implements model checkpoint callback**
"""
from torch.utils.tensorboard import SummaryWriter
from .Callback import Callback
from data_science_framework.pytorch_utils.metrics import Metric
import torch
import torch.nn as nn
import os
import numpy as np


class ModelCheckpoint(Callback):
    """
    Class that implements ModelCheckpoint

    :param writer: Tensorboad summary writter file
    :param metric: Metric under study
    :param save_folder: Folder that contains the saved model
    :param metric_to_minimize: True if the have to maximized
    """
    def __init__(
            self, writer: SummaryWriter, metric: Metric,
            save_folder: str, metric_to_minimize: bool = True
        ) -> None:
        super(ModelCheckpoint, self).__init__(writer=writer)
        self.metric = metric
        self.save_folder = save_folder
        self.metric_to_minimize = metric_to_minimize

        self.best_score = np.inf if self.metric_to_minimize else -np.inf
        self.best_epoch = 0
        self.on_epoch_start(0, None)

    def on_epoch_start(self, epoch: int, model: nn.Module):
        """
        Method called on epoch start
        
        :param epoch: Epoch value
        :param model: Model under study
        """
        self.metric_values = (self.metric, 0, 0)

    def on_epoch_end(self, epoch: int, model: nn.Module):
        """
        Method called on epoch end
        
        :param epoch: Epoch value
        :param model: Model under study
        """
        # Save results

        # Compute epoch score
        if self.metric_values[1] != 0:
            epoch_score = self.metric_values[2] / self.metric_values[1]
        else:
            epoch_score = self.best_score

        # Update best values
        if (self.metric_to_minimize and epoch_score < self.best_score) or\
           (not self.metric_to_minimize and epoch_score > self.best_score):
            self.best_score = epoch_score
            self.best_epoch = epoch
            torch.save(
                model.state_dict(), os.path.join(
                    self.save_folder,
                    'model_epoch_{}.pt'.format(epoch)
                )
            )

            torch.save(
                model.state_dict(), os.path.join(
                    self.save_folder,
                    'best_model_{}.pt'.format(self.metric.name)
                )
            )
            print('\t- ModelCheckpoint {} : Best score'.format(self.metric.name))
            # Log to tensorboard
            self.writer.add_text(
                'model checkpoint',
                'Best score for {} : {}'.format(
                    self.metric.name,
                    self.best_score
                ),
                epoch
            )

    def __call__(
            self, output: torch.Tensor,
            target: torch.Tensor, training: bool = True
        ) -> None: 
        """
        Method call

        :param training: True if metric is computed on test
        """
        # Launch 
        if not training:

            # Compute metric
            acc_, value_ = self.metric_values[0].compute(
                output=output, target=target
            )

            # Accumulate results
            self.metric_values = (
                self.metric_values[0],
                self.metric_values[1] + acc_,
                self.metric_values[2] + value_,
            )

