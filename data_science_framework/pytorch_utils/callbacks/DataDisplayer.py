"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-01-07

**Project** : data_science_framework

**Module that contains the codes that implements 3D data displayer callback**
"""
from torch.utils.tensorboard import SummaryWriter
from data_science_framework.pytorch_utils.callbacks import Callback
import torch
import torch.nn as nn
import os
import numpy as np


class DataDisplayer(Callback):
    """
    Class that implements DataDisplayer

    :param writer: Tensorboad summary writter file
    """
    def __init__(
            self, writer: SummaryWriter
        ) -> None:
        super(DataDisplayer, self).__init__(writer=writer)
        self.current_epoch = 0
        self.saved_image_training = False
        self.saved_image_validation = False

    def on_epoch_start(self, epoch: int, model: nn.Module):
        """
        Method called on epoch start

        :param epoch: Epoch value
        :param model: Model under study
        """
        self.current_epoch += 1
        self.saved_image_training = False
        self.saved_image_validation = False

    def __call__(
            self, output: torch.Tensor, target: torch.Tensor,
            training: bool = True
        ) -> None:
        """
        Method call

        :param training: True if metric is computed on test
        """
        # Analyse training variable
        if training:
            state = 'training'
        else:
            state = 'validation'

        if (training and not self.saved_image_training) or\
            (not training and not self.saved_image_validation):
            # Update saved image attributes
            if training:
                self.saved_image_training = True
            else:
                self.saved_image_validation = True

            # Save output images
            for i in range(output.shape[1]):
                self.writer.add_image(
                    'class_{}/{}_output'.format(i, state),
                    np.array(
                        [
                            output[
                                0, i, :,
                                :, int(output.shape[-1]/2)
                            ].detach().cpu().numpy(),
                        ]
                    ),
                    self.current_epoch
                )

            # Save target images
            for i in range(target.shape[1]):
                self.writer.add_image(
                    'class_{}/{}_target'.format(i, state),
                    np.array(
                        [
                            target[
                                0, i, :,
                                :, int(target.shape[-1]/2)
                            ].detach().cpu().numpy(),
                        ]
                    ),
                    self.current_epoch
                )

