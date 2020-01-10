"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-01-02

**Project** : data_science_framework

**Class that implements Callback superclass**
"""
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Callback:
    """
    Class that implements

    :param writer: Tensorboad summary writter file
    """
    def __init__(self, writer: SummaryWriter) -> None:
        self.writer = writer

    def on_epoch_end(self, epoch: int, model: nn.Module):
        """
        Method called on epoch end

        :param epoch: Epoch value
        """
        pass

    def on_epoch_start(self, epoch: int, model: nn.Module):
        """
        Method called on epoch start

        :param epoch: Epoch value
        :param model: Model under study
        """
        pass

    def save(self, epoch: int, model: nn.Module):
        """
        Method that save results to summary writter

        :param epoch: Epoch value
        :param model: Model under study
        """
        pass

    def __call__(
            self, output: torch.Tensor, target: torch.Tensor,
            training=True
        ) -> None:
        """
        Method call
        """
        pass

