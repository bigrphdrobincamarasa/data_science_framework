"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-01-02

**Project** : data_science_framework

**Module that contains the codes that implements model plotter callback**
"""
from torch.utils.tensorboard import SummaryWriter
from .Callback import Callback
from data_science_framework.pytorch_utils.metrics import Metric
import torch
import torch.nn as nn
import os
import numpy as np


class ModelPlotter(Callback):
    """
    Class that implements ModelPlotter

    :param writer: Tensorboad summary writter file
    :param model: Model under study
    """
    def __init__(
            self, writer: SummaryWriter, model: nn.Module
        ) -> None:
        super(ModelPlotter, self).__init__(writer=writer)
        self.writer.add_text(
            'model structure',
            str(model)
        )

