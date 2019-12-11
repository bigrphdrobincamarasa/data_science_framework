"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-10

**Project** : src

**Class that handles optimizers**
"""
import torch
from torch.optim.adadelta import Adadelta


class Optimizer:
    """
    Class that handles optimizers

    :param name: Name of the optimizer
    :param learning_rate: Value of the learning rate
    """
    def __init__(
            self, name='adadelta', learning_rate=0.001
    ):
        self.name = name
        self.learning_rate = learning_rate
        self.optimizer_initializer = None

    def process_parameters(self):
        """
        Method that processes parameters

        :return: None
        """
        # Adadelta case
        if self.name == 'adadelta':
            self.optimizer_initialize = lambda x: Adadelta(x, lr=self.learning_rate)
        else:
            raise NotImplementedError

    def get_optimizer(self, model: torch.nn.Module):
        """
        Method that generates optimizer with the model

        :return: Optimizer object
        """
        return self.optimizer_initialize(model.parameters())

