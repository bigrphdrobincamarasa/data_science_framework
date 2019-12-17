"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-17

**Project** : baseline_unet

** Class that implements Adadelta **
"""
from typing import Callable

from torch.optim.adadelta import Adadelta

from data_science_framework.pytorch_utils.optimizer import Optimizer


class AdadeltaOptimizer(Optimizer):
    """
    Class that implements Adadelta

    :param name: Name of the optimizer
    :param learning_rate: Coefficient that scale delta before it is applied to the parameters
    :param rho: Coefficient used for computing a running average of squared gradients
    :param epsilon: Term added to the denominator to improve numerical stability
    """
    def __init__(
            self, name: str='adadelta', learning_rate: float=1, rho: float=1,
            epsilon: float= 1
    ):
        self.name = name
        self.rho = rho
        self.epsilon = epsilon
        self.learning_rate = learning_rate

    def get_torch(self) -> Callable:
        """
        Generate torch optimizer function

        :return: Function that returns optimizer initialized with model parameters
        """
        return lambda model: Adadelta(
            model.parameters(), lr=self.learning_rate,
            rho=self.rho, eps=self.epsilon
        )
