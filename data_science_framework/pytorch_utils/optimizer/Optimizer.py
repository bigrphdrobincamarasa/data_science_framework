"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-17

**Project** : baseline_unet

** Class that implements Optimizer **
"""
from typing import Callable


class Optimizer:
    """
    Class that implements Optimizer

    :param name: Name of the optimizer
    """
    def __init__(self, name='loss'):
        self.name = name

    def get_torch(self) -> Callable:
        """
        Generate torch optimizer function

        :return: Function that returns optimizer initialized with model parameters
        """
        pass

