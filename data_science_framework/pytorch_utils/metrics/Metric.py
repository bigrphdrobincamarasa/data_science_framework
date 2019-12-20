"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-19

**Project** : libraries

**Class that implements metric super class**
"""
import torch


class Metric(object):
    """
    Class that implements metric super class

    ;param name: Name of the metric
    """
    def __init__(self, name: str):
        self.name = name
        
    def compute(
            self, output: torch.Tensor, target: torch.Tensor
        ) -> float:
        """
        Compute the metric on output and target

        :param output: Output value of the Neural Network
        :param target: Target value of the Neural Network
        """
        pass
