"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-17

**Project** : baseline_unet

**Class that implements Loss**
"""
import torch


class Loss:
    """
    Class that implements Loss

    :param name: Name of the loss
    """
    def __init__(self, name='loss'):
        self.name = name


    def get_function(self):
        """
        Generate loss function

        :return: Loss function
        """
        pass
