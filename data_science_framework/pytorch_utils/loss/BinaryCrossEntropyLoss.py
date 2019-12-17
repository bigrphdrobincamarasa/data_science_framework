"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-17

**Project** : baseline_unet

**Class that implements binary cross entropy loss function**
"""
from data_science_framework.pytorch_utils.loss.Loss import Loss
import torch.nn.functional as F


class BinaryCrossEntropyLoss(Loss):
    """
    Class that implements binary cross entropy loss function

    :param name: Name of the loss
    """
    def __init__(self, name='binary_cross_entropy'):
        self.name = name

    def get_function(self):
        """
        Generate loss function

        :return: Loss function
        """
        return F.binary_cross_entropy

