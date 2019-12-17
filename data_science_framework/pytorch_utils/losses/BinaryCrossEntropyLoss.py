"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-17

**Project** : baseline_unet

**Class that implements binary cross entropy losses function**
"""
from data_science_framework.pytorch_utils.losses import Loss
import torch.nn.functional as F


class BinaryCrossEntropyLoss(Loss):
    """
    Class that implements binary cross entropy losses function

    :param name: Name of the losses
    """

    def __init__(self, name='binary_cross_entropy'):
        super().__init__(name)

    def get_torch(self):
        """
        Generate torch loss function

        :return: Loss function
        """
        return F.binary_cross_entropy

