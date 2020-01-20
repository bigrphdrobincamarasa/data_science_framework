"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-01-20

**Project** : data_science_framework

**Class that implements a generalized dice loss function**

"""
from data_science_framework.pytorch_utils.losses import Loss
import torch.nn.functional as F
import torch


class GeneralizedDiceLoss(Loss):
    """
    Class that implements a generalized dice loss function that delete None classes

    :param name: Name of the losses
    :param device: Device used by torch
    """
    def __init__(
            self, name='generalized_dice_loss',
            epsilon = 0.001,
            nb_dimension=3, device='cuda'
        ):
        super().__init__(name)
        self.device = device
        self.epsilon = epsilon
        self.nb_dimension = nb_dimension
        self.dim_to_sum_along =  tuple(
            [0] + list(range(2, self.nb_dimension + 2))
        )

    def get_torch(self):
        """
        Generate torch loss function

        :return: Loss function
        """
        def generalized_dice_loss(input, target):
            # Compute intersection feature wise
            intersection = (input * target).sum(self.dim_to_sum_along)

            # Compute cardinals feature wise
            output_cardinal = input.sum(self.dim_to_sum_along)
            target_cardinal = target.sum(self.dim_to_sum_along)

            # Compute the squared of the L1 norm of the target
            target_norm = (target.sum(self.dim_to_sum_along)) ** 2

            # Get null class mask
            class_mask = 1. * (target_norm != 0)

            # Compute coefficients (1/||GT||**2)
            coefficients = class_mask / (target_norm + self.epsilon)

            # Normalization constant
            norm_constant = (
                coefficients * (
                    output_cardinal + target_cardinal
                ) / 2

            ).sum()

            # Return generalized dice value
            return (coefficients * intersection / norm_constant).sum()

        return generalized_dice_loss
