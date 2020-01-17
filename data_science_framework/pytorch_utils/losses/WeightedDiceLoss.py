"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-01-17

**Project** : data_science_framework

**Class that implements a weighted dice loss **

"""
from data_science_framework.pytorch_utils.losses import Loss
import torch.nn.functional as F
import torch


class WeightedDiceLoss(Loss):
    """
    Class that implements dice loss function

    :param name: Name of the losses
    :param epsilon: Value added to not divide by zero
    :param weights: Weights applied to each classes
    :param nb_dimension: Number of dimension of single (without  feature dimension and batchsize dimension)
    :param device: Device used by torch
    """
    def __init__(
            self, name='weigthed_dice_loss', epsilon=0.001,
            weights=[0.11, 0.11, 0.22, 0.22, 0.22, 0.11],
            nb_dimension=3, device='cuda'
        ):
        super().__init__(name)
        self.epsilon = epsilon
        self.nb_dimension = nb_dimension
        self.weights = weights
        self.device = device
        self.dim_to_sum_along =  tuple(
            [0] + list(range(2, self.nb_dimension + 2))
        )

    def get_torch(self):
        """
        Generate torch loss function

        :return: Loss function
        """
        def weighted_dice_loss(input, target):
            # Compute intersection feature wise
            intersection = (input * target).sum(self.dim_to_sum_along)

            # Compute cardinals feature wise
            output_cardinal = input.sum(self.dim_to_sum_along)
            target_cardinal = target.sum(self.dim_to_sum_along)

            # Get the dice
            dice_per_class = (
                (2. * intersection) / (
                    output_cardinal + target_cardinal + self.epsilon
                )
            )

            # Weights the dice
            weighted_dice = dice_per_class * torch.Tensor(self.weights).to(
                device=self.device
            )

            # Return dice loss
            return 1. - weighted_dice.sum()
        return weighted_dice_loss
