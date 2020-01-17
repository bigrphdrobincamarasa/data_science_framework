"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-01-17

**Project** : data_science_framework

**Class that implements a weighted cross entropy loss **

"""
from data_science_framework.pytorch_utils.losses import Loss
import torch.nn.functional as F
import torch


class WeightedCrossEntropy(Loss):
    """
    Class that implements weighted cross entropy loss function

    :param name: Name of the losses
    :param epsilon: Value added to not divide by zero
    :param weights: Weights applied to each classes
    :param nb_dimension: Number of dimension of single (without  feature dimension and batchsize dimension)
    :param device: Device used by torch
    """
    def __init__(
            self, name='weigthed_cross_entropy_loss', epsilon=0.001,
            nb_dimension=3, device='cuda'
        ):
        super().__init__(name)
        self.epsilon = epsilon
        self.nb_dimension = nb_dimension
        self.device = device
        self.dim_to_sum_along =  tuple(
            [0] + list(range(2, self.nb_dimension + 2))
        )

    def get_torch(self):
        """
        Generate torch loss function

        :return: Loss function
        """
        def weigthed_cross_entropy(
                input: torch.Tensor, target: torch.Tensor
            ):
            # Compute cardinals feature wise
            target_cardinal = target.sum(self.dim_to_sum_along)
            binary_class_mask = target_cardinal != 0

            # Get the inversed class proportion
            inversed_class_proportion = 1. /(
                target_cardinal + self.epsilon
            ) * target_cardinal.sum() * binary_class_mask

            # Compute the class weights
            class_weights = inversed_class_proportion /\
                    inversed_class_proportion.sum()

            # Create pixelwise weigths
            weights = torch.ones(input.shape)
            for i in range(weights.shape[1]):
                weights[: , i, :] = class_weights[i]

            weights = weights.to(self.device)

            return F.binary_cross_entropy(input, 1. * target, weight=weights)
        return weigthed_cross_entropy
