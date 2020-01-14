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


class DiceLoss(Loss):
    """
    Class that implements dice loss function

    :param name: Name of the losses
    :param epsilon: Value added to not divide by zero
    :param nb_dimension: Number of dimension of single (without  feature dimension and batchsize dimension)
    """
    def __init__(
            self, name='dice_loss', epsilon=0.001,
            nb_dimension=3
        ):
        super().__init__(name)
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
        def dice_loss(output, target):
            # Compute intersection feature wise
            intersection = (output * target).sum(self.dim_to_sum_along)

            # Compute cardinals feature wise
            output_cardinal = output.sum(self.dim_to_sum_along)
            target_cardinal = target.sum(self.dim_to_sum_along)

            # Return dice loss
            return 1. - (
                (2. * intersection) / (
                    output_cardinal + target_cardinal + self.epsilon
                )
            ).mean()
        return dice_loss
