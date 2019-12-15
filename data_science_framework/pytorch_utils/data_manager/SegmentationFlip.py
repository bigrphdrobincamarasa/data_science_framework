"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-15

**Project** : src

Class that implements SegmentationRotation
"""
from typing import Tuple
import numpy as np
from scipy.ndimage import rotate
import nibabel as nib

from data_science_framework.pytorch_utils.data_manager.SegmentationImageTransformation import \
    SegmentationImageTransformation


class SegmentationFlip(SegmentationImageTransformation):
    """
    Class that implements SegmentationRotation

    :param flip_x: Boolean that is true if function flips image along x axis
    :param flip_y: Boolean that is true if function flips image along y axis
    :param flip_z: Boolean that is true if function flips image along z axis
    :param random: True if randomness is enabled
    """
    def __init__(
            self, flip_x=False, flip_y=False,
            flip_z=False, random: bool = False
    ):
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.flip_z = flip_z
        self.random = random

    def get_transformation(self):
        """
        Get the transformation

        :return: Function that corresponds to the transformation
        """
        # Test randomness
        if self.random:
            flip_x = (self.flip_x and np.random.rand() < 0.5)
            flip_y = (self.flip_y and np.random.rand() < 0.5)
            flip_z = (self.flip_z and np.random.rand() < 0.5)
        else:
            flip_x = self.flip_x
            flip_y = self.flip_y
            flip_z = self.flip_z

        # Define flips
        flip_x_transform = lambda x: x[::-1, :, :] if flip_x else x
        flip_y_transform = lambda x: x[:, ::-1, :] if flip_y else x
        flip_z_transform = lambda x: x[:, :, ::-1] if flip_z else x
        flip_xyz = lambda x: flip_x_transform(flip_y_transform(flip_z_transform(x)))

        return lambda x: nib.Nifti1Image(
            flip_xyz(x.get_fdata()), affine=x.affine,
            header=x.header
        )
