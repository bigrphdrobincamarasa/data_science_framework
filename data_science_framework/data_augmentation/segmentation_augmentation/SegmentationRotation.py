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

from data_science_framework.data_augmentation.segmentation_augmentation.SegmentationImageTransformation import \
    SegmentationImageTransformation


class SegmentationRotation(SegmentationImageTransformation):
    """
    Class that implements SegmentationRotation

    :param angle_x: If random is enabled it corresponds to the max rotation angle around x axis otherwise it corresponds to the value of the rotation angle
    :param angle_y: If random is enabled it corresponds to the max rotation angle around y axis otherwise it corresponds to the value of the rotation angle
    :param angle_z: If random is enabled it corresponds to the max rotation angle around z axis otherwise it corresponds to the value of the rotation angle
    :param random: True if randomness is enabled
    """
    def __init__(
            self, angle_x: float = 0, angle_y: float = 0,
            angle_z: float = 0, random: bool = False
    ):
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.random = random

    def get_transformation(self):
        """
        Get the transformation

        :return: Function that corresponds to the transformation
        """
        # Test randomness
        if self.random:
            angle_x = (2 * np.random.rand() - 1) * self.angle_x
            angle_y = (2 * np.random.rand() - 1) * self.angle_y
            angle_z = (2 * np.random.rand() - 1) * self.angle_z
        else:
            angle_x = self.angle_x
            angle_y = self.angle_y
            angle_z = self.angle_z

        # Define rotations
        rotate_x = lambda x: rotate(x, reshape=False, angle=angle_x, axes=(1, 2))
        rotate_y = lambda x: rotate(x, reshape=False, angle=angle_y, axes=(2, 0))
        rotate_z = lambda x: rotate(x, reshape=False, angle=angle_z, axes=(0, 1))
        rotate_xyz = lambda x: rotate_x(rotate_y(rotate_z(x)))
        return lambda x: nib.Nifti1Image(
            rotate_xyz(x.get_fdata()), affine=x.affine,
            header=x.header
        )
