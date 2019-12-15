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


class SegmentationRotation(SegmentationImageTransformation):
    """
    Class that implements SegmentationRotation

    :param angle_x: If random is enabled it corresponds to the max rotation angle around x axis otherwise it corresponds to the value of the rotation angle
    :param angle_y: If random is enabled it corresponds to the max rotation angle around y axis otherwise it corresponds to the value of the rotation angle
    :param angle_z: If random is enabled it corresponds to the max rotation angle around z axis otherwise it corresponds to the value of the rotation angle
    :param random: True if randomness is enabled. In this case, the angle of rotation follows a uniforme distribution between -angle and angle for each direction
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
            angle_x = (2 * np.random.rand() - 1) * self.angle_x
            angle_y = (2 * np.random.rand() - 1) * self.angle_y
            angle_z = (2 * np.random.rand() - 1) * self.angle_z

        # Define rotations
        rotate_x = lambda x: rotate(x, reshape=False, angle=self.angle_x, axes=(1, 2))
        rotate_y = lambda x: rotate(x, reshape=False, angle=self.angle_y, axes=(2, 0))
        rotate_z = lambda x: rotate(x, reshape=False, angle=self.angle_z, axes=(0, 1))
        rotate_xyz = lambda x: rotate_x(rotate_y(rotate_z(x)))
        transform_image = lambda x: nib.Nifti1Image(
            rotate_xyz(x.get_fdata()), affine=x.affine,
            header=x.header
        )
        return transform_image

def flip_images(
    input_images: list, gt_images: list, flip_x=False, flip_y=False,
    flip_z=False, random: bool = False
):
    """
    Function that flip images

    :param input_images: List of input nifty images
    :param gt_images: List of ground truth nifty images
    :param random: Boolean that is true if the transformation is aleatoric
    :param flip_x: Boolean that is true if function flips image along x axis
    :param flip_y: Boolean that is true if function flips image along y axis
    :param flip_z: Boolean that is true if function flips image along z axis
    :param random: True if randomness is enabled
    :return: The modified input images and the modified gt images
    """
    # Test randomness
    if random:
        flip_x = (flip_x and np.random.rand() < 0.5)
        flip_y = (flip_y and np.random.rand() < 0.5)
        flip_z = (flip_z and np.random.rand() < 0.5)

    # Define flips
    flip_x_transform = lambda x: x[::-1, :, :] if flip_x else x
    flip_y_transform = lambda x: x[:, ::-1, :] if flip_y else x
    flip_z_transform = lambda x: x[:, :, ::-1] if flip_z else x
    flip_xyz = lambda x: flip_x_transform(flip_y_transform(flip_z_transform(x)))

    for i, input_image in enumerate(input_images):
        input_images[i] = nib.Nifti1Image(
            dataobj=flip_xyz(input_image.get_fdata()), affine=input_image.affine, header=input_image.header
        )

    for i, gt_image in enumerate(gt_images):
        gt_images[i] = nib.Nifti1Image(
            dataobj=flip_xyz(gt_image.get_fdata()), affine=gt_image.affine, header=gt_image.header
        )
    return input_images, gt_images
