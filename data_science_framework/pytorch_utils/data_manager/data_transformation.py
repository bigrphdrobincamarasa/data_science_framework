"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-09

**Project** : baseline_unet

TODO: doc
** File that contains the codes that implements data transformations **
"""
from typing import Tuple

from scipy.ndimage import rotate
import nibabel as nib
import numpy as np


def apply_transformations_to_batch(
        input_batch: list, gt_batch: list,
        transformations: list, random: bool = True
) -> Tuple:
    """
    Function that applies transformations to a batch of images

    :param input_batch: Input images of the batch
    :param gt_batch: Ground truth images of the
    :param random: True if randomness is enabled
    :return: the updated input batch and gt batch
    """
    pass


def rotate_images(
        input_images: list, gt_images: list,
        angle_x: int, angle_y: int, angle_z: int,
) -> Tuple:
    """
    Function that apply the same rotation to the two set of images

    :param input_images: List of input nifty images
    :param gt_images: List of ground truth nifty images
    :param angle_x: If random is enabled it corresponds to the max rotation angle around x axis
    otherwise it corresponds to the value of the rotation angle
    :param angle_y: If random is enabled it corresponds to the max rotation angle around y axis
    otherwise it corresponds to the value of the rotation angle
    :param angle_z: If random is enabled it corresponds to the max rotation angle around z axis
    otherwise it corresponds to the value of the rotation angle
    :return: The modified input images and the modified gt images
    """
    # Define rotations
    rotate_x = lambda x: rotate(x, reshape=False, angle=angle_x, axes=(1, 2))
    rotate_y = lambda x: rotate(x, reshape=False, angle=angle_y, axes=(2, 0))
    rotate_z = lambda x: rotate(x, reshape=False, angle=angle_z, axes=(0, 1))
    rotate_xyz = lambda x: rotate_x(rotate_y(rotate_z(x)))

    for i, input_image in enumerate(input_images):
        input_images[i] = nib.Nifti1Image(
            dataobj=rotate_xyz(input_image.get_fdata()), affine=input_image.affine, header=input_image.header
        )

    for i, gt_image in enumerate(gt_images):
        gt_images[i] = nib.Nifti1Image(
            dataobj=rotate_xyz(gt_image.get_fdata()), affine=gt_image.affine, header=gt_image.header
        )
    return input_images, gt_images


def crop_half_images(
        input_images: list, gt_images: list
) -> Tuple:
    """
    Function that crop the half of the image containing non null ground truth

    :param input_images: List of input nifty images
    :param gt_images: List of ground truth nifty images
    :return: The modified input images and the modified gt images
    """
    #TODO: implements
    pass


def flip_images(
    input_images: list, gt_images: list, flip_x=True, flip_y=True,
    flip_z=True
):
    """
    Function that flip images

    :param input_images: List of input nifty images
    :param gt_images: List of ground truth nifty images
    :param random: Boolean that is true if the transformation is aleatoric
    :param flip_x: Boolean that is true if function flips image along x axis
    :param flip_y: Boolean that is true if function flips image along y axis
    :param flip_z: Boolean that is true if function flips image along z axis
    :return: The modified input images and the modified gt images
    """
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

