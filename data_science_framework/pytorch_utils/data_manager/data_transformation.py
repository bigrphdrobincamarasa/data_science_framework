"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-09

**Project** : baseline_unet

**File that contains the codes that implements data transformations**
"""
from typing import Tuple

from scipy.ndimage import rotate
import nibabel as nib
import numpy as np


def tile_images(
        input_images: list, gt_images: list,
        shape=(16, 16, 16), expansion_factor: int = 2
) -> Tuple:
    """
    Function that tiles input images and gt_images the same way (all images must have the same size)

    :param input_images: Untiled nifty images
    :param gt_images: Untiled nifty images
    :param shape: Shape of the tiled images
    :param expansion_factor: Expansion factor linked to the overlap value
    :return: Tuple of tiled input_images, tiled gt_images and sum of the gt on the tile
    """
    # Get images shapes
    image_shape = input_images[0].shape
    tile_start_points = []

    # Get tiles start points on each dimension
    for dimension in range(3):
        tmp_ = []
        for k in range(expansion_factor):
            # Get dimension indices
            tmp_ += [
                        i
                        for i in range(
                        int((k * shape[dimension]) / expansion_factor),
                        image_shape[dimension],
                        shape[dimension]
                        )
                    ][:-1]

        # Append border
        tmp_.append(image_shape[dimension] - shape[dimension])

        # Remove duplicates
        tmp_ = list(set(tmp_))

        # Sort
        tmp_.sort()
        tile_start_points.append(
            tmp_.copy()
        )

    # Initialize output
    input_images_, gt_images_, gt_sum_ = [], [], []

    # Loop over start points on each dimension
    for i in tile_start_points[0]:
        for j in tile_start_points[1]:
            for k in tile_start_points[2]:
                input_images_.append(
                    [
                        nib.Nifti1Image(
                            dataobj=input_image.get_fdata()[
                                i:i+shape[0],
                                j:j+shape[1],
                                k:k+shape[2],
                            ],
                            affine=input_image.affine,
                            header=input_image.header
                        )
                        for input_image in input_images
                    ]
                )
                gt_images_.append(
                    [
                        nib.Nifti1Image(
                            dataobj=gt_image.get_fdata()[
                                    i:i+shape[0],
                                    j:j+shape[1],
                                    k:k+shape[2],
                                    ],
                            affine=gt_image.affine,
                            header=gt_image.header
                        )
                        for gt_image in gt_images
                    ]
                )
                gt_sum_.append(
                    np.array(
                        [
                            gt_image.get_fdata()[
                                i:i + shape[0],
                                j:j + shape[1],
                                k:k + shape[2]
                            ]
                            for gt_image in gt_images
                        ]
                    ).sum()
                )
    return input_images_, gt_images_, gt_sum_





def apply_transformations_to_batch(
        input_batch: list, gt_batch: list,
        transformations: list
) -> Tuple:
    """
    Function that applies transformations to a batch of images

    :param input_batch: Input images of the batch
    :param gt_batch: Ground truth images of the
    :return: the updated input batch and gt batch
    """
    # Initialise output
    input_batch_, gt_batch_ = [], []

    # Loop over the images
    for input_images, gt_images in zip(input_batch, gt_batch):
        input_images_, gt_images_ = input_images.copy(), gt_images.copy()

        # Loop over the transformations
        for transformation in transformations:
            input_images_, gt_images_ = transformation(
                input_images_, gt_images_
            )
        # Add transformed images
        input_batch_.append(input_images_.copy())
        gt_batch_.append(gt_images_.copy())

    # Return output
    return input_batch_, gt_batch_


def rotate_images(
        input_images: list, gt_images: list,
        angle_x: float = 0, angle_y: float = 0, angle_z: float = 0,
        random: bool = False
) -> Tuple:
    """
    Function that apply the same rotation to the two set of images

    :param input_images: List of input nifty images
    :param gt_images: List of ground truth nifty images
    :param angle_x: If random is enabled it corresponds to the max rotation angle around x axis otherwise it corresponds to the value of the rotation angle
    :param angle_y: If random is enabled it corresponds to the max rotation angle around y axis otherwise it corresponds to the value of the rotation angle
    :param angle_z: If random is enabled it corresponds to the max rotation angle around z axis otherwise it corresponds to the value of the rotation angle
    :param random: True if randomness is enabled. In this case, the angle of rotation follows a uniforme distribution between -angle and angle for each direction
    :return: The modified input images and the modified gt images
    """
    # Test randomness
    if random:
        angle_x = (2 * np.random.rand() - 1) * angle_x
        angle_y = (2 * np.random.rand() - 1) * angle_y
        angle_z = (2 * np.random.rand() - 1) * angle_z

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
    input_images_ = input_images.copy()
    gt_images_ = gt_images.copy()

    # Define new dimension
    updated_second_dimension = int(gt_images[0].shape[1]/2)

    # Analyse ground truth
    left_sum = (gt_images[0].get_fdata()[:, :updated_second_dimension, :]).sum()/2
    right_sum = (gt_images[0].get_fdata()[:, updated_second_dimension:, :]).sum()/2

    # Define cropping functions
    crop = lambda x: x[:, :updated_second_dimension, :] if left_sum > right_sum else \
        x[:, updated_second_dimension:, :]

    # Crop ground truths
    gt_images_[0] = nib.Nifti1Image(
        dataobj=crop(gt_images[0].get_fdata()), affine=gt_images[0].affine, header=gt_images[0].header
    )

    # Crop input images
    for i, input_image in enumerate(input_images):
        input_images_[i] = nib.Nifti1Image(
            dataobj=crop(input_image.get_fdata()), affine=input_image.affine, header=input_image.header
        )

    # Return output
    return input_images_, gt_images_


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
