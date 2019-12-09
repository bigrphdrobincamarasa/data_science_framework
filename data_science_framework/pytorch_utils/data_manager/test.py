"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-09

**Project** : baseline_unet

**File that tests codes of data_manager module**
"""
import nibabel as nib
import numpy as np
from data_science_framework.scripting.test_manager import set_test_folders
from data_science_framework.settings import RESSOURCES_ROOT

from data_science_framework.pytorch_utils.data_manager import MODULE

from data_science_framework.pytorch_utils.data_manager.data_transformation import rotate_images, flip_images, \
    crop_half_images


@set_test_folders(
    ressources_root=RESSOURCES_ROOT,
    current_module=MODULE
)
def test_apply_transformations_to_batch(ressources_structure: dict) -> None:
    """
    Function that tests apply_transformations_to_batch

    :param ressources_structure: Dictionnary containing the path and objects contained in the ressource folder
    :return: None
    """
    pass


@set_test_folders(
    ressources_root=RESSOURCES_ROOT,
    current_module=MODULE
)
def test_rotate_images(ressources_structure: dict) -> None:
    """
    Function that tests rotate_images

    :param ressources_structure: Dictionnary containing the path and objects contained in the ressource folder
    :return: None
    """
    input_images = []
    gt_images = []

    expected_rotated_input_images = []
    expected_rotated_gt_images = []

    # Get inputs
    for i in range(1, 6):
        input_images.append(
            nib.load(
                ressources_structure['input_{}.nii'.format(i)]['path']
            )
        )
        expected_rotated_input_images.append(
            nib.load(
                ressources_structure['rot_45_90_180_{}.nii'.format(i)]['path']
            )
        )

    # Get gt
    for i in range(6, 10):
        gt_images.append(
            nib.load(
                ressources_structure['input_{}.nii'.format(i)]['path']
            )
        )
        expected_rotated_gt_images.append(
            nib.load(
                ressources_structure['rot_45_90_180_{}.nii'.format(i)]['path']
            )
        )

    # Obtain rotated images
    rotated_input_images, rotated_gt_images = rotate_images(
        input_images=input_images, gt_images=gt_images,
        angle_x=45, angle_y=90, angle_z=180
    )

    # Test rotation
    for expected_, output_ in zip(expected_rotated_input_images, rotated_input_images):
        assert ((expected_.get_fdata() - output_.get_fdata()) ** 2).sum() < 0.01

    for expected_, output_ in zip(expected_rotated_gt_images, rotated_gt_images):
        assert ((expected_.get_fdata() - output_.get_fdata()) ** 2).sum() < 0.01


@set_test_folders(
    ressources_root=RESSOURCES_ROOT,
    current_module=MODULE
)
def test_flip_image(ressources_structure: dict) -> None:
    """
    Function that tests

    :param ressources_structure: Dictionnary containing the path and objects contained in the ressource folder
    :return: None
    """
    input_images = []
    gt_images = []

    expected_flipped_input_images = []
    expected_flipped_gt_images = []

    for axe, kwargs in [
        ('x', {'flip_x': True, 'flip_y': False, 'flip_z': False}),
        ('y', {'flip_x': False, 'flip_y': True, 'flip_z': False}),
        ('z', {'flip_x': False, 'flip_y': False, 'flip_z': True})
    ]:
        # Get inputs
        for i in range(1, 6):
            input_images.append(
                nib.load(
                    ressources_structure['input_{}.nii'.format(i)]['path']
                )
            )
            expected_flipped_input_images.append(
                nib.load(
                    ressources_structure['flip_{}_{}.nii'.format(axe, i)]['path']
                )
            )

        # Get gt
        for i in range(6, 10):
            gt_images.append(
                nib.load(
                    ressources_structure['input_{}.nii'.format(i)]['path']
                )
            )
            expected_flipped_gt_images.append(
                nib.load(
                    ressources_structure['flip_{}_{}.nii'.format(axe, i)]['path']
                )
            )

        # Obtain flipped images
        flipped_input_images, flipped_gt_images = flip_images(
            input_images=input_images, gt_images=gt_images, **kwargs
        )

        # Test flip
        for expected_, output_ in zip(expected_flipped_input_images, flipped_input_images):
            assert ((expected_.get_fdata() - output_.get_fdata()) ** 2).sum() < 10

        for expected_, output_ in zip(expected_flipped_gt_images, flipped_gt_images):
            assert ((expected_.get_fdata() - output_.get_fdata()) ** 2).sum() < 10


@set_test_folders(
    ressources_root=RESSOURCES_ROOT,
    current_module=MODULE
)
def test_crop_half_images(ressources_structure: dict) -> None:
    """
    Function that tests crop_half_images

    :param ressources_structure: Dictionnary containing the path and objects contained in the ressource folder
    :return: None
    """
    input_images = []
    gt_images = []

    for side in ['left', 'right']:

        # Get inputs
        for i in range(1, 10):
            input_images.append(
                nib.load(
                    ressources_structure['input_{}.nii'.format(i)]['path']
                )
            )

        # Get ground truth
        gt_images.append(
            nib.load(
                ressources_structure['{}_gt_1.nii'.format(side)]['path']
            )
        )

        # Obtain flipped images
        cropped_input_images, cropped_gt_images = crop_half_images(
            input_images=input_images, gt_images=gt_images
        )

        # Test gt shapes
        assert cropped_gt_images[0].shape[1] == int(gt_images[0].shape[1] / 2) + 1
        assert cropped_gt_images[0].shape[0] == gt_images[0].shape[0]
        assert cropped_gt_images[0].shape[2] == gt_images[0].shape[2]

        # Test gt non nullity of cropped gt
        assert cropped_gt_images[0].get_fdata().sum() != 0

        for input_image, cropped_input_image in zip(input_images, cropped_input_images):
            # Test image shape
            assert cropped_input_image.shape[1] == int(input_image.shape[1]/2) + 1
            assert cropped_input_image.shape[0] == input_image.shape[0]
            assert cropped_input_image.shape[2] == input_image.shape[2]

            # Test matching between input and gt
            assert (
                (cropped_input_image.get_fdata() * cropped_gt_images[0].get_fdata()).sum() - \
                (input_image.get_fdata() * gt_images[0].get_fdata()).sum()
            ).sum() < 0.05

