"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-09

**Project** : baseline_unet

**File that tests codes of data_manager module**
#TODO: test random
"""
import nibabel as nib
import numpy as np
from data_science_framework.scripting.test_manager import set_test_folders
from data_science_framework.settings import RESSOURCES_ROOT

from data_science_framework.pytorch_utils.data_manager import MODULE

from data_science_framework.pytorch_utils.data_manager.data_transformation import rotate_images, flip_images, \
    crop_half_images, apply_transformations_to_batch
from scipy.ndimage import rotate

from data_science_framework.pytorch_utils.data_manager.data_conversion import \
    convert_nifty_batch_to_torch_tensor


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
    # Initialize tests variables
    transformations = [
        lambda x, y: rotate_images(x, y, angle_x=10), lambda x, y: rotate_images(x, y, angle_x=30)
    ]
    input_batch = [
        [
            nib.load(ressources_structure['input_{}.nii'.format(3*i + j + 1)]['path'])
            for i in range(2)
        ]
        for j in range(3)
    ]
    gt_batch = [
        [nib.load(ressources_structure['input_{}.nii'.format(3 * 2 + j + 1)]['path'])]
        for j in range(3)
    ]


    # Apply function
    input_batch_, gt_batch_ = apply_transformations_to_batch(
        input_batch=input_batch, gt_batch=gt_batch,
        transformations=transformations
    )

    # Test number of elements in the batch
    assert len(input_batch_) == len(input_batch)
    assert len(gt_batch_) == len(gt_batch)

    # Test number of features in the element
    assert len(input_batch_[0]) == len(input_batch[0])
    assert len(gt_batch_[0]) == len(gt_batch[0])

    # Test transformation
    expected_input_images, expected_output_images = rotate_images(
        input_images=input_batch[0], gt_images=gt_batch[0],
        angle_x=40
    )
    assert np.array(
        (input_batch_[0][0].get_fdata() - expected_input_images[0].get_fdata())**2
    ).sum() < 1


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


@set_test_folders(
    ressources_root=RESSOURCES_ROOT,
    current_module=MODULE
)
def test_convert_nifty_batch_to_torch_tensor(ressources_structure: dict) -> None:
    """
    Function that tests convert_nifty_batch_to_torch_tensor

    #TODO test gpu and cpu options
    :param ressources_structure: Dictionnary containing the path and objects contained in the ressource folder
    :return: None
    """
    input = []
    for i, patient_images in enumerate(ressources_structure.values()):
        if not 'path' in patient_images.keys():
            input.append([nib.load(patient_image['path']) for patient_image in patient_images.values()])

    torch_output = convert_nifty_batch_to_torch_tensor(input, 'cpu')

    array = torch_output.detach().numpy()

    assert array.shape == (3, 5, 4, 4)
    assert array.sum() == 300

