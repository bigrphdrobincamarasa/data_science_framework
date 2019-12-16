"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-09

**Project** : baseline_unet

#TODO: test random

**File that tests codes of data_manager module**
"""
import nibabel as nib
import numpy as np
import os
from data_science_framework.scripting.test_manager import set_test_folders
from data_science_framework.settings import RESSOURCES_ROOT, TEST_ROOT

from data_science_framework.pytorch_utils.data_manager import MODULE

from data_science_framework.pytorch_utils.data_manager.data_conversion import \
    convert_nifty_batch_to_torch_tensor

from data_science_framework.pytorch_utils.data_manager.SegmentationTransformation import \
    SegmentationTransformation

from data_science_framework.pytorch_utils.data_manager.SegmentationPatientTransformation import \
    SegmentationPatientTransformation

from data_science_framework.pytorch_utils.data_manager.SegmentationImageTransformation import \
    SegmentationImageTransformation

from data_science_framework.pytorch_utils.data_manager.SegmentationRotation import \
    SegmentationRotation

from data_science_framework.pytorch_utils.data_manager.SegmentationFlip import \
    SegmentationFlip

from data_science_framework.pytorch_utils.data_manager.SegmentationCropHalf import \
    SegmentationCropHalf

from data_science_framework.pytorch_utils.data_manager.SegmentationNormalization import \
    SegmentationNormalization

from data_science_framework.pytorch_utils.data_manager.SegmentationTiling import SegmentationTiling


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


def test_SegmentationTransformation() -> None:
    """
    Function that tests SegmentationTransformation

    :return: None
    """
    segmentation_transformation = SegmentationTransformation()
    try:
        segmentation_transformation.get_transformation()
        segmentation_transformation.transform_batch(None, None)
        segmentation_transformation.transform_patient(None, None)
    except Exception as e:
        assert False


def test_SegmentationPatientTransformation() -> None:
    """
    Function that tests SegmentationPatientTransformation

    :return: None
    """
    segmentation_patient_transformation = SegmentationPatientTransformation()
    segmentation_patient_transformation.transform_patient = lambda x, y: (x-y, x+y)
    output = segmentation_patient_transformation.transform_batch(
        [4, 5, 6], [0, 1, 2]
    )
    assert tuple(output[0]) == (4, 4, 4)
    assert tuple(output[1]) == (4, 6, 8)


def test_SegmentationImageTransformation() -> None:
    """
    Function that tests SegmentationImageTransformation

    :return: None
    """
    segmentation_image_transformaton = SegmentationImageTransformation()
    segmentation_image_transformaton.get_transformation = lambda: lambda x: x**2
    output = segmentation_image_transformaton.transform_patient(
        [4, 5, 6], [0, 1, 2, 3]
    )
    assert tuple(output[0]) == (16, 25, 36)
    assert tuple(output[1]) == (0, 1, 4, 9)


@set_test_folders(
    output_root=TEST_ROOT,
    ressources_root=RESSOURCES_ROOT,
    current_module=MODULE
)
def test_SegmentationRotation(ressources_structure: dict, output_folder: str) -> None:
    """
    Function that tests SegmentationRotation

    :param output_folder: Path to the output folder
    :param ressources_structure: Dictionnary containing the path and objects contained in the ressource folder
    :return: None
    """
    template_image = nib.load(
        ressources_structure['patient_0']['image_3.nii.gz']['path']
    )
    input = nib.Nifti1Image(
        dataobj=np.arange(50 * 60 * 70).reshape(50, 60, 70),
        affine=template_image.affine,
        header=template_image.header
    )
    nib.save(
        input, os.path.join(output_folder, 'input.nii.gz')
    )
    for angle_x, angle_y, angle_z in [(45, 0, 0), (0, 45, 0), (0, 0, 45)]:
        segmentation_rotation = SegmentationRotation(
            angle_x=angle_x,
            angle_y=angle_y,
            angle_z=angle_z
        )
        output = segmentation_rotation.get_transformation()(input)
        nib.save(
            output,
            os.path.join(output_folder, 'x_{}_y_{}_z_{}.nii.gz'.format(angle_x, angle_y, angle_z))
        )
        assert input.shape == output.shape


@set_test_folders(
    output_root=TEST_ROOT,
    ressources_root=RESSOURCES_ROOT,
    current_module=MODULE
)
def test_SegmentationFlip(ressources_structure: dict, output_folder: str,) -> None:
    """
    Function that tests SegmentationFlip

    :param output_folder: Path to the output folder
    :param ressources_structure: Dictionnary containing the path and objects contained in the ressource folder
    :return: None
    """
    template_image = nib.load(
        ressources_structure['patient_0']['image_3.nii.gz']['path']
    )
    input = nib.Nifti1Image(
        dataobj=np.arange(50 * 60 * 70).reshape(50, 60, 70),
        affine=template_image.affine,
        header=template_image.header
    )
    nib.save(
        input, os.path.join(output_folder, 'input.nii.gz')
    )
    for flip_x, flip_y, flip_z in [(True, False, False), (False, True, False), (False, False, True)]:
        segmentation_flip = SegmentationFlip(
            flip_x=flip_x,
            flip_y=flip_y,
            flip_z=flip_z
        )
        output = segmentation_flip.get_transformation()(input)
        nib.save(
            output,
            os.path.join(output_folder, 'x_{}_y_{}_z_{}.nii.gz'.format(flip_x, flip_y, flip_z))
        )
        assert input.shape == output.shape


@set_test_folders(
    output_root=TEST_ROOT,
    ressources_root=RESSOURCES_ROOT,
    current_module=MODULE
)
def test_SegmentationCropHalf(ressources_structure: dict, output_folder: str) -> None:
    """
    Function that tests SegmentationCropHalf

    :param output_folder: Path to the output folder
    :param ressources_structure: Dictionnary containing the path and objects contained in the ressource folder
    :return: None
    """
    # Generate data
    template_image = nib.load(
        ressources_structure['patient_0']['image_3.nii.gz']['path']
    )
    input_data = np.arange(50 * 60 * 70).reshape(50, 60, 70)
    input = [
        nib.Nifti1Image(
            dataobj=i * input_data,
            affine=template_image.affine,
            header=template_image.header
        ) for i in range(5)
    ]

    for i, input_ in enumerate(input):
        nib.save(
            input_, os.path.join(output_folder, 'input_{}.nii.gz'.format(i))
        )
    for side in ['left', 'right']:
        if side == 'left':
            gt = [
                nib.Nifti1Image(
                    dataobj=input_data > input_data.mean(),
                    affine=template_image.affine,
                    header=template_image.header
                ) for i in range(4)
            ]
        else:
            gt = [
                nib.Nifti1Image(
                    dataobj=input_data < input_data.mean(),
                    affine=template_image.affine,
                    header=template_image.header
                ) for i in range(4)
            ]

        segmentation_crop_half = SegmentationCropHalf()
        input_transformed, gt_transformed = segmentation_crop_half.transform_patient(input, gt)

        # Save data
        for i, input_transformed_ in enumerate(input_transformed):
            nib.save(
                input_transformed_, os.path.join(output_folder, 'input_transformed_{}_{}.nii.gz'.format(side, i))
            )
        for i, (gt_, gt_transformed_) in enumerate(zip(gt, gt_transformed)):
            nib.save(
                gt_, os.path.join(output_folder, 'gt_{}_{}.nii.gz'.format(side, i))
            )
            nib.save(
                gt_transformed_, os.path.join(output_folder, 'gt_transformed_{}_{}.nii.gz'.format(side, i))
            )
        assert len(gt) == len(gt_transformed)
        assert len(input) == len(input_transformed)
        assert input_transformed[0].shape == (25, 60, 70)
        assert gt_transformed[0].shape == (25, 60, 70)


@set_test_folders(
    output_root=TEST_ROOT,
    ressources_root=RESSOURCES_ROOT,
    current_module=MODULE
)
def test_SegmentationNormalization(ressources_structure: dict, output_folder: str) -> None:
    """
    Function that tests SegmentationNormalization

    :param output_folder: Path to the output folder
    :param ressources_structure: Dictionnary containing the path and objects contained in the ressource folder
    :return: None
    """
    # Generate data
    template_image = nib.load(
        ressources_structure['patient_0']['image_3.nii.gz']['path']
    )
    input_data = np.arange(50 * 60 * 70).reshape(50, 60, 70)
    input = [
        nib.Nifti1Image(
            dataobj=i * input_data,
            affine=template_image.affine,
            header=template_image.header
        ) for i in range(5)
    ]
    for i in range(5):
        nib.save(
            input[i], os.path.join(
                output_folder, 'input_{}.nii.gz'.format(i)
            )
        )
    # Get segmentation normalization
    segmentation_normalization = SegmentationNormalization()
    output = segmentation_normalization.transform_patient(input, [])

    assert len(output[1]) == 0
    for i, input_transformed in enumerate(output[0]):
        assert input_transformed.shape == input[0].shape
        nib.save(
            input_transformed,
            os.path.join(
                output_folder, 'input_transformed_{}.nii.gz'.format(i)
            )
        )

@set_test_folders(
    output_root=TEST_ROOT,
    ressources_root=RESSOURCES_ROOT,
    current_module=MODULE
)
def test_SegmentationTiling(ressources_structure: dict, output_folder: str) -> None:
    """
    Function that tests SegmentationTiling

    :param ressources_structure: Dictionnary containing the path and objects contained in the ressource folder
    :param output_folder: Path to the output folder
    :return: None
    """
    segmentation_tiling = SegmentationTiling(expansion_factor=3)

    # Test split dimension
    output = segmentation_tiling.split_dimension(image_dimension=128, tile_dimension=16)
    assert tuple(output) == (
        0, 5, 10, 16, 21, 26, 32, 37, 42, 48, 53, 58, 64, 69,
        74, 80, 85, 90, 96, 101, 106, 112
    )

    # Test compute grid
    output = segmentation_tiling.compute_grid(image_shape=(16, 33, 17))
    assert tuple(output) == (
        (0, 0, 0), (0, 0, 1), (0, 5, 0), (0, 5, 1),
        (0, 10, 0), (0, 10, 1), (0, 16, 0), (0, 16, 1),
        (0, 17, 0), (0, 17, 1)
    )

    # Test is background
    gt_coordinates = ((50, 100), (50, 100), (50, 100))
    image_coordinates_list = [
        (i, j, k)
        for i in [75, 33, 101]
        for j in [75, 33, 101]
        for k in [75, 33, 101]
    ]
    labels = [False] + 26 * [True]
    for image_coordinates, label in zip(image_coordinates_list, labels):
        output = segmentation_tiling.is_background(
            gt_coordinates=gt_coordinates,
            image_coordinates=image_coordinates
        )
        assert label == output

    # Test get_gt_coordinates
    gts = []
    for i in range(10, 30, 10):
        # Create array
        gt = np.zeros((100, 150, 200))
        gt[i:i+10, 2*i:2*i+20, 3*i:3*i+30] = 1

        # Create image
        image_gt = nib.Nifti1Image(
            gt,
            np.eye(4)
        )

        # Append image
        gts.append(image_gt)
    output = segmentation_tiling.get_gt_coordinates(gts)
    assert output == ((10, 29), (20, 59), (30, 89))

    # Test get_transformation
    array = np.arange(50 * 100 * 200).reshape(50, 100, 200)
    output = segmentation_tiling.get_transformation()(
        nib.Nifti1Image(
            dataobj=array, affine=np.eye(4)
        ), (10, 20, 30)
    )
    expected_value = array[10:26, 20:36, 30:46]
    assert expected_value.shape == output.shape
    assert tuple(list(expected_value.ravel())) == tuple(list(output.get_fdata().ravel()))

    # Test transform patient
    segmentation_tiling = SegmentationTiling(expansion_factor=2, shape_x=4, shape_y=4, shape_z=4)
    input = [
        nib.Nifti1Image(
            dataobj=np.zeros((10, 20, 30)),
            affine=np.eye(4)
        )
        for _ in range(3)
    ]
    gt = [
        nib.Nifti1Image(
            dataobj=np.arange(10 * 20 * 30).reshape(10, 20, 30),
            affine=np.eye(4)
        )
        for _ in range(4)
    ]
    input_, gt_ = segmentation_tiling.transform_patient(
        input=input,
        gt=gt
    )
    assert len(input_) == 504
    assert len(gt_) == 504
    assert len(input_[0]) == 3
    assert len(gt_[0]) == 4
    assert input_[0][0].shape == (4, 4, 4)
    assert gt_[0][0].shape == (4, 4, 4)
