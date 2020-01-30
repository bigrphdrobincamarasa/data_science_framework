"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-09

**Project** : baseline_unet

#TODO: test random

**File that tests codes of data_augmentation module**
"""
import nibabel as nib
import numpy as np
import os
from data_science_framework.scripting.test_manager import set_test_folders
from data_science_framework.settings import RESSOURCES_ROOT, TEST_ROOT

from data_science_framework.data_augmentation.segmentation_augmentation import MODULE

from data_science_framework.data_augmentation.segmentation_augmentation import SegmentationTransformation, \
SegmentationPatientTransformation, SegmentationImageTransformation, SegmentationRotation, SegmentationFlip, \
SegmentationCropHalf, SegmentationNormalization, SegmentationTiling, SegmentationGTExpander, SegmentationInputExpander, \
SegmentationROISelector, SegmentationToTorch, SegmentationGTDropClasses


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
            dataobj=np.zeros((8, 12, 20)),
            affine=np.eye(4)
        )
        for _ in range(3)
    ]
    gt = [
        nib.Nifti1Image(
            dataobj=np.arange(8 * 12 * 20).reshape(8, 12, 20),
            affine=np.eye(4)
        )
        for _ in range(4)
    ]
    input_, gt_ = segmentation_tiling.transform_patient(
        input=input,
        gt=gt
    )
    assert len(input_) == 135
    assert len(gt_) == 135
    assert len(input_[0]) == 3
    assert len(gt_[0]) == 4
    assert input_[0][0].shape == (4, 4, 4)
    assert gt_[0][0].shape == (4, 4, 4)

    # Test transform batch
    input = [
        [
            nib.Nifti1Image(
                dataobj=np.zeros((8, 12, 20)),
                affine=np.eye(4)
            )
            for _ in range(3)
        ]
        for _ in range(5)
    ]
    gt = [
        [
            nib.Nifti1Image(
                dataobj=np.arange(8 * 12 * 20).reshape(8, 12, 20),
                affine=np.eye(4)
            )
            for _ in range(4)
        ]
        for _ in range(5)
    ]
    input_, gt_ = segmentation_tiling.transform_batch(
        input=input, gt=gt
    )
    assert len(input_) == 675
    assert len(gt_) == 675
    assert len(input_[0]) == 3
    assert len(gt_[0]) == 4
    assert input_[0][0].shape == (4, 4, 4)
    assert gt_[0][0].shape == (4, 4, 4)
    assert True


def test_SegmentationGTExpander() -> None:
    """
    Function that tests SegmentationGTExpander

    :return: None
    """
    # Test initialisation
    segmentation_gt_expander = SegmentationGTExpander(10)
    assert segmentation_gt_expander.nb_classes == 10

    # Test patient transformation
    segmentation_gt_expander = SegmentationGTExpander(10)

    # Create a vanilla transformation
    segmentation_gt_expander.get_transformation = lambda: lambda y: [1 for i in range(y)]
    input_, gt_ = segmentation_gt_expander.transform_patient('input', [5])

    assert tuple(gt_) == (1, 1, 1, 1, 1)
    assert input_ == input_

    # Test get_transformation
    segmentation_gt_expander = SegmentationGTExpander(10)
    transformation = segmentation_gt_expander.get_transformation()
    gt = nib.Nifti1Image(
        dataobj=np.arange(3 * 4 * 5).reshape(3, 4, 5) % 3,
        affine=np.eye(4)
    )
    gt_ = transformation(gt)
    assert len(gt_) == 10
    assert gt_[0].get_fdata().sum() != 0 and gt_[1].get_fdata().sum() == gt_[0].get_fdata().sum() \
        and gt_[2].get_fdata().sum() == gt_[0].get_fdata().sum()
    for i in range(3, 10):
        assert gt_[i].get_fdata().sum() == 0


def test_SegmentationExpandInput() -> None:
    """
    Function that tests SegmentationExpandInput

    :return: None
    """
    # Test initialisation
    segmentation_input_expander = SegmentationInputExpander()
    input = nib.Nifti1Image(
        dataobj=np.ones((16, 15, 17)),
        affine=np.eye(4)
    )

    # Get transformation
    transformation = segmentation_input_expander.get_transformation()

    # Compute output
    output = transformation(input)

    assert output.shape == (16, 16, 17)
    assert output.get_fdata().sum() == 16*15*17


@set_test_folders(
    output_root=TEST_ROOT,
    ressources_root=RESSOURCES_ROOT,
    current_module=MODULE
)
def test_SegmentationROISelector(ressources_structure: dict, output_folder: str) -> None:
    """
    Function that tests SegmentationROISelector

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
    gt = [
        nib.Nifti1Image(
            dataobj=input_data > input_data.mean(),
            affine=template_image.affine,
            header=template_image.header
        ) for i in range(4)
    ]

    segmentation_roi_selector = SegmentationROISelector(
            shape_x=16,
            shape_y=16,
            shape_z=16,
            centered=True
    )

    for i, input_ in enumerate(input):
        nib.save(
            input_, os.path.join(output_folder, 'input_{}.nii.gz'.format(i))
        )

    input_transformed, gt_transformed = segmentation_roi_selector\
            .transform_patient(input, gt)

    # Save data
    for i, (input_, input_transformed_) in enumerate(
            zip(input, input_transformed)
        ):
        nib.save(
            input_,
            os.path.join(output_folder, 'input_{}.nii.gz'.format(i))
        )
        nib.save(
            input_transformed_,
            os.path.join(output_folder, 'input_transformed_{}.nii.gz'.format(i))
        )

    for i, (gt_, gt_transformed_) in enumerate(zip(gt, gt_transformed)):
        nib.save(
            gt_,
            os.path.join(output_folder, 'gt_{}.nii.gz'.format(i))
        )
        nib.save(
            gt_transformed_,
            os.path.join(
                output_folder,
                'gt_transformed_{}.nii.gz'.format(i)
            )
        )
    assert len(gt) == len(gt_transformed)
    assert len(input) == len(input_transformed)
    assert input_transformed[0].shape == (16, 16, 16)
    assert gt_transformed[0].shape == (16, 16, 16)

@set_test_folders(
    output_root=TEST_ROOT,
    ressources_root=RESSOURCES_ROOT,
    current_module=MODULE
)
def test_SegmentationToTorch(ressources_structure: dict, output_folder: str) -> None:
    """
    Function that tests SegmentationToTorch

    :param output_folder: Path to the output folder
    :param ressources_structure: Dictionnary containing the path and objects contained in the ressource folder
    :return: None
    """
    # Get template image
    template_image = nib.load(
        ressources_structure['patient_0']['image_3.nii.gz']['path']
    )

    # Get transformation
    segmentation_to_torch = SegmentationToTorch(
        device='cpu'
    )

    # Generate data
    input = [
        [
            nib.Nifti1Image(
            dataobj=np.arange(3 * 4 * 5).reshape(3, 4, 5),
            affine=template_image.affine,
            header=template_image.header
            ) for i in range(5)
        ] for j in range(2)
    ]
    gt = [
        [
            nib.Nifti1Image(
            dataobj=np.arange(3 * 4 * 5).reshape(3, 4, 5),
            affine=template_image.affine,
            header=template_image.header
            ) for i in range(4)
        ] for j in range(2)
    ]

    # Perform transformation
    input_transformed, gt_transformed = segmentation_to_torch.transform_batch(
        input, gt
    )

    assert input_transformed.shape == (2, 5, 3, 4, 5)
    assert gt_transformed.shape == (2, 4, 3, 4, 5)


def test_SegmentationGTDropClasses() -> None:
    """
    Function that tests SegmentationGTDropClasses

    :return: None
    """
    # Test initialisation
    segmentation_gt_drop_classes = SegmentationGTDropClasses()
    assert tuple(segmentation_gt_drop_classes.dropped_classes) == tuple(
        list(range(2, 10, 2))
    )

    # Test patient transformation
    segmentation_gt_drop_classes = SegmentationGTDropClasses()
    segmentation_gt_drop_classes.get_transformation = lambda : lambda x:x**2
    input, gt = 3, 4
    input_, gt_ = segmentation_gt_drop_classes.transform_patient(input, gt)
    assert (input_, gt_) == (3, 16)

    # Test transformation
    segmentation_gt_drop_classes = SegmentationGTDropClasses()
    transformation = segmentation_gt_drop_classes.get_transformation()
    gt_ = transformation(list(range(10)))
    assert tuple(gt_) == (0, 1, 3, 5, 7, 9)

