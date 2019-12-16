"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-15

**Project** : src

Class that implements SegmentationTiling
"""
from typing import Tuple

from data_science_framework.pytorch_utils.data_manager.SegmentationTransformation import \
    SegmentationTransformation

import nibabel as nib


class SegmentationTiling(SegmentationTransformation):
    """
    Class that implements SegmentationTiling

    :param shape_x: Tile dimension in the x dimension
    :param shape_y: Tile dimension in the y dimension
    :param shape_z: Tile dimension in the z dimension
    :param expansion_factor: Number of grids in the made in the image
    :param background_rate: Pourcentage of background conserved
    """

    def __init__(
            self, shape_x=16, shape_y=16, shape_z=16,
            expansion_factor: int = 2,
            background_rate: float = 0.5
    ):
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.shape_z = shape_z
        self.expansion_factor = expansion_factor
        self.background_rate = background_rate
        self.tile_shape = (shape_x, shape_y, shape_z)

    def transform_batch(self, input, gt) -> Tuple:
        """
        Apply the transformation to the input and the ground truth that are batch formatted
        (batch_size, nfeature, shape)

        Apply a transformation to the input and the ground truth

        :param input: List of patient input formatted items
        :param gt: List of patient gt formatted items
        :return: Tuple of transformed values
        """
        pass

    def transform_patient(self, input, gt) -> Tuple:
        """
        Apply the transformation to the input and the ground truth that are patient formatted
        (nfeature, shape)

        Apply a transformation to the input and the ground truth

        :param input: List of input images
        :param gt: List of gt images
        :return: Tuple of transformed values
        """
        pass

    def get_transformation(self):
        """
        Get the transformation

        :return: Function that corresponds to the transformation
        """
        pass

    def compute_grid(self, image_shape: tuple) -> list:
        """
        Method that computes tiling grid from image shape

        :param image_shape: Shape of the image considered
        :return:
        """
        # Get grid coordinates
        grid_coordinates = []
        for image_dimension, tile_dimension in zip(
            list(image_shape), list(self.tile_shape)
        ):
            grid_coordinates.append(
                self.split_dimension(image_dimension, tile_dimension)
            )

        # Transform grid coordinates in usable output
        output = []
        for i in grid_coordinates[0]:
            for j in grid_coordinates[1]:
                for k in grid_coordinates[2]:
                    output.append((i, j, k))
        return output

    def split_dimension(
            self, image_dimension: int, tile_dimension: int
    ) -> list:
        """
        Method that splits dimension according to expansion
        factor and tiling shape

        :param image_dimension: Shape of the image in a considered dimension
        :param tile_dimension: Shape of the image in a considered dimension
        :return: List of the index that corresponds to split points in the dimension
        """
        output = []
        for k in range(self.expansion_factor):
            output += list(
                range(
                    int(k * tile_dimension / self.expansion_factor),
                    image_dimension - tile_dimension,
                    tile_dimension
                )
            )
        output += [image_dimension - tile_dimension]
        output.sort()
        return list(set(output))


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

