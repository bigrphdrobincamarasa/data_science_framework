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

from data_science_framework.data_augmentation.segmentation_augmentation.SegmentationTransformation import \
    SegmentationTransformation

import nibabel as nib
import numpy as np


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
        transformed_input, transformed_gt = [], []
        for input_, gt_ in zip(input, gt):
            transformed_input_, transformed_gt_ = self.transform_patient(input_, gt_)
            transformed_input += transformed_input_
            transformed_gt += transformed_gt_
        return transformed_input, transformed_gt

    def transform_patient(self, input, gt) -> Tuple:
        """
        Apply the transformation to the input and the ground truth that are patient formatted
        (nfeature, shape)

        Apply a transformation to the input and the ground truth

        :param input: List of input images
        :param gt: List of gt images
        :return: Tuple of transformed values
        """
        # Initialise output
        output = ([], [])

        # Get grid
        grid = self.compute_grid(input[0].shape)

        # Get gt coordinates
        gt_coordinates = self.get_gt_coordinates(gt)

        # Get transformation
        tile = self.get_transformation()

        # Tile generation loop
        for image_coordinates in grid:
            if not self.is_background(
                    gt_coordinates=gt_coordinates,
                    image_coordinates=image_coordinates
            ) or np.random.rand() < self.background_rate:
                # Initialize tiles input and gt lists
                tile_input_ = []
                tile_gt_ = []
                for input_ in input:
                    tile_input_.append(tile(input_, image_coordinates))
                for gt_ in gt:
                    tile_gt_.append(tile(gt_, image_coordinates))
                # Add tile
                output[0].append(tile_input_.copy())
                output[1].append(tile_gt_.copy())
        return output



    def get_transformation(self):
        """
        Get the transformation

        :return: Function that corresponds to the transformation
        """
        tile = lambda image_array, coordinates: image_array[
                                                coordinates[0]:coordinates[0] + self.tile_shape[0],
                                                coordinates[1]:coordinates[1] + self.tile_shape[1],
                                                coordinates[2]:coordinates[2] + self.tile_shape[2],
                                                ]
        transform = lambda image, coordinates: nib.Nifti1Image(
            dataobj=tile(image.get_fdata(), coordinates),
            affine=image.affine,
            header=image.header
        )
        return transform

    def get_gt_coordinates(self, gt: list) -> tuple:
        """
        Method that computes the background box of gt images

        :param gt: List of groundtruth images
        :return: tuple of tuple of min and max coordinates of non null pixel of gt in each direction
        """
        # Initialize values
        gt_mask = np.zeros(gt[0].shape)

        # Create gt mask
        for gt_item in gt:
            gt_mask += gt_item.get_fdata()

        # Get non zeros indices
        non_zeros_indices = gt_mask.nonzero()
        return tuple(
            [
                (non_zeros_indices_dim.min(), non_zeros_indices_dim.max())
                for non_zeros_indices_dim in list(non_zeros_indices)
            ]
        )

    def is_background(self, gt_coordinates: tuple, image_coordinates: tuple) -> bool:
        """
        Method that states if a tile is part of backgroud

        :param gt_coordinates: Tuple of Tuple of min and max gt coordinates in each direction
        :param image_coordinates: Tile coordinates of image
        :return: Boolean true if the image is part of background
        """
        for (gt_min_coor, gt_max_coor), image_coor, tile_dim in zip(
                list(gt_coordinates), image_coordinates, list(self.tile_shape)
        ):
            if image_coor > gt_max_coor or image_coor + tile_dim < gt_min_coor:
                return True
        return False

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
