"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-15

**Project** : src

Class that implements SegmentationROISelector
"""
from typing import Tuple

from data_science_framework.data_augmentation.segmentation_augmentation import SegmentationPatientTransformation

import nibabel as nib
import numpy as np


class SegmentationROISelector(SegmentationPatientTransformation):
    """
    Class that implements SegmentationROISelector

    :param shape_x: Tile dimension in the x dimension
    :param shape_y: Tile dimension in the y dimension
    :param shape_z: Tile dimension in the z dimension
    :param centered: True if the groundtruth is centered
    """

    def __init__(
            self, shape_x: int = 16, shape_y: int = 16,
            shape_z: int = 16, centered: bool = False
    ):
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.shape_z = shape_z
        self.centered = centered
        self.tile_shape = (shape_x, shape_y, shape_z)

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
        input_, gt_ = [], []

        # Get transfomation
        tile = self.get_transformation(gt)

        # Transform input
        for input_item in input:
            input_.append(tile(input_item))

        # Transform gt
        for gt_item in gt:
            gt_.append(tile(gt_item))

        return (input_, gt_)

    def get_transformation(self, gt: tuple):
        """
        Get the transformation

        :param gt: List of groundtruth images
        :return: Function that corresponds to the transformation
        """
        # Get the ground truth center
        gt_centers = self.get_gt_centers(gt)
        if self.centered:
            tile_start = tuple(
                [
                    max(
                        0,
                        min(
                            int(gt_centers[i] - self.tile_shape[i] / 2),
                            gt[0].shape[i] - self.tile_shape[i],
                        )
                    )
                    for i in range(len(list(gt[0].shape)))
                ]
            )
        else:
            tile_start = tuple(
                [
                    np.random.randint(
                        low=max(0, gt_centers[i] - self.tile_shape[i]),
                        high=min(
                            gt_centers[i], gt[0].shape[i] - self.tile_shape[i]
                        )+1
                    )
                    for i in range(len(list(gt[0].shape)))
                ]
            )

        # Tile the image
        tile = lambda image_array, tile_start: image_array[
            tile_start[0]:tile_start[0] + self.tile_shape[0],
            tile_start[1]:tile_start[1] + self.tile_shape[1],
            tile_start[2]:tile_start[2] + self.tile_shape[2],
        ]

        # Apply the full transformation
        transform = lambda image: nib.Nifti1Image(
            dataobj=tile(
                image.get_fdata(),
                tile_start
            ),
            affine=image.affine,
            header=image.header
        )
        return transform

    def get_gt_centers(self, gt: list) -> tuple:
        """
        Methods that computes the center of the gt image

        :param gt: List of groundtruth images
        :return: tuple of tuple of min and max coordinates of non null pixel of gt in each direction
        """
        # Initialize values
        gt_mask = np.zeros(gt[0].shape)

        # Create gt mask
        # The first channel is for the background
        for gt_item in gt[1:]:
            gt_mask += gt_item.get_fdata()

        # Get non zeros indices
        non_zeros_indices = gt_mask.nonzero()
        return tuple(
            [
                non_zeros_indices_dim.mean()
                for non_zeros_indices_dim in list(non_zeros_indices)
            ]
        )
