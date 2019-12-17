"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-15

**Project** : src

Class that implements SegmentationInputExpander
"""
from typing import Tuple
import numpy as np
from scipy.ndimage import rotate
import nibabel as nib

from data_science_framework.data_augmentation.segmentation_augmentation.SegmentationImageTransformation import \
    SegmentationImageTransformation


class SegmentationInputExpander(SegmentationImageTransformation):
    """
    Class that implements SegmentationInputExpander

    :param tile_shape_x: Dimension of the tile in the x axis
    :param tile_shape_y: Dimension of the tile in the y axis
    :param tile_shape_z: Dimension of the tile in the z axis
    """
    def __init__(
            self, tile_shape_x=16, tile_shape_y=16, tile_shape_z=16
    ):
        self.tile_shape_x = tile_shape_x
        self.tile_shape_y = tile_shape_y
        self.tile_shape_z = tile_shape_z
        self.tile_shape = (
            self.tile_shape_x, self.tile_shape_y, self.tile_shape_z
        )

    def get_transformation(self):
        """
        Get the transformation

        :return: Function that corresponds to the transformation
        """
        def transformation(image):
            # Compute dimension
            tile_compliant_shape = tuple(
                [
                    max(image_dim, tile_dim)
                    for image_dim, tile_dim in zip(
                    list(image.shape), list(self.tile_shape)
                    )
                ]

            )
            if image.shape == tile_compliant_shape:
                return image
            else:
                redimensionned_array = np.zeros(tile_compliant_shape)
                redimensionned_array[:image.shape[0], :image.shape[1], :image.shape[2]] = image.get_fdata()
                return nib.Nifti1Image(
                    dataobj=redimensionned_array,
                    affine=image.affine,
                    header=image.header
                )
        return transformation
