"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-15

**Project** : src

Class that implements SegmentationCropHalf
"""
from typing import Tuple

from data_science_framework.data_augmentation.segmentation_augmentation.SegmentationPatientTransformation import \
    SegmentationPatientTransformation
import numpy as np
import nibabel as nib


class SegmentationCropHalf(SegmentationPatientTransformation):
    """
    Class that implements SegmentationCropHalf
    """

    def transform_patient(self, input: list, gt: list) -> Tuple:
        """
        Apply the transformation to the input and the ground truth that are patient formatted
        (nfeature, shape)

        Apply a transformation to the input and the ground truth

        :param input: List of input images
        :param gt: List of gt images
        :return: Tuple of transformed values
        """
        output = ([], [])
        transformation = self.get_transformation(gt)

        # Apply transformation to input
        for input_item_ in input:
            input_item_ = transformation(input_item_)
            output[0].append(input_item_)

        # Apply transformation to gt
        for gt_item_ in gt:
            gt_item_ = transformation(gt_item_)
            output[1].append(gt_item_)
        return output

    def get_transformation(self, gt: list):
        """
        Get the transformation

        :param gt: List of gt classes
        :return: Function that corresponds to the transformation
        """
        # Initialize values
        gt_mask = np.zeros(gt[0].shape)
        gt_x_half_shape = int(gt_mask.shape[0]/2)

        # Create gt mask
        for gt_item in gt[1:]:
            gt_mask += gt_item.get_fdata()

        if gt_mask[:gt_x_half_shape, :, :].sum() < gt_mask[gt_x_half_shape:, :, :].sum():
            crop = lambda x: x[gt_x_half_shape:, :, :]
        else:
            crop = lambda x: x[:gt_x_half_shape, :, :]

        return lambda x: nib.Nifti1Image(
            crop(x.get_fdata()), affine=x.affine,
            header=x.header
        )
