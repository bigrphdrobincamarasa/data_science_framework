"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-15

**Project** : src

Class that implements SegmentationNormalization
"""
from typing import Tuple

from data_science_framework.data_augmentation.segmentation_augmentation.SegmentationPatientTransformation import \
    SegmentationPatientTransformation
import numpy as np
import nibabel as nib


class SegmentationNormalization(SegmentationPatientTransformation):
    """
    Class that implements SegmentationNormalization

    :param min_percentile: Percentile in the distribution of the min value
    :param max_percentile: Percentile in the distribution of the max value
    """

    def __init__(self, min_percentile: float = 0.05, max_percentile: float = 0.95):
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    def transform_patient(self, input: list, gt: list) -> Tuple:
        """
        Apply the transformation to the input that is patient formatted
        (nfeature, shape)

        :param input: List of input images
        :param gt: List of gt images
        :return: Tuple of transformed values
        """
        output = ([], gt.copy())
        transformation = self.get_transformation()

        # Apply transformation to input
        for input_item_ in input:
            input_item_ = transformation(input_item_)
            output[0].append(input_item_)

        return output

    def get_transformation(self):
        """
        Get the transformation

        :return: Function that corresponds to the transformation
        """
        def normalization(input):
            # Sort image values
            input_array = np.array(input.get_fdata(), dtype=float)
            sorted_input_array = np.sort(input.get_fdata().ravel())

            # Get percentiles
            min_values_ = sorted_input_array[
                int(self.min_percentile * sorted_input_array.shape[0])
            ]
            max_values_ = sorted_input_array[
                int(self.max_percentile * sorted_input_array.shape[0])
            ]

            # Process array
            input_array = (input_array - min_values_) / (max_values_ - min_values_)

            return nib.Nifti1Image(
                dataobj=input_array, affine=input.affine, header=input.header
            )
        return normalization

