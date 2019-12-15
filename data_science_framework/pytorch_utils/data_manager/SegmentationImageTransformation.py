"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-15

**Project** : src

Class that implements SegmentationTransformation
"""
from typing import Tuple

from data_science_framework.pytorch_utils.data_manager.SegmentationPatientTransformation import \
    SegmentationPatientTransformation


class SegmentationImageTransformation(SegmentationPatientTransformation):
    """
    Class that implements SegmentationImageTransformation

    :param angle_x: If random is enabled it corresponds to the max rotation angle around x axis otherwise it corresponds to the value of the rotation angle
    :param angle_y: If random is enabled it corresponds to the max rotation angle around y axis otherwise it corresponds to the value of the rotation angle
    :param angle_z: If random is enabled it corresponds to the max rotation angle around z axis otherwise it corresponds to the value of the rotation angle
    :param random: True if randomness is enabled. In this case, the angle of rotation follows a uniforme distribution between -angle and angle for each direction
    """

    def transform_patient(self, input, gt) -> Tuple:
        """
        Apply the transformation to the input and the ground truth that are patient formatted
        (nfeature, shape)

        Apply a transformation to the input and the ground truth

        :param input: List of input images
        :param gt: List of gt images
        :return: Tuple of transformed values
        """
        output = ([], [])
        transformation = self.get_transformation()

        # Apply transformation to input
        for input_item_ in input:
            input_item_ = transformation(input_item_)
            output[0].append(input_item_)

        # Apply transformation to gt
        for gt_item_ in gt:
            gt_item_ = transformation(gt_item_)
            output[1].append(gt_item_)
        return output

    def get_transformation(self):
        """
        Get the transformation

        :return: Function that corresponds to the transformation
        """
        pass

