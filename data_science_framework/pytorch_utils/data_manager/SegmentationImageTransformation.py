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

from data_science_framework.pytorch_utils.data_manager.SegmentationTransformation import \
    SegmentationTransformation


class SegmentationImageTransformation(SegmentationTransformation):
    """
    Class that implements SegmentationImageTransformation
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
        transformation = self.get_transformation
        for item in zip(input, gt):
            input_item_, output_item_ = transformation(*item)
            output[0].append(input_item_)
            output[1].append(output_item_)
        return output

    def get_transformation(self):
        """
        Get the transformation

        :return: Function that corresponds to the transformation
        """
        pass
