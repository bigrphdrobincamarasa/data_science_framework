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


class SegmentationTransformation:
    """
    Class that implements SegmentationTransformation
    """
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
