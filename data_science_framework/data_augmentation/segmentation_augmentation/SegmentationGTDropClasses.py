"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-15

**Project** : src

Class that implements SegmentationGTDropClasses
"""
from typing import Tuple

from data_science_framework.data_augmentation.segmentation_augmentation import \
        SegmentationGTExpander
import nibabel as nib


class SegmentationGTDropClasses(SegmentationGTExpander):
    """
    Class that implements SegmentationGTDropClasses

    :param dropped_classes: The classes dropped in the ground truth
    """
    def __init__(self, dropped_classes=[2, 4, 6, 8]):
        self.dropped_classes = dropped_classes

    def transform_patient(self, input, gt) -> Tuple:
        """
        Apply the transformation to the input and the ground truth that are patient formatted
        (nfeature, shape)

        Apply a transformation to the input and the ground truth

        :param input: List of input images
        :param gt: List of gt images
        :return: Tuple of transformed values
        """
        transformation = self.get_transformation()
        return input, transformation(gt)

    def get_transformation(self):
        """
        Get the transformation

        :return: Function that corresponds to the transformation
        """
        def transformation(gt):
            return [
                gt_ for i, gt_ in enumerate(gt)
                if not i in self.dropped_classes
            ]
        return transformation

