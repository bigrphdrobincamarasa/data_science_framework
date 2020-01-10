"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-15

**Project** : src

Class that implements SegmentationGTExpander
"""
from typing import Tuple

from data_science_framework.data_augmentation.segmentation_augmentation.SegmentationPatientTransformation import \
    SegmentationPatientTransformation

import nibabel as nib


class SegmentationGTExpander(SegmentationPatientTransformation):
    """
    Class that implements SegmentationGTExpander

    :param nb_classes: The number of classes of the problem
    """
    def __init__(self, nb_classes):
        self.nb_classes = nb_classes

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
        return input, transformation(gt[0])

    def get_transformation(self):
        """
        Get the transformation

        :return: Function that corresponds to the transformation
        """
        def transformation(gt):
            gt_array = gt.get_fdata()
            return [
                nib.Nifti1Image(
                    dataobj=(gt_array == i),
                    affine=gt.affine,
                    header=gt.header
                )
                for i in range(self.nb_classes)
            ]
        return transformation
