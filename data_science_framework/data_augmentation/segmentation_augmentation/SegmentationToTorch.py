"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-15

**Project** : src

Class that implements SegmentationToTorch
"""
from typing import Tuple
import numpy as np
import torch

from data_science_framework.data_augmentation.segmentation_augmentation.SegmentationTransformation import \
    SegmentationTransformation


class SegmentationToTorch(SegmentationTransformation):
    """
    Class that implements SegmentationToTorch

    :param device: Device used by torch
    """
    def __init__(self, device):
        self.device = device

    def transform_batch(self, input, gt) -> Tuple:
        """
        Apply the transformation to the input and the ground truth that are batch formatted
        (batch_size, nfeature, shape)

        Apply a transformation to the input and the ground truth

        :param input: List of patient input formatted items
        :param gt: List of patient gt formatted items
        :return: Tuple of transformed values
        """
        transformation = self.get_transformation()
        return transformation(input), transformation(gt)


    def get_transformation(self):
        """
        Get the transformation

        :return: Function that corresponds to the transformation
        """
        to_torch = lambda x: torch.tensor(
            x, dtype=torch.float32
        ).to(self.device)
        to_array = lambda patient_folder: np.array(
            [
                [
                    patient_image.get_fdata()
                    for patient_image in patient_folder
                ]
                for patient_folder in patient_folders
            ]
        )
        transformation = lambda x: to_torch(to_array(x))


