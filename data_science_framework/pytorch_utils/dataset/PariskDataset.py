"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-06

**Project** : baseline_unet

** Class that implements DataGenerator **

"""
import os

from baseline_unet.experiment_objects import PariskDataSplitter
from torch.utils.data import Dataset
from baseline_unet.settings import DEVICE, DATA_ROOT
from data_science_framework.data_spy.loggers.experiment_loggers import timer
from torch.utils.data import Dataset
import torch
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from baseline_unet.utils import generate_filename
from data_science_framework.data_augmentation.segmentation_augmentation import \
    SegmentationTransformation
from typing import List


class PariskDataset(Dataset):
    """
    Class that generates data

    :param batch_size: Size of the batch
    """

    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size

    def process_parameters(
            self, data_splitter: PariskDataSplitter, dataframe,
            transformations: List[SegmentationTransformation]
    ) -> None:
        """
        Method that processes parameters

        :param data_splitter: Data splitter object defining your experiment
        :param dataframe: Dataframe containing the patients studied by this datagenerator
        :return: None
        """
        self.data_ressources = data_splitter.data_ressources
        self.dataframe = dataframe
        self.transformations = transformations

    def __getitem__(self, idx, *args, **kwargs):
        # Get indices
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) == int:
            idx = [idx]

        # Get dataframe
        dataframe_ = self.dataframe.reset_index().iloc[idx]

        # Get data
        input_batch = []
        gt_batch = []
        for index, row in dataframe_.iterrows():
            input_batch.append(
                [
                    nib.load(
                        self.data_ressources['input_images'][
                            generate_filename(row, 'input_image', modality=i, extension='nii.gz')
                        ]['path']
                    ) for i in range(1, 6)
                ]
            )
            gt_batch.append(
                [
                    nib.load(
                        self.data_ressources['gt_images'][
                            generate_filename(row, 'gt_image', extension='nii.gz')
                        ]['path']
                    )
                ]
            )

        # Apply transformations
        for i, transformation in enumerate(self.transformations):
            input_batch, gt_batch = transformation.transform_batch(
                input_batch, gt_batch
            )

        # Return batch
        return input_batch, gt_batch

    def __len__(self):
        return int(len(self.dataframe)/self.batch_size)
