"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-09

**Project** : src

** File that contains the code to convert data to torch tensor **
"""
import torch
import numpy as np


def convert_nifty_batch_to_torch_tensor(patients_images: list, device: str) -> torch.Tensor:
    """
    Function that converts a nifty batch to a torch tensor

    :param patients_images: List of list of nifty images loaded by nibabel
    :param device: Name of the device (cpu or gpu)
    :return: Torch tensor attached to the required device
    """
    array_ = np.array(
        [
            [
                patient_image.get_fdata()
                for patient_image in patient_images
            ]
            for patient_images in patients_images
        ]
    )
    return torch.tensor(
        array_, dtype=torch.float32
    ).to(device)

