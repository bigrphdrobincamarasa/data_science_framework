"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-09

**Project** : baseline_unet

**Module that contains the codes that segmentation augmentation**
"""

MODULE = ['data_science_framework', 'data_augmentation', 'segmentation_augmentation']

from .SegmentationFlip import SegmentationFlip
from .SegmentationCropHalf import SegmentationCropHalf
from .SegmentationGTExpander import SegmentationGTExpander
from .SegmentationImageTransformation import SegmentationImageTransformation
from .SegmentationNormalization import SegmentationNormalization
from .SegmentationPatientTransformation import SegmentationPatientTransformation
from .SegmentationRotation import SegmentationRotation
from .SegmentationTiling import SegmentationTiling
from .SegmentationToTorch import SegmentationToTorch
from .SegmentationTransformation import SegmentationTransformation
from .SegmentationInputExpander import SegmentationInputExpander
