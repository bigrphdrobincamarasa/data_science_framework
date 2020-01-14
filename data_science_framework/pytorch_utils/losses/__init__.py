"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-17

**Project** : baseline_unet

**Module that contains the codes that implements losses**
"""

MODULE = ['data_science_framework', 'pytorch_utils', 'losses']

from .Loss import Loss
from .BinaryCrossEntropyLoss import BinaryCrossEntropyLoss
from .DiceLoss import DiceLoss
