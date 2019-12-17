"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-17

**Project** : baseline_unet

**Module that contains the codes that implements losses**
"""

MODULE = ['losses']

from .Loss import Loss
from .BinaryCrossEntropyLoss import BinaryCrossEntropyLoss