"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-17

**Project** : baseline_unet

**Module that contains the codes the different trainers**
"""
MODULE = ['data_science_framework', 'pytorch_utils', 'trainer']

from .Trainer import Trainer
from .VanillaTrainer import VanillaTrainer
