"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-10

**Project** : src

**Class that implements layer generator structures**
"""
import torch.nn as nn
import torch
from data_science_framework.pytorch_utils.layers import MCDownConvolution2Axis3DLayer,\
    MCDownConvolution3DLayer, MCUpConvolution3DLayer


class MCLayerGenerator(object):
    """
    Class that implements MCLayerGenerator structure

    :param dropout: Value of the dropout
    :param layer: Class of the considered layer
    """
    def __init__(self, dropout: float, layer) -> None:
        self.dropout = dropout
        self.layer = layer

    def __call__(self, *args, **kwargs):
        """__call__

        :param *args: Arguments
        :param **kwargs: Keyword arguments
        """
        return self.layer(dropout=self.dropout, *args, **kwargs)


