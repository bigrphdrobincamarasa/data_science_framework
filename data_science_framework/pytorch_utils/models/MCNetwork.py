"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-11

**Project** : src

**Class that implements MCNetwork structure**

"""
import torch.nn as nn
import torch

class MCNetwork(nn.Module):
    """
    Class that implements MCNetwork structure

    :param n_iter: Number of iteration of the MCDropout
    """
    def __init__(n_iter: int, dropout: float):
        self.dropout = dropout
        self.n_iter = n_iter


