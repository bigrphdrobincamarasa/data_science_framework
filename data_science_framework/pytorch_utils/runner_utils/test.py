"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-10

**Project** : src

**File that tests codes of runner_utils module**
"""
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adadelta import Adadelta
from data_science_framework.pytorch_utils.runner_utils.Optimizer import Optimizer


def test_Optimizer() -> None:
    """
    Function that tests Optimizer class

    :return: None
    """
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

    # Test adadelta
    optimizer = Optimizer(
        name='adadelta',
        learning_rate=10 ** (-3)
    )
    optimizer.process_parameters()
    model = Model()
    output = optimizer.get_optimizer(model)
    assert type(output) == type(Adadelta(model.parameters()))


