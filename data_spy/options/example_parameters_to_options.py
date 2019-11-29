"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-11-25

**Project** : data_science_framework

TODO: doc
**  **
"""
import click
import numpy as np
from pyfiglet import Figlet

from data_spy.options.option_manager import parameters_to_options, initialize_experiment_parameters


class Network1:
    """
    Class Network1

    :param depth: Depth of the network
    :param in_channel: Number of channel of the input layer
    :param out_channel: Number of channels of the output layer
    :param dropout: Type of dropout applied
    :param callbacks: Callbacks applied to the network
    """
    def __init__(self, depth=4, in_channel=4, out_channel=6, dropout='training', callbacks=None):
        self.depth = depth
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.dropout = dropout
        self.callbacks = callbacks


class Network2:
    """
    Class Network2

    :param depth: Depth of the network
    :param in_channel: Number of channel of the input layer
    :param out_channel: Number of channels of the output layer
    :param mc_dropout: Type of dropout applied
    :param callbacks: Callbacks applied to the network
    """
    def __init__(self, depth=4, callbacks=None):
        self.depth = depth
        self.callbacks = callbacks


EXPERIMENT_OBJECTS = {
    'generator': Network1(depth=6),
    'discriminator': Network2(depth=3)
}


@click.command()
@parameters_to_options(experiment_objects=EXPERIMENT_OBJECTS)
def experiment(**option_values):
    initialize_experiment_parameters(
        experiment_objects=EXPERIMENT_OBJECTS,
        option_values=option_values
    )
    print(6)


if __name__ == '__main__':
    experiment()
