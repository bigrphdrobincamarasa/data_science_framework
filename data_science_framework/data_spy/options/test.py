"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-11-28

**Project** : data_science_framework

**File that the functions and classes of the data_spy.options module**
"""

from data_science_framework.data_spy.options import Parameter
from data_science_framework.data_spy.options.option_manager import get_parameters,\
    parameters_to_options, initialize_experiment_parameters
import numpy as np


def test_initialize_experiment_parameters() -> None:
    """
    Function that tests initialize_experiment_parameters

    :return: None
    """
    class Fooa:
        def __init__(self, a=5, b=6, c=7, d=[1, 2, 3], **kwargs):
                    self.a = a
                    self.b = b
                    self.c = c
                    self.d = d

    # Initialize experiment objects
    experiment_objects = {
        'test': Fooa(4)
    }

    # Initialize experiment objects
    initialize_experiment_parameters(
        experiment_objects=experiment_objects,
        option_values={
            'test_a': 6,
            'test_b': 8,
            'test_d': '1,2,3'
        }
    )
    assert experiment_objects['test'].a == 6
    assert experiment_objects['test'].b == 8
    assert experiment_objects['test'].c == 7
    assert experiment_objects['test'].d == ['1', '2', '3']


def test_manage_option() -> None:
    """
    Function that tests manage_option
    #TODO: improve test

    :return: None
    """
    class Fooa:
        def __init__(self, a, b=6, c=7, *args, **kwargs):
            self.a=a
            self.b=b
            self.c=c

    experiment_objects = {
        'test': Fooa(4)
    }
    try:
        parameters_to_options(experiment_objects=experiment_objects)
    except:
        assert False


def test_Parameter() -> None:
    """
    Function that tests Parameter

    :return: None
    """
    # Test initialisation
    docstring = 'Header :param ressources: test'

    parameter = Parameter(
        parameter='ressources',
        default=4,
        object_key='test',
        documentation=docstring
    )
    assert 'test' in parameter.help
    assert parameter.argument_name == 'test_ressources'
    assert parameter.option_name == '--test-ressources'
    assert parameter.type == int
    assert parameter.default == 4

    try:
        @parameter.render_click_option
        def foo(**kwargs):
            pass
        assert foo is None
    except:
        assert False


def test_get_parameters() -> None:
    """
    Function that tests get_parameters

    :return: None
    """
    #Define a foo class
    class Fooa:
        def __init__(self, a=1, b_c=['a', 'b'], d=np.array(5), **kwargs):
            self.a = a
            self.b_c = b_c
            self.d = d
            self.c = kwargs['c']

    # Initiate object
    fooa = Fooa(a=3, c='test')

    # Get parameters
    parameters = get_parameters(
        optionable_class_object=fooa,
        object_key='test'
    )

    # Test arg variable
    assert fooa.c == 'test'

    # Test the number of parameters
    assert len(parameters) == 3

    # Test the option name
    assert parameters[0].option_name == '--test-a'

    # Test the default value
    assert parameters[0].default == 3

