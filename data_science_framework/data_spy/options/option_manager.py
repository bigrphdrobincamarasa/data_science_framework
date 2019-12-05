"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-11-28

**Project** : data_science_framework

** File that contains the options manager **
"""
import inspect
from typing import Callable

from data_science_framework.data_spy.options import Parameter


def parameters_to_options(experiment_objects: dict) -> Callable:
    """
    This function manage the options of your experiments

    :param experiment_objects: A dictionnary of list of optionable
    objects used in your experiment
    :return: A decorator that can be applied on every fonction
    """
    def decorator(function):
        # Loop over each function
        for object_key, experiment_object in experiment_objects.items():
            for experiment_parameter in get_parameters(
                optionable_class_object=experiment_object, object_key=object_key
            ):
                experiment_parameter.render_click_option(
                    f=function,
                )
        return function
    return decorator


def get_parameters(optionable_class_object: object, object_key):
    """
    Function that transforms your constructor items into
    a list of Parameter objects

    :param optionable_class_object: Object that is optionable (every constructor argument has default value)
    :param object_key: Name of the main object
    :return: None
    """
    # Initialise output
    output = []

    # Get __init__ values
    init_values = inspect.getfullargspec(optionable_class_object.__init__)._asdict()

    # Get default
    for parameter in init_values['args'][1:]:
        documentation = optionable_class_object.__doc__
        try:
            output.append(
                Parameter(
                    parameter=parameter, default=optionable_class_object.__getattribute__(parameter),
                    object_key=object_key, documentation=documentation
                )
            )
        except Exception as e:
            print(str(e))
    return output


def initialize_experiment_parameters(
        experiment_objects: dict, option_values: dict
) -> None:
    """
    Function that returns experiment objects initialized by option parser input

    :param experiment_objects: Dictionnary of objects corresponding to the experiment
    required objects
    :param option_values: Dictionnary containing the values inputted as options
    :return: None
    """
    for key, value in option_values.items():
        try:
            # Extract experiment object key
            experiment_object_key_ = key.split('_')[0]

            # Extract parameter key
            parameter_key_ = '_'.join(key.split('_')[1:])
            experiment_objects[experiment_object_key_].__setattr__(parameter_key_, value)
        except:
            pass
