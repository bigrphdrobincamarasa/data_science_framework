"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-11-24

**Project** : data_science_framework

**This class represents a parameters**
"""
import click
from click import Option
from click.decorators import _param_memo

from data_science_framework.scripting.doc_management import parse_class_docstring


class Parameter:
    def __init__(
            self, parameter: str, default,
            object_key: str, documentation: str=''
    ):
        """
        Class that represents a parameter

        :param parameter: Parameter name
        :param default: Default value of the option
        :param object_key: Key of the object
        :param documentation: Documentation of the function
        """
        # Set option name
        self.option_name = '-'.join(
            ['-', object_key, parameter.replace('_', '-')]
        )
        # Set argument name
        self.argument_name = '_'.join([object_key, parameter])

        # Set help information
        try:
            self.help = parse_class_docstring(documentation)[parameter]
        except Exception as e:
            self.help = ''
            print(
                '- {} : Could not document {} because it does not follow docstring format'\
                    .format(object_key, parameter)
            )

        # Set default value
        self.default = default

        # Set type
        self.type = type(default)


    def render_click_option(self, f):
        """
        Add click option to the click command interface

        :param f: The function under study
        :return: None
        """
        if self.type in [str, int, float, bool]:
            click_dict = {
                'show_default': True,
                'default': self.default,
                'help': self.help,
                'type': self.type,
            }
            OptionClass = click_dict.pop('cls', Option)
            _param_memo(f, OptionClass((self.option_name, ), **click_dict))
        elif self.type == list:
            click_dict = {
                'show_default': True,
                'default': self.default,
                'help': self.help,
                'type': str,
            }
            OptionClass = click_dict.pop('cls', Option)
            _param_memo(f, OptionClass((self.option_name, ), **click_dict))

    def __str__(self):
        return '; '.join(
            [
                self.option_name, str(self.type),
                str(self.default), self.help
            ]
        )
