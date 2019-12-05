"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-11-28

**Project** : data_science_framework

** File that contains the docstring management functions **
"""


def parse_class_docstring(docstring: str) -> dict:
    """
    Parse class docstring to dictionnary

    :param docstring: Content of the class docstring
    :return: Dictionnary with docstring arguments
    """
    try:
        # Get arguments documentation
        documented_arguments = docstring.split(':param ')[1:]
        output = {}
        for documented_argument in documented_arguments:
            try:
                output[documented_argument.split(': ')[0]] = documented_argument.split(': ')[1]
            except:
                pass
        return output
    except:
        print('This class do not follow docstring convention')
        return {}


def parse_function_docstring(docstring: str) -> dict:
    """
    Parse function docstring to dictionnary

    :param docstring: Content of the class docstring
    :return: Dictionnary with docstring arguments
    """
    try:
        # Get function return docstring
        split_documentation = docstring.split(':return:')
        if len(split_documentation) == 1:
            output = {}
        elif len(split_documentation) == 2:
            output = {'return': split_documentation[1]}
        else:
            print('This function do not follow docstring convention')
            return {}

        # Get arguments docstring
        split_documentation = split_documentation[0].split(':param ')
        output['header'] = split_documentation[0]
        for split_item in split_documentation[1:]:
            values_ = split_item.split(':')
            if len(values_) != 2:
                print('This function do not follow docstring convention')
                return {}
            output[values_[0]] = values_[1]
        return output
    except Exception as e:
        print('This function do not follow docstring convention')
        return {}



