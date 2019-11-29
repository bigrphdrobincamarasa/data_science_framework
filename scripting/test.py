"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-11-28

**Project** : data_science_framework

** File that test scripting functions, classes, decorators ... **
"""
import os
import shutil

import pandas as pd
import numpy as np

from scripting import MODULE
from scripting.doc_management import parse_function_docstring, parse_class_docstring
from scripting.file_structure_manager import get_file_structure, get_dir_structure, create_error_less_directory
from scripting.test_manager import set_test_folders
from settings import RESSOURCES_ROOT, TEST_ROOT


def test_parse_function_docstring() -> None:
    """
    Function that test function docstring parser

    :param ressources: Folder parsed as a dictionnary
    :param output_path: Output save path
    :return: None
    """
    def test_function(test):
        """
        Header

        :return: None
        """
        pass
    output = parse_function_docstring(test_function.__doc__)
    assert 'Header' in output['header']
    assert 'None' in output['return']

    def test_function(test):
        """
        Header

        :param test: test_doc
        :return: None
        """
        pass
    output = parse_function_docstring(test_function.__doc__)
    assert 'Header' in output['header']
    assert 'test_doc' in output['test']
    assert 'None' in output['return']


def test_parse_class_docstring() -> None:
    """
    Function that test class_docstring parser

    :param ressources: Folder parsed as a dictionnary
    :param output_path: Output save path
    :return: None
    """
    class Test:
        """
        Header

        :param test: test_doc
        """
        pass
    output = parse_class_docstring(Test.__doc__)
    assert 'test_doc' in output['test']


@set_test_folders(
    current_module=MODULE, ressouces_root=RESSOURCES_ROOT,
    output_root=TEST_ROOT
)
def test_set_test_folders(
        ressources_file_structure:dict, output_root:str) -> None:
    """
    Function that test test_file_structure decorator

    :return: None
    """
    assert type(ressources_file_structure) == dict
    assert os.path.isdir(output_root)
    assert output_root == os.path.join(TEST_ROOT, *tuple(MODULE))


def test_create_error_less_directory() -> None:
    """
    Function that test create_error_less_directory

    :return: None
    """
    # Set folder under study
    output_path = os.path.join(TEST_ROOT, *tuple(MODULE))

    # Try with non existant folder
    try:
        shutil.rmtree(output_path)
    except:
        pass
    create_error_less_directory(output_path, override=True)
    assert os.path.isdir(output_path)
    try:
        shutil.rmtree(output_path)
    except:
        pass
    create_error_less_directory(output_path, override=False)
    assert os.path.isdir(output_path)

    # Try with existant folder
    subfolder = os.path.join(output_path, 'test')
    os.makedirs(subfolder)
    create_error_less_directory(output_path, override=False)
    assert os.path.isdir(subfolder)
    create_error_less_directory(output_path, override=True)
    assert not os.path.isdir(subfolder)


def test_get_dir_structure() -> None:
    """
    Function that test test_file_structure decorator

    :return: None
    """
    # Get path value
    path_root = os.path.join(RESSOURCES_ROOT, *tuple(MODULE))

    # Test subfolder tree height parameter
    output = get_dir_structure(
        path_root=path_root,
        subfolder_tree_height=3,
    )
    assert output['test']['test']['test']['object'] == 'dir'
    output = get_dir_structure(
        path_root=path_root,
        subfolder_tree_height=None,
    )
    assert output['test']['test']['test']['test']['test']['test']['test.txt']['object'] == ''




def test_get_file_structure() -> None:
    """
    Function that test test_file_structure decorator
    #TODO: test supported extensions

    :return: None
    """
    # Initialize tested files
    files = [
        'test.json', 'test.txt', 'test.npy', 'test.csv',
        'broken_test.json', 'broken_test.txt', 'broken_test.npy', 'broken_test.csv',
    ]
    # Initialize tested objects
    objects = [
        {'test': 'test'}, 'test\n', np.arange(5), pd.DataFrame(
            {
                'test': list('abcd')
            },
            index=pd.Index(list('1234'), name='id')
        ),
        None, None, None, None
    ]

    # Initialize values
    values = [
        {
            'path': os.path.join(RESSOURCES_ROOT, *tuple(MODULE), file),
            'object': object
        }
        for i, (file, object) in enumerate(zip(files, objects))
    ]
    for value in values:
        file_, object_ = tuple(get_file_structure(file_path=value['path']).values())
        assert file_ == value['path']
        assert type(object_) == type(value['object'])


