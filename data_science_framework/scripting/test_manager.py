"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-11-28

**Project** : data_science_framework

**File that contains the codes related to file management for test**
"""
import os
from typing import Callable

from data_science_framework.scripting.file_structure_manager import get_dir_structure, create_error_less_directory


def set_test_folders(
        current_module: str, ressources_root: str = None, output_root: str = None
) -> Callable:
    """
    Decorator that generate output test directory and load ressources
    for a test

    :param ressources_root: Path root of the ressources folder
    :param output_root: Path root of the output folder
    :param current_module: List of the modules
    :return: The test function updated
    """
    def decorator(f):
        def wrapper():
            # Generate path
            if ressources_root != None:
                ressources_folder = os.path.join(
                    ressources_root, *tuple(current_module)
                )
                ressources_structure = get_dir_structure(ressources_folder)
            if output_root != None:
                output_folder = os.path.join(
                    output_root, *tuple(current_module)
                )
                create_error_less_directory(output_folder, override=True)

            if ressources_root == None:
                f(output_folder)
            elif output_root == None:
                f(ressources_structure)
            else:
                f(ressources_structure, output_folder)
        return wrapper
    return decorator
