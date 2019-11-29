"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-11-28

**Project** : data_science_framework

** File that contains the codes related to file management for test **
"""
import os
from typing import Callable

from scripting.file_structure_manager import get_dir_structure, create_error_less_directory


def set_test_folders(
        current_module: str, ressouces_root: str, output_root: str
) -> Callable:
    """
    Decorator that generate output test directory and load ressources
    for a test

    :param ressouces_root: Path root of the ressources folder
    :param output_root: Path root of the output folder
    :return: The test function updated
    """
    def decorator(f):
        def wrapper():
            # Generate path
            ressources_folder = os.path.join(
                ressouces_root, *tuple(current_module)
            )
            output_folder = os.path.join(
                output_root, *tuple(current_module)
            )

            # Create output directory
            create_error_less_directory(output_folder)

            # Apply test
            f(
                get_dir_structure(ressources_folder),
                output_folder
            )
        return wrapper
    return decorator
