"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-11-28

**Project** : data_science_framework

**  **
"""
import json
import os
import shutil

import pandas as pd
import numpy as np


def create_error_less_directory(path: str, override: bool = False) -> None:
    """
    Function that create a dir

    :param path: Path of the folder to create
    :param override: Boolean true if user want to override
    :return: None
    """
    # Test if path exists
    if os.path.isdir(path):
        if override:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        # Create folder
        os.makedirs(path)


def get_dir_structure(
        path_root: str, subfolder_tree_height: int = None,
        supported_extensions: list = ['csv', 'json', 'npy', 'txt'],
) -> dict:
    """
    Function that transform a folder tree into a python dictionnary

    :param path_root: Path root of the folder under study
    :param subfolder_tree_height: Height of the subfolders parsed by the
    function
    :param supported_extensions: Extension of the files that are transformed
    into objects. (supported librairies : pandas, json, numpy)
    :return: The path tree
    """
    if subfolder_tree_height == 0:
        return {'path': path_root, 'object': 'dir'}

    # Initialize output
    output = {}

    # Loop over folder items
    for folder_item in os.listdir(path_root):
        path_root_ = os.path.join(path_root, folder_item)
        # Test if it is a directory
        if os.path.isdir(path_root_):
            subfolder_tree_height= None if subfolder_tree_height is None else subfolder_tree_height -1
            output[folder_item] = get_dir_structure(
                path_root=path_root_,
                subfolder_tree_height=subfolder_tree_height
            )
        else:
            output[folder_item] = get_file_structure(
                file_path=path_root_,
                supported_extensions=supported_extensions
            )
    return output


def get_file_structure(
        file_path: str, supported_extensions: list = ['csv', 'json', 'npy', 'txt'],
        csv_delimiters=[',']
) -> dict:
    """
    Function that transform a path into a dictionnary

    :param file_path: Path root of the file under study
    :param supported_extensions: Extension of the files that are transformed
    into objects. (supported librairies : pandas, json, numpy)
    :param csv_delimiters: Format used to read csv file
    :return: The path, the object (None if not applicable)
    """
    # Get filename and extension if applicable
    try:
        filename, extension = tuple(
            file_path.split('/')[-1].split('.')
        )
    except:
        return {'path': file_path, 'object': None}
    if not extension in supported_extensions:
        return {'path': file_path, 'object': None}

    # Test csv files
    if extension == 'csv':
        for csv_delimiter in [',', ';']:
            try:
                return {
                    'path': file_path,
                    'object': pd.read_csv(file_path, csv_delimiter)
                }
            except Exception as e:
                print(e)
                pass
        print('{}: does not follow the correct format')
        return {'path': file_path, 'object': None}

    # Test json files
    if extension == 'json':
        try:
            with open(file_path, 'r') as handle:
                return {'path': file_path, 'object': json.load(handle)}
        except:
            print('{}: does not follow the correct format')

    # Test numpy files
    if extension == 'npy':
        try:
                return {'path': file_path, 'object': np.load(file_path)}
        except:
            print('{}: does not follow the correct format')

    # Test txt files
    if extension == 'txt':
        try:
            with open(file_path, 'r') as handle:
                return {'path': file_path, 'object': handle.read()}
        except:
            print('{}: does not follow the correct format')
    return {'path': file_path, 'object': None}
