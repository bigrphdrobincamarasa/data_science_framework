"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-11-28

**Project** : data_science_framework

**File that contains the main loggers**
"""
import datetime
import json
import os
from typing import Callable

import pandas as pd
import numpy as np
import nibabel as nib
from pyfiglet import Figlet

from data_science_framework.data_spy.loggers.experiment_utils import get_git_current_state
from data_science_framework.scripting.file_structure_manager import create_error_less_directory
from data_science_framework.settings import FILENAME_TEMPLATE


def global_logger(
        folder: str, project_root: str, tag: str = 'experiment',
        test: bool = False
) -> Callable:
    """
    Function that logs the experiments parameters and the output of the experiment

    :param folder: Path to the folder that contains the experiments
    :param project_root: Root of the project (for github logs)
    :param tag: Tag associated with the experiment
    :param test: Boolean that is true only if the experiment is a test
    :return: Decorator to apply above the experiment function under study
    """
    def decorator(f):
        def wrapper(*args, **kwargs):
            # Update tag if necessary
            if test:
                tag_ = tag + '_test'
            else:
                tag_ = tag

            # Manage dataframe
            csv_path = os.path.join(folder, FILENAME_TEMPLATE['glogger']['input'].format(tag_))

            # Create parent directory of the csv file
            create_error_less_directory(os.path.dirname(os.path.abspath(csv_path)))

            if os.path.isfile(csv_path):
                df = pd.read_csv(csv_path)
                index = int(df['index'].max() + 1)
            else:
                df = pd.DataFrame()
                index = 0

            # Get git values
            git_values = get_git_current_state(project_root)
            git_values.pop('status')

            # Get time values
            start_time = datetime.datetime.now()

            # Append experiments input values
            df = df.append(
                {'index': index, 'comment': '',  'time': start_time, **git_values, **kwargs},
                ignore_index=True
            )

            # Save csv file
            df.to_csv(csv_path, index=False)

            # Create experiment folder
            experiment_folder = os.path.join(folder, '#{}_{}'.format(tag_, index))
            create_error_less_directory(experiment_folder)

            #Apply fonction
            print(Figlet().renderText('{} {}'.format(tag_, index)))
            return f(
                *args, **kwargs, experiment_folder=experiment_folder,
                index=index
            )

        return wrapper
    return decorator


def metric_logger(f: Callable) -> Callable:
    """
    Decorator that logs a metric function output

    :param f: Function under study
    :return: Decorated function
    """
    def wrapper(*args, **kwargs):
        try:
            if kwargs['save']:
                # apply function
                output = f(*args, **kwargs)

                # Get index
                index = kwargs['meta'].pop('index')

                # Get csv path
                csv_metric_file = os.path.join(
                    kwargs['folder'], FILENAME_TEMPLATE['mlogger'].format(kwargs['tag'])
                )

                # Get already saved data
                if os.path.isfile(csv_metric_file):
                    metric_dataframe = pd.read_csv(csv_metric_file).set_index('index')
                else:
                    metric_dataframe = pd.DataFrame(
                        columns=list(kwargs['meta'])
                    )

                # Add value in the dataframe
                if index in metric_dataframe.index:
                    # if the row already exists
                    row_to_update = {
                        key: value
                        for key, value in metric_dataframe.loc[index].to_dict().items()
                        if key != kwargs['metric']
                    }
                    metric_dataframe = metric_dataframe.loc[metric_dataframe.index != index]
                    metric_dataframe = metric_dataframe.reset_index().append(
                        {
                            'index': index,
                            kwargs['metric']: output,
                            **row_to_update
                        },
                        ignore_index=True
                    )
                else:
                    metric_dataframe = metric_dataframe.reset_index().append(
                        {
                            'index': index,
                            kwargs['metric']: output,
                            **kwargs['meta']
                        },
                        ignore_index=True
                    )
                metric_dataframe.to_csv(csv_metric_file, index=False)
            else:
                output = f(*args, **kwargs)
        except Exception as e:
            output = f(*args, **kwargs)
        return output
    return wrapper


def timer(f):
    """
    Decorator that times a function

    :param f: Function to time
    :return: Decorated function
    """
    def wrapper(*args, **kwargs):
        try:
            if kwargs['save']:
                # Create folder
                create_error_less_directory(path=kwargs['folder'])

                # Get process name
                process_name = f.__name__

                # Generate filename
                filename = os.path.join(kwargs['folder'], FILENAME_TEMPLATE['tlogger'].format(kwargs['tag']))

                # Get time
                start_time = datetime.datetime.now().timestamp()

                # Launch function
                output = f(*args, **kwargs)

                # Get end time
                end_time = datetime.datetime.now().timestamp()

                # Process log
                process = {
                    'name': process_name,
                    'process_time': (end_time - start_time).__round__(4),
                    'start_time': start_time.__round__(4),
                    'end_time': end_time.__round__(4),
                    'subprocesses': []
                }
                prototype = {
                    'name': kwargs['tag'],
                    'process_time': (end_time - start_time).__round__(4),
                    'start_time': start_time.__round__(4),
                    'end_time': end_time.__round__(4),
                    'subprocesses': []
                }

                # Test if tag timer exists
                if os.path.isfile(filename):
                    # Load json file
                    with open(filename, 'r') as handle:
                        timer_dict_ = json.load(handle)

                    # Update end time
                    timer_dict_['end_time'] = process['end_time']

                    # Loop over subprocesses
                    subprocesses = timer_dict_['subprocesses'].copy()
                    timer_dict_['subprocesses'] = []

                    for subprocess_ in subprocesses:
                        if subprocess_['end_time'] < process['start_time']:
                            # Case subprocess do not overlap current process
                            timer_dict_['subprocesses'].append(subprocess_.copy())
                        else:
                            # Case of overlapping
                            process['subprocesses'].append(subprocess_.copy())

                    # Add current process to the subprocesses
                    timer_dict_['subprocesses'].append(process)

                    with open(filename, 'w') as handle:
                        json.dump(timer_dict_, handle)

                else:
                    prototype['subprocesses'].append(process)
                    with open(filename, 'w') as handle:
                        json.dump(prototype, handle)
                return output
            else:
                return f(*args, **kwargs)
        except:
            return f(*args, **kwargs)
    return wrapper


def data_saver(f: Callable) -> Callable:
    """
    Decorator that saves data of a function

    :param f: Function under study
    :return: Decorated function
    """
    def wrapper(*args, **kwargs):
        try:
            if kwargs['save']:
                # Create save path
                if 'subdirectories' in kwargs.keys():
                    # If subdirectory is define by user
                    save_directory = os.path.join(
                        kwargs['folder'],
                        *tuple(
                            [
                                FILENAME_TEMPLATE['dlogger']['directory'].format(subdirectory)
                                for subdirectory in kwargs['subdirectories']
                            ]
                        )
                    )
                else:
                    # If subdirectory is not define by user
                    save_directory = os.path.join(
                        kwargs['folder'],
                        FILENAME_TEMPLATE['dlogger']['directory'].format(f.__name__)
                    )

                create_error_less_directory(save_directory)

                # Get output
                output = f(*args, **kwargs)
                for key, value in output.items():
                    if type(value) == str:
                        # String case
                        path_ = os.path.join(
                            save_directory, FILENAME_TEMPLATE['dlogger']['file'].format(key, 'txt')
                        )
                        with open(path_, 'w') as handle:
                            handle.write(value)

                    if type(value) == dict:
                        # Dictionnary case
                        path_ = os.path.join(
                            save_directory, FILENAME_TEMPLATE['dlogger']['file'].format(key, 'json')
                        )
                        with open(path_, 'w') as handle:
                            json.dump(value, handle)

                    elif type(value) == type(pd.DataFrame()):
                        # Dataframe case
                        path_ = os.path.join(
                            save_directory, FILENAME_TEMPLATE['dlogger']['file'].format(key, 'csv')
                        )
                        value.to_csv(path_)

                    elif type(value) == type(nib.Nifti1Image(np.eye(4), np.eye(4))):
                        # Nifty image case
                        path_ = os.path.join(
                            save_directory, FILENAME_TEMPLATE['dlogger']['file'].format(key, 'nii.gz')
                        )
                        nib.save(value, path_)
                return output
            else:
                return f(*args, **kwargs)
        except Exception as e:
            return f(*args, **kwargs)
    return wrapper
