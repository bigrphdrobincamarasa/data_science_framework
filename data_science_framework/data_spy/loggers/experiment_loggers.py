"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-11-28

**Project** : data_science_framework

** File that contains the main loggers **
"""
import datetime
import json
import os
import pandas as pd
from pyfiglet import Figlet

from data_science_framework.data_spy.loggers.experiment_utils import get_git_current_state
from data_science_framework.scripting.file_structure_manager import create_error_less_directory
from data_science_framework.settings import FILENAME_TEMPLATE


def global_logger(
        folder: str, project_root: str, tag: str = 'experiment',
        test: bool = False
):
    """
    Function that return a decorator that logs the experiments parameters and the output of the experiment

    :param folder: Path to the folder that contains the experiments
    :param project_root: Root of the project (for github logs)
    :param tag: Tag associated with the experiment
    :param test: Boolean that is true only if the experiment is a test
    :return: Decorator
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
            if os.path.isfile(csv_path):
                df = pd.read_csv(csv_path).set_index('index')
            else:
                df = pd.DataFrame(columns=['index']).set_index('index')


            # Get git values
            git_values = get_git_current_state(project_root)
            git_values.pop('status')

            # Get time values
            start_time = datetime.datetime.now()

            # Append experiments input values
            df = df.append(
                {'comment': '',  'time': start_time, **git_values, **kwargs},
                ignore_index=True
            )

            # Save csv file
            df.reset_index().to_csv(csv_path, index=False)

            # Get the added row
            row_with_id = df.reset_index().iloc[-1].to_dict()

            # Create experiment folder
            experiment_folder = os.path.join(folder, '#{}_{}'.format(tag_, row_with_id['index']))
            create_error_less_directory(experiment_folder)

            #Apply fonction
            print(Figlet().renderText('{} {}'.format(tag_, row_with_id['index'])))
            output = f(
                *args, **kwargs, experiment_folder=experiment_folder,
                index=row_with_id['index']
            )

            # Get csv output path
            csv_output_path = os.path.join(folder, FILENAME_TEMPLATE['glogger']['output'].format(tag_))

            # Log output of the function
            if os.path.isfile(csv_output_path):
                df = pd.read_csv(csv_output_path)
            else:
                df = pd.DataFrame(columns=['index'])

            if type(output) == None:
                df = df.append(
                    {
                        'index': row_with_id['index'],
                        'total_time': (
                                datetime.datetime.now().timestamp() - start_time.timestamp()
                        ).__round__(4),
                        **output
                    },
                    ignore_index=True
                )
            else:
                df = df.append(
                    {
                        'index': row_with_id['index'],
                        'total_time': (
                                datetime.datetime.now().timestamp() - start_time.timestamp()
                        ).__round__(4)
                    },
                    ignore_index=True
                )
            df.to_csv(csv_output_path, index=False)
            return output
        return wrapper
    return decorator


def metric_logger(f):
    """
    Decorator that logs a metric

    :param f: Decorated function
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
                        columns=list(kwargs['meta']) + ['index']
                    ).set_index('index')

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
        except:
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

