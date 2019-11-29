"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-11-28

**Project** : data_science_framework

** File that helps handle git **
"""
import datetime
import json
import os
from collections import Callable


def get_git_current_state(project_root: str) -> dict:
    """
    Function that get the current git state of a folder

    :param project_root: root of the project under study
    :return: None
    """
    output={}
    # Get hash
    with os.popen(
            cmd='git --git-dir {}/.git rev-parse HEAD'.format(project_root)
    ) as stream:
        output['hash'] = stream.read()[:-1]

    # Get status
    with os.popen(
            cmd='git --git-dir {}/.git status'.format(project_root)
    ) as stream:
        output['status'] = stream.read()

    # Get status
    with os.popen(
            cmd='git --git-dir {}/.git rev-parse --abbrev-ref HEAD'.format(project_root)
    ) as stream:
        output['branch'] = stream.read()[:-1]
    return output


def timer(
        process_name: str, folder: str,
        tag: str
) -> Callable:
    """
    TODO doc and test
    """
    def decorator(f):
        def wrapper(*args, **kwargs):
            # Generate filename
            filename = os.path.join(folder, '#timer_{}.json'.format(tag))

            # Get time
            start_time = datetime.datetime.now().timestamp()

            # Launch function
            f(*args, **kwargs)

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
                'name': tag,
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

                if timer_dict_['end_time'] < process['start_time']:
                    # Subprocess case
                    timer_dict_['subprocesses'].append(process)
                    timer_dict_['end_time'] = process['end_time']

                    with open(filename, 'w') as handle:
                        json.dump(timer_dict_, handle)
                else:
                    process['subprocesses'] = timer_dict_['subprocesses']
                    prototype['subprocesses'] = [process]

                    with open(filename, 'w') as handle:
                        json.dump(prototype, handle)
            else:
                prototype['subprocesses'].append(process)
                with open(filename, 'w') as handle:
                    json.dump(prototype, handle)
        return wrapper
    return decorator


