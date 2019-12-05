"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-11-28

**Project** : data_science_framework

** File that helps handle git **
"""
import os
import pandas as pd

from data_science_framework.settings import FILENAME_TEMPLATE


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


def clear_experiments(folder: str) -> None:
    """
    Clear uncommented experiments

    :param folder: Path to the folder to clear
    :return: None
    """
    # Get tags items
    items = {
        '_'.join(item.split('_')[1:]).split('.')[0]
        for item in os.listdir(folder) if '#glogger' in item
    }
    for item in list(items):
        # Get csv input
        csv_input = os.path.join(
            folder, FILENAME_TEMPLATE['glogger']['input'].format(item)
        )

        # Get csv output
        csv_output = os.path.join(
            folder, FILENAME_TEMPLATE['glogger']['output'].format(item)
        )

        # Input dataframe
        df_input = pd.read_csv(csv_input)
        df_output = pd.read_csv(csv_output)

        # Get commented experiments
        df_input = df_input.fillna('')
        commented_ids = df_input[df_input['comment'] != '']['index']

        # Update dataframes
        df_input = df_input[df_input['index'].isin(commented_ids)]
        df_output = df_output[df_output['index'].isin(commented_ids)]

        # Save updated dataframes
        df_input.to_csv(csv_input, index=False)
        df_output.to_csv(csv_output, index=False)


