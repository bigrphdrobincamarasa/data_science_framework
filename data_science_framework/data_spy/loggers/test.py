"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-11-28

**Project** : data_science_framework

** File that tests the module functions and decorators **
"""
import json
import os
import time
import pandas as pd
import nibabel as nib
import numpy as np

from data_science_framework.data_spy.loggers import MODULE
from data_science_framework.data_spy.loggers.experiment_loggers import metric_logger, \
    global_logger, timer, data_saver
from data_science_framework.data_spy.loggers.experiment_utils import get_git_current_state, clear_experiments
from data_science_framework.scripting.file_structure_manager import get_file_structure
from data_science_framework.scripting.test_manager import set_test_folders
from data_science_framework.settings import PROJECT_ROOT, TEST_ROOT, RESSOURCES_ROOT


def test_git_current_state() -> None:
    """
    Function that test the current git status of a folder

    :return: None
    """
    output = get_git_current_state(
        os.path.join(RESSOURCES_ROOT, *tuple(MODULE))
    )
    assert output['hash'] != ''
    assert output['branch'] != ''
    assert output['status'] != ''

    output = get_git_current_state(os.path.join(RESSOURCES_ROOT, 'foo'))
    assert output['hash'] == ''
    assert output['branch'] == ''
    assert output['status'] == ''


@set_test_folders(output_root=TEST_ROOT, current_module=MODULE)
def test_data_saver(output_folder: str) -> None:
    """
    Function that tests data_saver

    :param output_folder: Path to the output folder
    :return: None
    """
    @data_saver
    def f(*args, **kwargs):
        return {
            'train_df': pd.DataFrame(
                data=[{'a': 1, 'b': 2, 'index': 3}]
            ).set_index('index'),
            'parameters': {'a': 1, 'b': 2, 'index': 3},
            'read': 'hello world',
            'image': nib.Nifti1Image(np.eye(4), np.eye(4))
        }

    output = f(save=True, folder=output_folder)
    assert type(output) == dict
    output = f(save=True, folder=output_folder, subdirectories=['test1', 'test2'])
    assert type(output) == dict

    # Check file creation without subdirectories
    assert os.path.isfile(os.path.join(output_folder, '#dlogger_f', '#dlogger_train_df.csv'))
    assert os.path.isfile(os.path.join(output_folder, '#dlogger_f', '#dlogger_parameters.json'))
    assert os.path.isfile(os.path.join(output_folder, '#dlogger_f', '#dlogger_read.txt'))
    assert os.path.isfile(os.path.join(output_folder, '#dlogger_f', '#dlogger_image.nii.gz'))

    # Check file creation with subdirectories
    assert os.path.isfile(os.path.join(output_folder, '#dlogger_test1', '#dlogger_test2', '#dlogger_train_df.csv'))
    assert os.path.isfile(os.path.join(output_folder, '#dlogger_test1', '#dlogger_test2', '#dlogger_parameters.json'))
    assert os.path.isfile(os.path.join(output_folder, '#dlogger_test1', '#dlogger_test2', '#dlogger_read.txt'))
    assert os.path.isfile(os.path.join(output_folder, '#dlogger_test1', '#dlogger_test2', '#dlogger_image.nii.gz'))

    # Check file consistency
    assert pd.read_csv(os.path.join(output_folder, '#dlogger_f', '#dlogger_train_df.csv')).shape == (1, 3)
    with open(os.path.join(output_folder, '#dlogger_f', '#dlogger_read.txt'), 'r') as handle:
        assert handle.read() == 'hello world'
    with open(os.path.join(output_folder, '#dlogger_f', '#dlogger_parameters.json'), 'r') as handle:
        tmp_ = json.load(handle)
        assert 'a' in tmp_.keys()
        assert 'b' in tmp_.keys()
        assert 'index' in tmp_.keys()
    try:
        nib.load(os.path.join(output_folder, '#dlogger_f', '#dlogger_image.nii.gz'))
    except:
        assert False
    assert pd.read_csv(os.path.join(output_folder, '#dlogger_f', '#dlogger_train_df.csv')).shape == (1, 3)


@set_test_folders(output_root=TEST_ROOT, current_module=MODULE)
def test_timer(output_folder):
    """
    Test timer decorator generator

    :param output_folder: Path to the output folder
    :return: None
    """

    @timer
    def g(*args, **kwargs):
        time.sleep(0.1)

    @timer
    def f(*args, **kwargs):
        for i in range(5):
            g(folder=output_folder, tag='test', save=(i % 2 == 0))

    f(folder=output_folder, tag='test', save=True)
    f(folder=output_folder, tag='test', save=True)
    with open(os.path.join(output_folder, '#tlogger_test.json')) as handle:
        output_dict = json.load(handle)

    # Test tag process
    assert output_dict['name'] == 'test'
    assert len(output_dict['subprocesses']) == 2

    # Test main process
    assert output_dict['subprocesses'][0]['name'] == 'f'
    assert len(output_dict['subprocesses'][0]['subprocesses']) == 3

    # Test subprocess
    assert output_dict['subprocesses'][0]['subprocesses'][0]['name'] == 'g'
    assert len(output_dict['subprocesses'][0]['subprocesses'][0]['subprocesses']) == 0


@set_test_folders(output_root=TEST_ROOT, current_module=MODULE)
def test_metric_logger(output_folder):
    """
    Test metric logger decorator

    :param output_folder: Path to the output folder
    :return: None
    """
    @metric_logger
    def f(a, b, *args, **kwargs):
        return a * b

    @metric_logger
    def g(c, d, *args, **kwargs):
        return c + d

    @metric_logger
    def h(g, h, *args, **kwargs):
        return g - h

    for i in range(10):
        f(
            i, i**2, folder=output_folder, save=(i % 2) == 0, meta={'index': i, 'epoch': i},
            tag='phase1', metric='f'
        )
        g(
            i, i**3, folder=output_folder, save=(i % 3) == 0, meta={'index': i, 'epoch': i},
            tag='phase1', metric='g'
        )

    for i in range(5):
        h(
            i, i**3, folder=output_folder, save=True, meta={'index': i, 'epoch': i},
            tag='phase2', metric='h'
        )

    # Get paths
    phase1_path = os.path.join(output_folder, '#mlogger_phase1.csv')
    phase2_path = os.path.join(output_folder, '#mlogger_phase2.csv')

    # Get dataframes
    df_phase1 = pd.read_csv(phase1_path, index_col='index')
    df_phase2 = pd.read_csv(phase2_path, index_col='index')

    # Check sizes of the dataframe
    assert df_phase1.shape == (7, 3)
    assert df_phase2.shape == (5, 2)
    assert pd.isna(df_phase1['g'][8])
    assert pd.isna(df_phase1['f'][3])


@set_test_folders(output_root=TEST_ROOT, current_module=MODULE)
def test_global_logger(output_folder: str) -> None:
    """
    Function that test the global logger

    :param output_folder: Path to the output folder
    :return:None
    """

    @global_logger(tag='training', folder=output_folder, project_root=PROJECT_ROOT)
    def g(testa, testb, index, experiment_folder, *args, **kwargs):
        assert testa ** 2 == testb
        return {'result': testa * testb}

    for i in range(5):
        g(testa=i, testb=i ** 2)

    assert os.path.isfile(os.path.join(output_folder, '#glogger_training.csv'))

    # Test input logger
    df_test = get_file_structure(
        os.path.join(
            os.path.join(output_folder, '#glogger_training.csv')
        )
    )['object']
    assert 'index' in list(df_test)
    assert 'comment' in list(df_test)
    assert 'branch' in list(df_test)
    assert 'time' in list(df_test)
    assert 'hash' in list(df_test)
    assert df_test.shape[0] == 5

    # Test output logger
    df_test = get_file_structure(
        os.path.join(
            os.path.join(output_folder, '#gloggeroutput_training.csv')
        )
    )['object']
    assert 'index' in list(df_test)
    assert 'total_time' in list(df_test)
    assert df_test.shape[0] == 5


@set_test_folders(output_root=TEST_ROOT, current_module=MODULE)
def test_clear_experiments(output_folder: str) -> None:
    """
    Function that tests clear_experiments function

    :param output: Path to the output folder
    :return: None
    """
    @global_logger(tag='training', folder=output_folder, project_root=PROJECT_ROOT)
    def g1(testa, testb, index, experiment_folder, *args, **kwargs):
        return {'result': testa * testb}

    @global_logger(tag='training2', folder=output_folder, project_root=PROJECT_ROOT, test=True)
    def g2(testa, testb, index, experiment_folder, *args, **kwargs):
        return {'result': testa * testb}

    for i in range(5):
        g1(testa=i, testb=i**2)
        g2(testa=i, testb=i**3)

    # Comment two experiments
    input_path = os.path.join(output_folder, '#glogger_training.csv')
    df_test = get_file_structure(input_path)['object'].set_index('index')

    # Get input and output path
    df_test.loc[0, 'comment'] = 'This is a test'
    df_test.loc[4, 'comment'] = 'This is a test'
    df_test.to_csv(input_path)

    # Clear experiments
    clear_experiments(output_folder)

    # Check results
    for length, tag in [(2, 'training'), (0, 'training2_test')]:
        input_path = os.path.join(
            output_folder, '#glogger_{}.csv'.format(tag)
        )
        df = pd.read_csv(input_path)
        assert df.shape[0] == length
