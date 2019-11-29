"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-11-28

**Project** : data_science_framework

TODO: rewrite all functions
TODO: doc
**  **
"""
import json
import os
import time

from data_spy.loggers import MODULE
from data_spy.loggers.experiment_utils import get_git_current_state, timer
from scripting.test_manager import set_test_folders
from settings import PROJECT_ROOT, TEST_ROOT


def test_git_current_state() -> None:
    """
    Function that test the current git status of a folder

    :return: None
    """
    output = get_git_current_state(PROJECT_ROOT)
    assert output['hash'] != ''
    assert output['branch'] != ''
    assert output['status'] != ''

    output = get_git_current_state(os.path.join(PROJECT_ROOT, 'foo'))
    assert output['hash'] == ''
    assert output['branch'] == ''
    assert output['status'] == ''


@set_test_folders(output_root=TEST_ROOT, current_module=MODULE)
def test_timer(output_folder):
    """
    TODO doc

    :param output:
    :return:
    """
    @timer(process_name='g', folder=output_folder, tag='test')
    def g(*args, **kwargs):
        time.sleep(0.1)

    @timer(process_name='f', folder=output_folder, tag='test')
    def f(*args, **kwargs):
        for i in range(5):
            g()
    f()
    with open(os.path.join(output_folder, '#timer_test.json')) as handle:
        output_dict = json.load(handle)
    assert output_dict['name'] == 'test'

    # Test main process
    assert output_dict['subprocesses'][0]['name'] == 'f'
    assert len(output_dict['subprocesses'][0]['subprocesses']) == 5

    # Test subprocess
    assert output_dict['subprocesses'][0]['subprocesses'][0]['name'] == 'g'
    assert len(output_dict['subprocesses'][0]['subprocesses'][0]['subprocesses']) == 0
