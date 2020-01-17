"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-11-25

**Project** : data_science_framework

**Contains the global settings of the template**
"""
import os

# Set project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# Set text_file.txt root
TEST_ROOT = os.path.join(PROJECT_ROOT, 'test')

# Set ressources root
RESSOURCES_ROOT = os.path.join(PROJECT_ROOT, 'ressources')

FILENAME_TEMPLATE = {
    'glogger': {
        'input': '#glogger_{}.csv',
        'output': '#gloggeroutput_{}.csv',
        'experiment_folder': '#{}_{}'
    },
    'tlogger': '#tlogger_{}.json',
    'mlogger': '#mlogger_{}.csv',
    'dlogger': {
        'file': '#dlogger_{}.{}',
        'directory': '#dlogger_{}'
    }
}

DEVICE = 'cpu'
