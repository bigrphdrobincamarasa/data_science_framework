"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-17

**Project** : baseline_unet

**Class that implements DataSplitter**
"""
from data_science_framework.data_spy.loggers.experiment_loggers import data_saver, timer


class DataSplitter:
    """
    Class that implements DataSplitter
    """
    @data_saver
    @timer
    def split_data(self, *args, **kwargs) -> dict:
        """
        Function that splits dataset into train, validation and test

        :return: Dictionnary with 'train', 'validation' and 'test' keys
        """
        pass

