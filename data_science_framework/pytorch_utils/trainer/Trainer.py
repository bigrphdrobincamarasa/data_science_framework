"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-17

**Project** : baseline_unet

**Class that implements trainer**
"""
from data_science_framework.data_spy.loggers.experiment_loggers import timer
from data_science_framework.pytorch_utils.optimizer import Optimizer


class Trainer(object):
    """
    Class that implements Trainer
    """
    def run(self) -> None:
        """
        Run entire training

        :return: None
        """
        pass

    def set_objects_attributes(self, **kwargs) -> None:
        """
        Set attributes that are objects

        :return: None
        """
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def run_epoch(self, epoch: int) -> None:
        """
        Run an entire epoch

        :param epoch: Number of the current epoch
        :return: None
        """
        pass

    def run_validation(self, epoch: int) -> None:
        """
        Run validation

        :param epoch: Number of the current epoch
        :return: None
        """
        pass

    def run_training_batch(self, epoch: int) -> None:
        """
        Run training batch

        :param epoch: Number of the current epoch
        :return: None
        """
        pass

