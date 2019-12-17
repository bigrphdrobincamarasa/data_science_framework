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
from datascience_framework.pytorch_utils.trainer import Trainer
import torch


class VanillaTrainer(Trainer):
    """
    Class that implements Trainer

    :param nb_epoch: Number of epochs of training
    :param batch_size: Number of value in the batch size
    :param trainning_dataset: Trainning dataset
    :param validation_dataset: Validation dataset
    """
    def __init__(
            self, nb_epochs: int,
            batch_size: int, validation_generator: torch.nn.Dataset,
            trainning_generator: torch.nn.Dataset
    ):
        # Parameters
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size

        # Parameters
        self.loss = None
        self.optimizer = None

    def set_loss(self, loss):
        """
        Method that set loss

        :param loss:
        :return:
        """
        pass

    @timer
    def run(self) -> None:
        """
        Run entire training

        :return: None
        """
        pass

    @timer
    def run_epoch(self) -> None:
        """
        Run an entire epoch

        :return: None
        """
        pass

    @timer
    def run_validation(self) -> None:
        """
        Run validation

        :return: None
        """
        pass

    @timer
    def run_training_batch(self) -> None:
        """
        Run training batch

        :return: None
        """
        pass
