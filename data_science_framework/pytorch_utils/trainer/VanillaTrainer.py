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
from data_science_framework.pytorch_utils.trainer import Trainer
import torch
import numpy as np


class VanillaTrainer(Trainer):
    """
    Class that implements Trainer

    :param nb_epoch: Number of epochs of training
    :param batch_size: Number of value in the batch size
    :param trainning_dataset: Trainning dataset
    """
    def __init__(
            self, nb_epochs: int = 250,
            batch_size: int = 2, device: str = 'cpu'
    ):
        # Parameters
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.device = device

        # Parameters
        self.model = None
        self.loss = None
        self.optimizer = None
        self.validation_generator = None
        self.trainning_generator = None
        self.callbacks = None

    @timer
    def run(self, **kwargs) -> None:
        """
        Run entire training

        :return: None
        """
        for epoch in range(self.nb_epochs):
            print(
                'Epoch {} / {}'.format(epoch + 1, self.nb_epochs)
            )
            self.run_epoch(epoch=epoch, **kwargs)

    @timer
    def run_epoch(self, epoch: int, **kwargs) -> None:
        """
        Run an entire epoch

        :param epoch: Number of the current epoch
        :return: None
        """
        # TODO: make it a method of the Dataset
        start_batch_indices = list(range(0, len(self.trainning_generator), self.batch_size))[:-1]

        # Loop over each epoch
        batch_losses = []
        for start_batch_index in start_batch_indices:
            # Get values from batch
            data, target = self.trainning_generator[
                           start_batch_index: start_batch_index + self.batch_size
                           ]

            # Clear gradient
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)

            # Compute loss
            loss = self.loss_function(output, target)
            loss_value = loss.item()

            # Backward pass
            loss.backward()

            # Optimizer step
            self.optimizer.step()

        # Test validation
        self.run_validation(epoch=epoch, **kwargs)


    @timer
    def run_validation(self, epoch, **kwargs) -> None:
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
