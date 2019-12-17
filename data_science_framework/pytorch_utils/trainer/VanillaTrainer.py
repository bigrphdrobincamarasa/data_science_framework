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
import time
from tqdm import tqdm
import numpy as np


class VanillaTrainer(Trainer):
    """
    Class that implements Trainer

    :param nb_epoch: Number of epochs of training
    :param batch_size: Number of value in the batch size
    :param trainning_dataset: Trainning dataset
    """
    def __init__(
            self, nb_epochs: int = 250, device: str = 'cpu'
    ):
        # Parameters
        self.nb_epochs = nb_epochs
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

            self.run_epoch(epoch=epoch, **kwargs)

    @timer
    def run_epoch(self, epoch: int, **kwargs) -> None:
        """
        Run an entire epoch

        :param epoch: Number of the current epoch
        :return: None
        """
        # Initialize losses value
        loss_value = 0

        # Initialize progressbar
        progress_bar = tqdm(
                enumerate(self.trainning_generator),
                desc='Epoch {} / {}'.format(epoch + 1, self.nb_epochs),
                total=len(self.trainning_generator),
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]',
                postfix={
                    'losses': loss_value
                }
        )

        # Loop over each epoch
        for i, (data, target) in progress_bar:
            # Update losses value
            loss_value = ((i * loss_value) + self.run_training_batch(data, target))/(i+1)
            progress_bar.set_postfix({'losses': loss_value})
            progress_bar.update(1)

        # Test validation
        self.run_validation(epoch=epoch, **kwargs)

    @timer
    def run_validation_batch(self, epoch, **kwargs) -> None:
        """
        Run validation

        :return: None
        """
        pass

    @timer
    def run_training_batch(
            self, data: torch.Tensor, target: torch.Tensor
    ) -> None:
        """
        Run training batch

        :param data: Input data
        :param target:
        :return: None
        """
        # Clear gradient
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(data)

        # Compute losses
        loss = self.loss_function(output, target)
        loss_value = loss.item()

        # Backward pass
        loss.backward()

        # Optimizer step
        self.optimizer.step()
        return loss_value
