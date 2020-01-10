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
from data_science_framework.pytorch_utils.losses import Loss
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
        self.loss_function = None
        self.optimizer = None
        self.validation_generator = None
        self.trainning_generator = None
        self.callbacks = []
        self.writer = None

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
        # Initialize callbacks'epoch
        for callback in self.callbacks:
            callback.on_epoch_start(epoch=epoch, model=self.model)

        # Initialize losses values
        loss_training = 0
        loss_validation = 0

        # Initialize training progressbar
        progress_bar = tqdm(
                enumerate(self.trainning_generator),
                desc='Train epoch {} / {}'.format(epoch + 1, self.nb_epochs),
                total=len(self.trainning_generator),
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]',
                postfix={
                    'losses': loss_training
                }
        )

        # Loop over each training batch
        for i, (data, target) in progress_bar:

            # Update losse training value
            loss_training = (
                (i * loss_training) + self.run_training_batch(
                    data, target, **kwargs
                )
            )/(i+1)
            progress_bar.set_postfix({'losses': loss_training})
            progress_bar.update(1)
        
        # Initialize validation progressbar
        progress_bar = tqdm(
                enumerate(self.validation_generator),
                desc='Validation {} / {}'.format(epoch + 1, self.nb_epochs),
                total=len(self.validation_generator),
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]',
                postfix={
                    'losses': loss_validation
                }
        )

        # Loop over each validation batch
        for i, (data, target) in progress_bar:

            # Update losse training value
            loss_validation = (
                (i * loss_validation) + self.run_validation_batch(
                    data, target, **kwargs
                )
            )/(i+1)
            progress_bar.set_postfix({'losses': loss_validation})
            progress_bar.update(1)

        # Initialize callbacks'epoch
        for callback in self.callbacks:
            callback.on_epoch_end(epoch=epoch, model=self.model)
        self.writer.add_scalars(
            'loss', {
                'training': loss_training,
                'validation': loss_validation,
            },
            epoch
        )

    @timer
    def run_validation_batch(
            self, data: torch.Tensor, target: torch.Tensor,
            **kwargs
        ) -> None:
        """
        Run validation

        :param data: Input data
        :param target: Targetted data
        :return: Value of the loss on the batch
        """
        # Disable gradient
        output = self.model(data)

        # Compute callbacks
        for callback in self.callbacks:
            callback(output, target, training=False)

        # Compute loss
        loss_value = self.loss_function(input=output, target=target).item()
        return loss_value

    @timer
    def run_training_batch(
            self, data: torch.Tensor, target: torch.Tensor,
            **kwargs
    ) -> float:
        """
        Run training batch

        :param data: Input data
        :param target: Targetted data
        :return: Value of the loss on the batch
        """
        # Clear gradient
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(data)

        # Compute callbacks
        for callback in self.callbacks:
            callback(output, target, training=True)

        # Compute losses
        loss = self.loss_function(input=output, target=target)
        loss_value = loss.item()

        # Backward pass
        loss.backward()

        # Optimizer step
        self.optimizer.step()

        return loss_value

    def set_optimizer(self, optimizer: Optimizer) -> None:
        """
        Set optimizer

        :return: None
        """
        self.optimizer = optimizer.get_torch()(self.model)

    def set_loss(self, loss: Loss) -> None:
        """
        Set optimizer

        :return: None
        """
        self.loss_function = loss.get_torch()
