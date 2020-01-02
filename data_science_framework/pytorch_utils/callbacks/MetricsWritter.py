"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-01-02

**Project** : data_science_framework

**Module that contains the codes that implements metric callback**
"""
from torch.utils.tensorboard import SummaryWriter
from .Callback import Callback
import torch


class MetricsWritter(Callback):
    """
    Class that implements 

    :param writer: Tensorboad summary writter file
    :param metrics: List of metrics
    """
    def __init__(self, writer: SummaryWriter, metrics: list = []) -> None:
        super(MetricsWritter, self).__init__(writer=writer)
        self.metrics = metrics
        self.on_epoch_start(0)

    def on_epoch_end(self, epoch: int):
        """
        Method called on epoch end
        
        :param epoch: Epoch value
        """

        # Save results
        print('\n\t- Save metrics results')
        self.save(epoch=epoch)

        # Loop over metrics
        for metric_object, acc_train, value_train, acc_val, value_val in self.metric_values:
            print('\t\t- Save {} : '.format(metric_object.name))
            self.writer.add_scalars(
                metric_object.name,
                {
                    'training': value_train / (acc_train + 0.0001),
                    'validation': value_val / (acc_train + 0.0001),
                },
                epoch
            )

    def on_epoch_start(self, epoch: int):
        """
        Method called on epoch start
        
        :param epoch: Epoch value
        """
        self.metric_values = list(zip(
            self.metrics,
            len(self.metrics) * [0],
            len(self.metrics) * [0],
            len(self.metrics) * [0],
            len(self.metrics) * [0],
        ))

    def __call__(
            self, output: torch.Tensor, target: torch.Tensor,
            training: bool = True
        ) -> None: 
        """
        Method call

        :param training: True if  metric is computed on test
        """
        # Loop over the metrics
        for i in range(len(self.metric_values)):

            # Compute metric
            acc_, value_ = self.metric_values[i][0].compute(
                output=output, target=target
            )

            # Accumulate results
            if training:
                self.metric_values[i] = (
                    self.metric_values[i][0],
                    self.metric_values[i][1] + acc_,
                    self.metric_values[i][2] + value_,
                    self.metric_values[i][3],
                    self.metric_values[i][4],
                )
            else:
                self.metric_values[i] = (
                    self.metric_values[i][0],
                    self.metric_values[i][1],
                    self.metric_values[i][2],
                    self.metric_values[i][3] + acc_,
                    self.metric_values[i][4] + value_,
                )

