"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-01-16

**Project** : data_science_framework

**Class that implements MetricsAnalyser**
"""
from data_science_framework.data_analyser.plotter import BoxPlotter
from data_science_framework.data_analyser.analyser import Analyser
from data_science_framework.pytorch_utils.metrics import MetricPerClass
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from typing import List
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import os


class MetricsAnalyser(Analyser):
    """
    Class that implements MetricsAnalyser

    :param writer: Tensorboad summary writter file
    :param plotter: Plots in use for the analyse
    :param save_path: Path to the save folder
    :param subset_name: Name of the datasubset
    :param threshold: Value of the threshold for each metric
    """
    def __init__(
            self, writer: SummaryWriter,
            save_path: str, subset_name: str,
            metrics: List[MetricPerClass],
            threshold: float = 0.5
        ) -> None:
        super().__init__(
            writer=writer, save_path=save_path,
            subset_name=subset_name
        )
        # Initialize attributes
        self.metrics = metrics
        self.plotter = [
            BoxPlotter(
                title='{} computed on {}'.format(
                    metric.name,
                    self.subset_name
                )
            )
            for metric in self.metrics
        ]
        self.dataframe = pd.DataFrame()
        self.threshold=threshold

        # Initialize metric threshold
        for metric in self.metrics:
            metric.threshold = self.threshold

    def __call__(
            self, output: torch.Tensor,
            target: torch.Tensor, **kwargs
        ) -> None:
        """__call__

        :param output: Output of the network
        :type output: torch.Tensor
        :param target: Targetted ground truth
        :type target: torch.Tensor
        :rtype: None
        """
        # Initialize row
        row_ = {}

        # Add meta data
        if 'meta' in kwargs.keys():
            row_.update(kwargs['meta'])

        # Loop over the metrics
        for metric in self.metrics:
            metric_values_ = metric.compute(
                output, target
            )
            for i, metric_value_ in enumerate(metric_values_):
                row_.update(
                    {
                        '{}_class_{}'.format(metric.name, i) : metric_value_
                    }
                )

        # Append row
        self.dataframe = self.dataframe.append(row_, ignore_index=True)

    def save_data(self) -> None:
        """save_data

        Saves the data to log files

        :rtype: None
        """
        # Save row data

        self.dataframe.to_csv(
            os.path.join(
                self.save_path, 'metrics_{}.csv'.format(
                    self.subset_name
                )
            ),
            index=False
        )

        # Generate human human_readable_dataframe
        human_readable_dataframe = pd.DataFrame()

        # Compute statistics and utils variables
        statistics = self.dataframe.describe()
        metrics_names = [metric.name for metric in self.metrics]
        nb_classes = statistics[
            [
                column
                for column in statistics.columns.to_list()
                if metrics_names[0] in column
            ]
        ].shape[1]

        # Reformat table
        for metric_name in metrics_names:
            row_ = {'metric': metric_name}
            for class_number in range(nb_classes):
                row_['class_{}'.format(class_number)] = '{} ± {}'.format(
                    statistics['{}_class_{}'.format(metric_name, class_number)]['mean'].__round__(3),
                    statistics['{}_class_{}'.format(metric_name, class_number)]['std'].__round__(3)

                )
            human_readable_dataframe = human_readable_dataframe.append(
                row_, ignore_index=True
            )
        human_readable_dataframe = human_readable_dataframe.set_index('metric')

        # Save reformatted data
        human_readable_dataframe.to_csv(
                os.path.join(
                self.save_path,
                'metrics_human_readable_{}.csv'.format(self.subset_name)
            )
        )

    def save_to_tensorboard(self) -> None:
        """save_to_tensorboard

        Save to tensorboard

        :rtype: None
        """
        for metric, plotter in zip(self.metrics, self.plotter):

            # Get dataframe with the correct columns
            dataframe_ = self.dataframe[
                [
                    column
                    for column in self.dataframe.columns.to_list()
                    if metric.name in column
                ]
            ]
            figure_ = plotter(dataframe_)

            # Save figure
            self.writer.add_figure(
                'test/{}'.format(plotter.title),
                figure_
            )

        # Save metrics
        metrics = {
            'metrics/{}_mean'.format(column): self.dataframe[column].\
                    mean().__round__(3)
            for column in list(self.dataframe)
            if '_class_' in column
        }
        metrics.update(
            {
                'metrics/{}_std'.format(column): self.dataframe[column].\
                        std().__round__(3)
                for column in list(self.dataframe)
                if '_class_' in column
            }
        )
        self.writer.add_hparams(
                {}, metrics
        )
