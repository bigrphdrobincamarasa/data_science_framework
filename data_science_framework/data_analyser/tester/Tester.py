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
from data_science_framework.data_analyser.analyser import Analyser
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from typing import List
import torch


class Tester(object):
    """
    Class that implements Tester

    :param result_folder: Path to the test folder
    :param writer: Tensorboad summary writer file
    :param dataset: Subset of the dataset
    :param analysers: List of analysis launched on the dataset
    :param model: Tested model
    """
    def __init__(
            self, result_folder: str, writer: SummaryWriter,
            dataset: Dataset, analysers : List[Analyser],
            model: torch.nn.Module
        ) -> None:
        self.result_folder = result_folder
        self.writer = writer
        self.dataset = dataset
        self.model = model
        self.analysers = analysers

    def __call__(self) -> None:
        """__call__

        Run analysis on the dataset object

        :rtype: None
        """
        # Apply analysis
        with torch.no_grad():
            for input, target, meta in self.dataset:
                for analyser in self.analysers:
                    analyser(
                        self.model(input), target, meta=meta
                    )

        # Save results
        for analyser in self.analysers:
            analyser.save_data()
            analyser.save_to_tensorboard()

