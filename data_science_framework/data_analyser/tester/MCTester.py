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
from data_science_framework.pytorch_utils.models import MCUnet
from data_science_framework.data_analyser.tester import Tester
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from typing import List
import torch


class MCTester(Tester):
    """
    Class that implements MCTester

    :param result_folder: Path to the test folder
    :param writer: Tensorboad summary writer file
    :param dataset: Subset of the dataset
    :param analysers: List of analysis launched on the dataset
    :param mc_analyser: List of analysis launched on the dataset
    :param model: Tested model
    """
    def __init__(
            self, result_folder: str, writer: SummaryWriter,
            dataset: Dataset, analysers : List[Analyser],
            mc_analysers : List[Analyser],
            model: MCUnet
        ) -> None:
        super(MCTester, self).__init__(
            result_folder=result_folder, writer=writer,
            dataset=dataset, analysers=analysers,
            model=model
        )
        self.mc_analysers=mc_analysers

    def __call__(self) -> None:
        """__call__

        Run analysis on the dataset object

        :rtype: None
        """
        # Apply analysis
        with torch.no_grad():
            for input, target, meta in self.dataset:
                output = self.model.mc_forward(input)
                for analyser in self.analysers:
                    analyser(
                        output.mean(0),
                        target, meta=meta
                    )
                for mc_analyser in self.mc_analysers:
                    mc_analyser(output, target, meta=meta)

        # Save results
        for analyser in self.analysers + self.mc_analysers:
            analyser.save_data()
            analyser.save_to_tensorboard()

