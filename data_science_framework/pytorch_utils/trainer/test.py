"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-17

**Project** : baseline_unet

**File that tests codes of trainer module**
"""
from data_science_framework.pytorch_utils.models.Unet import Unet
from data_science_framework.pytorch_utils.optimizer import AdadeltaOptimizer
from data_science_framework.pytorch_utils.trainer import Trainer
from data_science_framework.pytorch_utils.losses import BinaryCrossEntropyLoss
from data_science_framework.pytorch_utils.trainer.VanillaTrainer import VanillaTrainer
from data_science_framework.pytorch_utils.callbacks import MetricsWritter
from data_science_framework.pytorch_utils.metrics import Metric
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from data_science_framework.scripting.test_manager import set_test_folders
from data_science_framework.settings import TEST_ROOT
from data_science_framework.pytorch_utils.trainer import MODULE
import numpy as np


def test_Trainer() -> None:
    """
    Function that tests Trainer

    :return: None
    """
    # Test set object attributes
    trainer = Trainer()
    trainer.test = None
    trainer.set_objects_attributes(test=['a', 'b'])
    assert tuple(trainer.test) == ('a', 'b')


@set_test_folders(
    output_root=TEST_ROOT,
    current_module=MODULE
)
def test_VanillaTrainer_run(output_folder: str) -> None:
    """
    Function that tests VanillaTrainer run method

    :param output_folder: Path to the output folder
    :return: None
    """
    trainer = VanillaTrainer()

    # Initialize trainer
    trainer = VanillaTrainer(nb_epochs=10)
    trainer.run_training_batch = lambda data, target: data - target
    trainer.run_validation_batch = lambda data, target: data - target
    trainer.set_objects_attributes(
        writer=SummaryWriter(log_dir=output_folder)
    )

    # Reset trainning generator
    trainer.trainning_generator = [(2*i, i)for i in range(10)]
    trainer.validation_generator = [(2*i, i)for i in range(10)]
    
    # Launch function
    trainer.run()


@set_test_folders(
    output_root=TEST_ROOT,
    current_module=MODULE
)
def test_VanillaTrainer_run_epoch(output_folder: str) -> None:
    """
    Function that tests VanillaTrainer run_epoch method

    :param output_folder: Path to the output folder
    :return: None
    """
    # Initialize trainer
    trainer = VanillaTrainer(nb_epochs=1)
    trainer.run_training_batch = lambda data, target: data - target
    trainer.run_validation_batch = lambda data, target: data - target
    trainer.set_objects_attributes(
        writer=SummaryWriter(log_dir=output_folder)
    )

    # Reset trainning generator
    trainer.trainning_generator = [(2*i, i)for i in range(10)]
    trainer.validation_generator = [(2*i, i)for i in range(10)]

    trainer.run_epoch(epoch=5)


def test_VanillaRunner_set_optimizer() -> None:
    """
    Function that tests VanillaRunner_set_optimizer

    :return: None
    """
    # Initialize trainer
    trainer = VanillaTrainer(nb_epochs=1)
    trainer.set_objects_attributes(model=Unet())
    try:
        trainer.set_optimizer(AdadeltaOptimizer())
    except:
        assert False


def test_VanillaRunner_run_training_batch() -> None:
    """
    Function that tests VanillaRunner_run_training_batch

    :return: None
    """
    # Initialize arrays
    data_array = np.zeros((2, 3, 16, 16, 16)) + 0.1
    target_array = np.zeros((2, 4, 16, 16, 16)) 
    target_array[:, 1, :, :, :] = 1

    # Initialize torch tensors
    data_torch = torch.tensor(
            data_array, dtype=torch.float32
    ).to('cpu')
    target_torch = torch.tensor(
            target_array, dtype=torch.float32
    ).to('cpu')

    # Initialize network
    model = Unet(in_channels=3, out_channels=4) 
    # Initialize Trainer
    vanilla_trainer = VanillaTrainer()
    vanilla_trainer.set_objects_attributes(
            model=model,
    )
    vanilla_trainer.set_optimizer(
        optimizer=AdadeltaOptimizer()
    )
    vanilla_trainer.set_loss(
        loss=BinaryCrossEntropyLoss()
    )

    # Apply fonction
    loss_value = vanilla_trainer.run_training_batch(
        data=data_torch, target=target_torch
    )


def test_VanillaTrainer_set_loss() -> None:
    """
    Function that tests VanillaTrainer_set_loss

    :return: None
    """
    # Initialize trainer
    trainer = VanillaTrainer(nb_epochs=1)
    trainer.set_objects_attributes(model=Unet())
    try:
        trainer.set_loss(loss=BinaryCrossEntropyLoss())
    except:
        assert False


@set_test_folders(
    output_root=TEST_ROOT,
    current_module=MODULE
)
def test_VanillaRunner_run_validation_batch(output_folder: str) -> None:
    """
    Function that tests VanillaRunner_run_validation_batch

    :param output_folder: Path to the output folder
    :return: None
    """
    # Initialize arrays
    data_array = np.zeros((2, 3, 16, 16, 16)) + 0.1
    target_array = np.zeros((2, 4, 16, 16, 16)) 
    target_array[:, 1, :, :, :] = 1

    # Initialize torch tensors
    data_torch = torch.tensor(
            data_array, dtype=torch.float32
    ).to('cpu')
    target_torch = torch.tensor(
            target_array, dtype=torch.float32
    ).to('cpu')

    # Initialize network
    model = Unet(in_channels=3, out_channels=4) 

    # Create vanilla metric
    class VanillaMetric(Metric):
        def __init__(self, name):
            self.name = name
        def compute(self, output, target):
            return 1, np.random.rand()

    # Initialize metrics writter
    metrics_callback = MetricsWritter(
        writer=SummaryWriter(log_dir=output_folder),
        metrics=[
            VanillaMetric('metric_1'),
            VanillaMetric('metric_2')
        ]
    )

    # Initialize Trainer
    vanilla_trainer = VanillaTrainer()
    vanilla_trainer.set_objects_attributes(
            model=model,
            callbacks=[metrics_callback]
    )
    vanilla_trainer.set_optimizer(
        optimizer=AdadeltaOptimizer()
    )
    vanilla_trainer.set_loss(
        loss=BinaryCrossEntropyLoss()
    )

    # Apply fonction
    loss_value = vanilla_trainer.run_validation_batch(
        data=data_torch, target=target_torch
    )

