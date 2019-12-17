"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-12-17

**Project** : baseline_unet

**File that tests codes of trainer module**
"""
from data_science_framework.pytorch_utils.trainer import Trainer

from data_science_framework.pytorch_utils.trainer.VanillaTrainer import VanillaTrainer
import time


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


def test_VanillaTrainer_run() -> None:
    """
    Function that tests VanillaTrainer run method

    :return: None
    """
    trainer = VanillaTrainer()
    trainer.run_epoch = lambda epoch, **kwargs: None
    try:
        trainer.run()
    except:
        assert False


def test_VanillaTrainer_run_epoch() -> None:
    """
    Function that tests VanillaTrainer run_epoch method

    :return: None
    """
    # Initialize trainer
    trainer = VanillaTrainer(nb_epochs=1)
    trainer.run_training_batch = lambda data, target: data - target

    # Reset trainning generator
    trainer.trainning_generator = [(2*i, i)for i in range(10)]
    trainer.run_validation = lambda epoch: time.sleep(0.8)
    trainer.run_epoch(epoch=5)
    assert True
