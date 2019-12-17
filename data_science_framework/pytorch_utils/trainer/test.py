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


