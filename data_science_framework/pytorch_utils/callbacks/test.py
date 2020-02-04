"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-01-02

**Project** : data_science_framework

**Module that contains the codes that implements callbacks**
"""
import os
from torch.utils.tensorboard import SummaryWriter
from data_science_framework.pytorch_utils.callbacks import Callback, MetricsWritter, ModelCheckpoint,\
        ModelPlotter, DataDisplayer, ConfusionMatrixCallback
from data_science_framework.pytorch_utils.models import ApproxMCUnet2Axis
from data_science_framework.pytorch_utils.callbacks import MODULE
from data_science_framework.scripting.test_manager import set_test_folders
from data_science_framework.settings import TEST_ROOT
from data_science_framework.pytorch_utils.metrics import Metric
import numpy as np
import torch


@set_test_folders(
    output_root=TEST_ROOT,
    current_module=MODULE
)
def test_Callback(output_folder: str) -> None:
    """
    Function that tests Callback

    :param output_folder: Path to the output folder
    :return: None
    """
    # Initialize Callback
    callback = Callback(
        writer=SummaryWriter(log_dir=output_folder)
    )
    assert type(callback.writer) == SummaryWriter


@set_test_folders(
    output_root=TEST_ROOT,
    current_module=MODULE
)
def test_MetricsWritter(output_folder: str) -> None:
    """
    Function that tests MetricsWritter

    :param output_folder: Path to the output folder
    :return: None
    """
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
            VanillaMetric('metric_2'),
        ]
    )

    # Test on epoch start
    metrics_callback.on_epoch_start(0, None)
    assert len(metrics_callback.metric_values) == 2
    for _, acc_train, val_train, acc_val, val_val in metrics_callback.metric_values:
        assert acc_train == 0
        assert val_train == 0
        assert acc_val == 0
        assert val_val == 0

    # Test call method
    metrics_callback(3, 4, training=True)
    assert metrics_callback.metric_values[0][1] == 1
    assert metrics_callback.metric_values[0][2] != 0
    assert metrics_callback.metric_values[0][3] == 0
    assert metrics_callback.metric_values[0][4] == 0

    metrics_callback.on_epoch_start(0, None)
    metrics_callback(3, 4, training=False)
    assert metrics_callback.metric_values[0][1] == 0
    assert metrics_callback.metric_values[0][2] == 0
    assert metrics_callback.metric_values[0][3] == 1
    assert metrics_callback.metric_values[0][4] != 0

    # Test on epoch end
    for epoch in range(10):
        metrics_callback.on_epoch_start(epoch, None)
        for i in range(epoch):
            metrics_callback(3, 4, training=True)
            metrics_callback(3, 4, training=False)
        metrics_callback.on_epoch_end(epoch, None)


@set_test_folders(
    output_root=TEST_ROOT,
    current_module=MODULE
)
def test_ModelCheckpoint(output_folder: str) -> None:
    """
    Function that tests ModelCheckpoint

    :param output_folder: Path to the output folder
    :return: None
    """
    # Create vanilla metric
    class VanillaMetric(Metric):
        def __init__(self, name):
            self.name = name
        def compute(self, output, target):
            return 1, np.random.rand()

    # Initialize model checkpoint
    model_checkpoint = ModelCheckpoint(
        writer=SummaryWriter(log_dir=output_folder),
        metric=VanillaMetric('loss'),
        save_folder=output_folder,
        metric_to_minimize=False
    )

    # Test on epoch start
    model_checkpoint.on_epoch_start(0, ApproxMCUnet2Axis())
    assert len(list(model_checkpoint.metric_values)) == 3
    assert os.path.isfile(
        os.path.join(output_folder, 'model.pkl')
    )

    # Test call method
    model_checkpoint(3, 4, training=True)
    assert model_checkpoint.metric_values[1] == 0
    assert model_checkpoint.metric_values[2] == 0

    model_checkpoint.on_epoch_start(0, None)
    model_checkpoint(3, 4, training=False)
    assert model_checkpoint.metric_values[1] != 0
    assert model_checkpoint.metric_values[2] != 0

    # Test on epoch end
    model = ApproxMCUnet2Axis()
    for epoch in range(10):
        model_checkpoint.on_epoch_start(epoch, model)
        for i in range(epoch):
            model_checkpoint(3, 4, training=False)
        model_checkpoint.on_epoch_end(epoch, model)


@set_test_folders(
    output_root=TEST_ROOT,
    current_module=MODULE
)
def test_ModelPlotter(output_folder: str) -> None:
    """
    Function that tests ModelPlotter

    :param output_folder: Path to the output folder
    :return: None
    """
    # Initialize callback
    model = ApproxMCUnet2Axis(
        in_channels=5,
        out_channels=3,
        depth=3,
        n_features=2,
        kernel_size=3,
        pool_size=2,
        padding=1,
    )

    ModelPlotter(
        writer=SummaryWriter(log_dir=output_folder),
        model=model
    )


@set_test_folders(
    output_root=TEST_ROOT,
    current_module=MODULE
)
def test_DataDisplayer(output_folder: str) -> None:
    """
    Function that tests DataDisplayer

    :param output_folder: Path to the output folder
    :return: None
    """
    # Initialize model checkpoint
    data_displayer = DataDisplayer(
        writer=SummaryWriter(log_dir=output_folder),
    )
    assert data_displayer.saved_image_training == False
    assert data_displayer.saved_image_validation == False

    # Test on epoch start
    data_displayer.on_epoch_start(0, None)
    assert data_displayer.saved_image_training == False
    assert data_displayer.saved_image_validation == False

    # Test call method
    for training_, saved_image_training_, saved_image_validation_ in zip(
        [True, True, False, False],
        [True, True, True, True],
        [False, False, True, True],
    ):
        output_ = torch.rand((1, 2, 3, 4, 5))
        target_ = torch.rand((1, 2, 3, 4, 5))
        data_displayer(
            output=output_,
            target=target_,
            training=training_
        )
        assert data_displayer.saved_image_training == saved_image_training_
        assert data_displayer.saved_image_validation == saved_image_validation_


@set_test_folders(
    output_root=TEST_ROOT,
    current_module=MODULE
)
def test_ConfusionMatrixCallback(output_folder: str) -> None:
    """
    Function that tests ConfusionMatrixCallback

    :param output_folder: Path to the output folder
    :return: None
    """
    # Initialize model checkpoint
    confusion_matrix_callback = ConfusionMatrixCallback(
        writer=SummaryWriter(log_dir=output_folder),
    )
    assert confusion_matrix_callback.nb_classes == None

    # Test on epoch start
    confusion_matrix_callback.on_epoch_start(0, None)
    assert len(
        confusion_matrix_callback.training_confusion_matrices
    ) == 0
    assert len(
        confusion_matrix_callback.validation_confusion_matrices
    ) == 0

    # Test call method
    for training_, training_length_, validation_length_ in zip(
        [True, False],
        [2, 2],
        [0, 2]
    ):
        for i in range(2):
            output_ = torch.rand((2, 10, 3, 4, 5))
            target_ = torch.rand((2, 10, 3, 4, 5))
            if i == 1:
                output_[:, -1 , :, :] = 0
                target_[:, -1 , :, :] = 0
            confusion_matrix_callback(
                output=output_,
                target=target_,
                training=training_
            )
        assert len(
            confusion_matrix_callback.training_confusion_matrices
        ) == training_length_
        assert len(
            confusion_matrix_callback.validation_confusion_matrices
        ) == validation_length_

    # Test on epoch end
    confusion_matrix_callback.on_epoch_end(
        epoch=0,
        model=None
    )

