"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-01-02

**Project** : data_science_framework

**Module that contains the codes that implements callbacks**
"""
from torch.utils.tensorboard import SummaryWriter
from data_science_framework.pytorch_utils.callbacks import Callback, MetricsWritter
from data_science_framework.pytorch_utils import MODULE
from data_science_framework.scripting.test_manager import set_test_folders
from data_science_framework.settings import TEST_ROOT, RESSOURCES_ROOT
from data_science_framework.pytorch_utils.metrics import Metric
import numpy as np


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
    metrics_callback.on_epoch_start(0)
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

    metrics_callback.on_epoch_start(0)
    metrics_callback(3, 4, training=False)
    assert metrics_callback.metric_values[0][1] == 0
    assert metrics_callback.metric_values[0][2] == 0
    assert metrics_callback.metric_values[0][3] == 1
    assert metrics_callback.metric_values[0][4] != 0

    # Test on epoch end
    for epoch in range(10):
        metrics_callback.on_epoch_start(epoch)
        for i in range(epoch):
            metrics_callback(3, 4, training=True)
            metrics_callback(3, 4, training=False)
        metrics_callback.on_epoch_end(epoch)
