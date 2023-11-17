# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created November 2023
Author: Arne Thomsen
"""

import horovod.tensorflow as hvd
import os

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


class NullContextManager:
    """A context manager that does nothing."""

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class HorovodInputContext:
    """Included for compatibility with tf.distribute.Strategy to shard a dataset across multiple input pipelines like
    in https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
    """

    def __init__(self):
        self.num_input_pipelines = hvd.size()
        self.input_pipeline_id = hvd.rank()


class HorovodClusterResolver:
    """Included for compatibility with tf.distribute.Strategy."""

    def __init__(self):
        self.task_id = hvd.rank()
        self.task_type = "worker"
        self.cluster_spec = None


class HorovodStrategy:
    """Wrapper class for Horovod that mimics the tf.distribute.Strategy interface for the most important methods and
    attributes. This allows to use the same code for both tf.distribute.Strategy and Horovod.
    """

    def __init__(self):
        hvd.init()
        self.num_replicas_in_sync = hvd.size()
        self.replica_id = hvd.rank()
        self.cluster_resolver = HorovodClusterResolver()

    def distribute_datasets_from_function(self, dataset_fn):
        return dataset_fn(HorovodInputContext())

    def scope(self):
        """Don't do anything. For the builtin strategies, this is important as the trainable variable have to be
        created within the context.
        """
        
        return NullContextManager()


def setup_horovod():
    """Perlmutter specific setup for Horovod.

    Returns:
        HorovodStrategy: Strategy compatible with distributed training on a single or multiple Perlmutter node(s).
    """

    n_tasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
    assert (
        n_tasks_per_node == 4
    ), f"On Perlmutter, n_tasks_per_node should be equal to gpus_per_node 4, but is {n_tasks_per_node}"

    gpus_per_task = int(os.environ["SLURM_GPUS_PER_TASK"])
    assert gpus_per_task == 1, f"Horovod expects a single GPU per task, but got {gpus_per_task}"

    LOGGER.warning(f"Training is distributed, using Horovod")
    return HorovodStrategy()
