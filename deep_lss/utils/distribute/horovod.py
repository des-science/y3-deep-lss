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
        """See https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function"""
        return dataset_fn(HorovodInputContext())

    def scope(self):
        """Don't do anything. For the builtin strategies, this is important as the trainable variables have to be
        created within the context.
        """

        return NullContextManager()

    def gather(self, tensor, axis=0):
        """For compatibility with tf.distribute.Strategy.

        Args:
            tensor (tf.Tensor): The tensor to gather, where every worker has its own.
            axis (int, optional): The axis along which to gather. Defaults to 0, which is the only value that is
                generally allowed with Horovod.

        Returns:
            tf.Tensor: A tensor of the same type as tensor, concatenated on dimension zero across all processes.
                The shape is identical to the input shape, except for the first dimension, which may be greater and is
                the sum of all first dimensions of the tensors in different Horovod processes.
        """

        # NOTE one could implement the gather along a different axis using transpose twice. But this is only possible
        # if the number of dimensions of the tensor is known, so should be done outside this class.
        assert axis == 0, f"Horovod only supports gathering along axis 0, but got {axis}"

        return hvd.allgather(tensor)

    def run(self, fn, args=(), kwargs={}):
        """For compatibility with tf.distribute.Strategy.

        Args:
            fn (function): The function to evaluate on the different workers
            args (tuple, optional): Ordered arguments. Defaults to ().
            kwargs (dict, optional): Keyqord arguments. Defaults to {}.

        Returns:
            tf.Tensor: The result of fn(*args, **kwargs).
        """
        
        return fn(*args, **kwargs)


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
