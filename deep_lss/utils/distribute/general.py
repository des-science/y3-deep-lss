# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2023
Author: Arne Thomsen

Utils for distributed training with tf.distribute and horovod.
"""

import tensorflow as tf
import os

from deep_lss.utils.distribute import tensorflow, horovod

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


def get_strategy(strategy_name=None):
    """Returns a tf.distribute.Strategy or (custom) HorovodStrategy instance.

    Args:
        strategy_name (str, optional): The distribution strategy to use. Valid values are 'mirrored',
            'multi_worker_mirrored' and 'horovod'. Defaults to None, then the builtin default strategy is used.

    Returns:
        tf.distribute.Strategy: The distribution strategy.
    """
    try:
        n_tasks = int(os.environ["SLURM_NTASKS"])
        LOGGER.info(f"Running on {n_tasks} tasks")
    except KeyError:
        LOGGER.info(f"Running locally as SLURM_NTASKS is not set")

    if strategy_name is None:
        strategy = tf.distribute.get_strategy()
        LOGGER.warning(f"Training is not distributed, using the default strategy")
    elif strategy_name == "mirrored":
        strategy = tensorflow.setup_tf_distribute_mirrored_strategy()
    elif strategy_name == "multi_worker_mirrored":
        strategy = tensorflow.setup_tf_distribute_multi_worker_mirrored_strategy()
    elif strategy_name == "horovod":
        strategy = horovod.setup_horovod()
    else:
        ValueError(f"Unknown distribution strategy {strategy_name}")

    return strategy


def check_devices():
    """Logs the number of discovered CPUs and GPUs

    Returns:
        (int, int): CPU core count, GPU device count
    """
    try:
        n_cpus = len(os.sched_getaffinity(0))
        if n_cpus != os.cpu_count():
            LOGGER.debug(
                f"len(os.sched_getaffinity(0)) = {len(os.sched_getaffinity(0))} and",
                f" os.cpu_count() = {os.cpu_count()} disagree",
            )
    except AttributeError:
        n_cpus = os.cpu_count()
    LOGGER.info(f"Running on {n_cpus} CPU cores")

    n_gpus = len(tf.config.list_physical_devices("GPU"))
    if n_gpus == 0:
        LOGGER.warning(f"No GPU discovered by TensorFlow, running on CPUs only")
    else:
        LOGGER.info(f"Running on {n_gpus} GPUs")

    try:
        n_gpus_cuda = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        assert (
            n_gpus == n_gpus_cuda
        ), f"The number of GPUs in TensorFlow {n_gpus} and CUDA {n_gpus_cuda} should be equal"
    except KeyError:
        LOGGER.warning(f"No CUDA enabled GPUs found")

    return n_cpus, n_gpus


def get_local_batch_size(strategy, global_batch_size):
    """Calculates the local (per replica) batch size given a strategy and global batch size

    Args:
        strategy (tf.distribute.Strategy): The instance of the strategy.
        global_batch_size (int): Batch size over all of the replicas.

    Raises:
        ValueError: If the global batch size is not divisible by the number of replicas

    Returns:
        int: The per replica batch size
    """
    n_replicas = strategy.num_replicas_in_sync

    # adjust the batch size to the strategy
    if global_batch_size % n_replicas == 0:
        local_batch_size = global_batch_size // n_replicas
        LOGGER.info(f"Using the local batch size {local_batch_size}")
    else:
        raise ValueError(
            f"The global batch size {global_batch_size} has to be divisible by the number of synced replicas {n_replicas}"
        )

    return local_batch_size


def get_global_batch_size(strategy, local_batch_size):
    """Calculates the global (accross all replicas) batch size given a strategy and local batch size

    Args:
        strategy (tf.distribute.Strategy): The instance of the strategy.
        local_batch_size (int): Batch size of a single replica.

    Returns:
        int: The global batch size over all replicas
    """
    n_replicas = strategy.num_replicas_in_sync

    global_batch_size = int(local_batch_size * n_replicas)
    LOGGER.info(f"Using the global batch size {global_batch_size}")

    return global_batch_size
