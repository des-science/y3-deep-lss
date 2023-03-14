# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2023
Author: Arne Thomsen, Janis Fluri
"""

import tensorflow as tf
import os, atexit

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


def check_devices():
    """Logs the number of discovered CPUs and GPUs

    Returns:
        (int, int): CPU core count, GPU device count
    """
    try:
        n_cpus = len(os.sched_getaffinity(0))
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


def get_strategy(distributed, strategy_type="mirrored", cross_device_ops=tf.distribute.NcclAllReduce(num_packs=1)):
    """Sets up the tf.distribute.Strategy

    Args:
        distributed (bool): Whether to use a distribute strategy
        strategy_type (str): One of "mirrored" and "multi_mirrored". Determines the subclass of Strategy to be used.
        cross_device_ops (tf.distribute.CrossDeviceOps, optional): Cross-device reduction and broadcasting algorithms.
            Defaults to tf.distribute.NcclAllReduce(num_packs=1).

    Returns:
        tf.distribute.Strategy: The distribution strategy
    """
    n_gpus = len(tf.config.list_physical_devices("GPU"))

    if n_gpus > 1 and distributed:
        if strategy_type == "mirrored":
            strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)
            LOGGER.info(f"Training is distributed, using MirroredStrategy")

            # correct exit behavior as in https://github.com/tensorflow/tensorflow/issues/50487#issuecomment-997304668
            atexit.register(strategy._extended._collective_ops._pool.close)

            n_replicas = strategy.num_replicas_in_sync
            assert n_replicas == n_gpus

        elif strategy_type == "multi_mirrored":
            strategy = tf.distribute.MultiWorkerMirroredStrategy(cross_device_ops=cross_device_ops)
            LOGGER.info(f"Training is distributed, using MutliWorkerMirroredStrategy")

        else:
            raise NotImplementedError

    else:
        strategy = tf.distribute.get_strategy()
        LOGGER.warning(f"Training is not distributed, using the default strategy")

    return strategy


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
