# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2023
Author: Arne Thomsen, Janis Fluri
"""

import tensorflow as tf
import os, atexit, json

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


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


def get_strategy(distributed):
    """Sets up the tf.distribute.Strategy

    Args:
        distributed (bool): Whether to use a distribute strategy
        strategy_type (str): One of "mirrored" and "multi_mirrored". Determines the subclass of Strategy to be used.
        cross_device_ops (tf.distribute.CrossDeviceOps, optional): Cross-device reduction and broadcasting algorithms.
            Defaults to tf.distribute.NcclAllReduce(num_packs=1).

    Returns:
        tf.distribute.Strategy: The distribution strategy
    """

    if distributed:
        try:
            n_tasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
            gpus_per_task = int(os.environ["SLURM_GPUS_PER_TASK"])
        except KeyError:
            n_tasks_per_node = None
            gpus_per_task = None

        # MultiWorkerMirroredStrategy
        if n_tasks_per_node == 4 and gpus_per_task == 1:
            # NOTE this only works for the setup where a node has four GPUs and the MultiWorkerStrategy is run on that
            # single node, where each GPU is a worker. For this, the .sh slurm submission script must include
            # --nodes=1 --gpus-per-node=4 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=32

            os.environ["TF_CONFIG"] = json.dumps(
                {
                    "cluster": {
                        "worker": ["localhost:12345", "localhost:12346", "localhost:12347", "localhost:12348"]
                    },
                    "task": {"type": "worker", "index": int(os.environ["SLURM_LOCALID"])},
                }
            )
            LOGGER.debug(os.environ["TF_CONFIG"])

            communication_options = tf.distribute.experimental.CommunicationOptions(
                implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
                # implementation=tf.distribute.experimental.CommunicationImplementation.RING
            )

            strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)
            LOGGER.info(f"Training is distributed, using MultiWorkerMirroredStrategy")

        # MirroredStrategy
        else:
            cross_device_ops = tf.distribute.NcclAllReduce(num_packs=1)
            # cross_device_ops = tf.distribute.HierarchicalCopyAllReduce(num_packs=1)
            # cross_device_ops = tf.distribute.ReductionToOneDevice(num_packs=1)

            strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)
            LOGGER.info(f"Training is distributed, using MirroredStrategy")

            # correct exit behavior as in https://github.com/tensorflow/tensorflow/issues/50487#issuecomment-997304668
            atexit.register(strategy._extended._collective_ops._pool.close)

            n_replicas = strategy.num_replicas_in_sync
            n_gpus = len(tf.config.list_physical_devices("GPU"))
            assert n_replicas == n_gpus

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
