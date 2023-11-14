# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2023
Author: Arne Thomsen, Janis Fluri
"""

import tensorflow as tf
import os, atexit, json, re

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
        distributed (bool): Whether to use a distribute strategy.

    Returns:
        tf.distribute.Strategy: The distribution strategy.
    """

    if distributed:
        try:
            # equal to n_nodes * n_tasks_per_node
            n_tasks = int(os.environ["SLURM_NTASKS"])
            # equal to n_gpus_per_node
            n_tasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
            # always set to 1
            gpus_per_task = int(os.environ["SLURM_GPUS_PER_TASK"])
            # in [0, n_nodes)
            node_id = int(os.environ["SLURM_NODEID"])
            # in [0, n_tasks_per_node)
            task_id = int(os.environ["SLURM_LOCALID"])
            # has the format 'nid[001013,001016]' for example, where $HOSTNAME = nid001013 for the first node
            nodelist = os.environ["SLURM_NODELIST"]

        except KeyError:
            LOGGER.warning(
                f"One of the environmental variables couldn't be retrieved, falling back to MirroredStrategy"
            )
            n_tasks_per_node = None
            gpus_per_task = None

        # MultiWorkerMirroredStrategy on Perlmutter where every node has 4 GPUs. The .sh slurm submission script must
        # include --nodes=n --gpus-per-node=4 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=32
        if n_tasks_per_node == 4 and gpus_per_task == 1:
            assert n_tasks < 100, f"n_tasks {n_tasks} is too large for the port numbers"

            # \d+ matches one or more digits
            pattern = re.compile(r"\d+")
            node_names = pattern.findall(nodelist)
            # repeat every list element n_tasks_per_node times
            node_names = [name for name in node_names for _ in range(n_tasks_per_node)]
            # generate unique addresses for each task (the format 123{i:02} is arbitrary)
            worker_ports = [f"123{i:02}" for i in range(n_tasks)]
            # combine into a list of the form 'nid001234:12345'
            workers = [f"nid{name}:{port}" for name, port in zip(node_names, worker_ports)]

            # in [0, n_tasks)
            index = n_tasks_per_node * node_id + task_id

            os.environ["TF_CONFIG"] = json.dumps(
                {
                    "cluster": {"worker": workers},
                    "task": {"type": "worker", "index": index},
                }
            )
            LOGGER.info(f"TF_CONFIG = " + os.environ["TF_CONFIG"])

            communication_options = tf.distribute.experimental.CommunicationOptions(
                implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
                # implementation=tf.distribute.experimental.CommunicationImplementation.RING
            )

            strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)
            LOGGER.warning(f"Training is distributed, using the MultiWorkerMirroredStrategy with {n_tasks} workers")

        # MirroredStrategy
        else:
            cross_device_ops = tf.distribute.NcclAllReduce(num_packs=1)
            # cross_device_ops = tf.distribute.HierarchicalCopyAllReduce(num_packs=1)
            # cross_device_ops = tf.distribute.ReductionToOneDevice(num_packs=1)

            strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)

            # correct exit behavior as in https://github.com/tensorflow/tensorflow/issues/50487#issuecomment-997304668
            atexit.register(strategy._extended._collective_ops._pool.close)

            n_replicas = strategy.num_replicas_in_sync
            n_gpus = len(tf.config.list_physical_devices("GPU"))
            assert n_replicas == n_gpus

            LOGGER.warning(f"Training is distributed, using the MirroredStrategy")

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
