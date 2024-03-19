# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created November 2023
Author: Arne Thomsen
"""

import tensorflow as tf
import os, atexit, json, re

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


def setup_tf_distribute_mirrored_strategy():
    """Perlmutter specific setup for tf.distribute.MirroredStrategy.

    Returns:
        tf.distribute.MirroredStrategy: To be used for multiple GPUs on a single node.
    """

    cross_device_ops = tf.distribute.NcclAllReduce(num_packs=1)
    # cross_device_ops = tf.distribute.HierarchicalCopyAllReduce(num_packs=1)
    # cross_device_ops = tf.distribute.ReductionToOneDevice(num_packs=1)

    strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)

    # correct exit behavior as in https://github.com/tensorflow/tensorflow/issues/50487#issuecomment-997304668
    # atexit.register(strategy._extended._collective_ops._pool.close)

    n_replicas = strategy.num_replicas_in_sync
    n_gpus = len(tf.config.list_physical_devices("GPU"))
    assert n_replicas == n_gpus

    LOGGER.warning(f"Training is distributed, using the MirroredStrategy")
    return strategy


def setup_tf_distribute_multi_worker_mirrored_strategy():
    """Perlmutter specific setup for tf.distribute.MultiWorkerMirroredStrategy.

    Returns:
        tf.distribute.MirroredStrategy: To be used for multiple GPUs on a single or multiple node(s).
    """

    n_tasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
    assert (
        n_tasks_per_node == 4
    ), f"On Perlmutter, n_tasks_per_node should be equal to gpus_per_node 4, but is {n_tasks_per_node}"

    communication_options = tf.distribute.experimental.CommunicationOptions(
        # RING would also be possible, but NCCL is faster
        implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
    )

    port_base = 8888

    cluster_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(
        port_base=port_base, gpus_per_node=4, gpus_per_task=1, tasks_per_node=4
    )

    # NOTE The following commented code doesn't work since due to a suspected bug, when --nodes>1, the leading
    # zeros in the node names are removed, e.g. for two nodes on Perlmutter:
    # ['nid1444:8888', 'nid1444:8889', 'nid1444:8890', 'nid1444:8891', 'nid1445:8892', 'nid1445:8893', 'nid1445:8894', 'nid1445:8895']
    # instead of
    # ['nid001444:8888', 'nid001444:8889', 'nid001444:8890', 'nid001444:8891', 'nid001445:8892', 'nid001445:8893', 'nid001445:8894', 'nid001445:8895']
    # tf_config = {
    #     "cluster": {"worker": cluster_resolver.cluster_spec().as_dict()["worker"]},
    #     "task": {"type": cluster_resolver.task_type, "index": cluster_resolver.task_id},
    # }

    # this function does not remove the leading zeros
    tf_config = _get_handcrafted_tf_config(port_base=port_base)
    os.environ["TF_CONFIG"] = json.dumps(tf_config)
    LOGGER.info(f"tf_config = {tf_config}")

    strategy = tf.distribute.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver,
        communication_options=communication_options,
    )

    LOGGER.warning(f"Training is distributed, using the MultiWorkerMirroredStrategy")
    return strategy


def _get_handcrafted_tf_config(port_base=12345):
    """Manually constructs a tf_config dictionary for the MultiWorkerMirroredStrategy from the slurm environmental
    variables. This is obsolete since tf.distribute.cluster_resolver.SlurmClusterResolver can be used instead.

    Args:
        port_base (int, optional): The first port number to start with for processes on a node. Defaults to 12345.

    Raises:
        KeyError: If one of the necessary slurm environmental variables is not set.

    Returns:
        dict: The tf_config dictionary.
    """
    try:
        # equal to n_nodes * n_tasks_per_node
        n_tasks = int(os.environ["SLURM_NTASKS"])
        # equal to n_gpus_per_node
        n_tasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
        # in [0, n_nodes)
        node_id = int(os.environ["SLURM_NODEID"])
        # in [0, n_tasks_per_node)
        task_id = int(os.environ["SLURM_LOCALID"])
        # has the format 'nid[001013,001016]' for example, where $HOSTNAME = nid001013 for the first node
        nodelist = os.environ["SLURM_NODELIST"]
    except KeyError:
        LOGGER.warning(
            f"One of the slurm environmental variables couldn't be retrieved, can't use the MultiWorkerMirroredStrategy "
            f"like this"
        )

        raise KeyError

    # \d+ matches one or more digits
    pattern = re.compile(r"\d+")
    node_names = pattern.findall(nodelist)
    # repeat every list element n_tasks_per_node times
    node_names = [name for name in node_names for _ in range(n_tasks_per_node)]
    # generate unique addresses for each task (the format 123{i:02} is arbitrary)
    worker_ports = [str(port_base + i) for i in range(n_tasks)]
    # combine into a list of the form 'nid001234:12345'
    workers = [f"nid{name}:{port}" for name, port in zip(node_names, worker_ports)]

    # in [0, n_tasks)
    index = n_tasks_per_node * node_id + task_id

    tf_config = {
        "cluster": {"worker": workers},
        "task": {"type": "worker", "index": index},
    }

    return tf_config
