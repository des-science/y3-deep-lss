# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2023
Author: Arne Thomsen

Evaluate the DeepSphere graph neural networks on the CosmoGrid
"""

import numpy as np
import tensorflow as tf
import os, warnings, h5py, math, logging

from msfm.fiducial_pipeline import FiducialPipeline
from msfm.grid_pipeline import GridPipeline
from msfm.utils import logger

from deep_lss.utils import distribute

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)

# suppress a specific warning
logging.getLogger("tensorflow").addFilter(
    lambda record: "gather/all_gather with NCCL or HierarchicalCopy is not supported" not in record.getMessage()
)


def _get_out_file(dir_out, label):
    if label is None:
        out_file = f"preds.h5"
    else:
        out_file = f"preds_{label}.h5"

    return os.path.join(dir_out, out_file)


def _stack_grid_cosmos(tensors, sorted_indices, n_examples_per_cosmo):
    """Reshapes the batched evaluations into the correct shape.

    Args:
        tensors (list): List of tensors, where axis 0 of each element is the global batch size and len(tensors) is
            equal to the number of batches.
        sorted_indices (tf.constant): Index tensor coming from the Sobol indices by which the tensor is sorted
        n_examples_per_cosmo (int): How many example footprints there are per cosmology.

    Returns:
        tensors: Of shape (n_cosmos, n_examples_per_cosmo, None), where the None dimension is determined by the last
            axes of the input
    """
    # concatenate all of the cosmologies into the first axis, shape (n_cosmos * n_examples_per_cosmo, None)
    tensors = tf.concat(tensors, axis=0)
    # instead of numpy fancy indexing
    tensors = tf.gather(tensors, sorted_indices)
    # split according to the cosmology, list of len n_cosmos with elements of shape (n_examples_per_cosmo, None)
    tensors = tf.split(tensors, tensors.shape[0] // n_examples_per_cosmo)
    # stack the cosmologies into the 0th axis, shape (n_cosmos, n_examples_per_cosmo, None)
    tensors = tf.stack(tensors, axis=0)

    return tensors


def _remove_example_axis(array):
    """Takes in a tensor of shape (n_cosmos, n_examples_per_cosmo, None) or (n_cosmos, n_examples_per_cosmo) and checks
    whether the value along axis 1 is constant to remove that redundant axis.

    Args:
        tensor (np.ndarray): Shape (n_cosmos, n_examples_per_cosmo, None)

    Raises:
        RuntimeError: If the values along the axis of length n_examples_per_cosmo are not all equal

    returns:
        array: Shape (n_cosmos, None), where the redundancy has been removed
    """
    # double check that the cosmologies are sorted correctly and remove the redundant axis
    if np.all([np.equal(array[:, i], array[:, i + 1]) for i in range(array.shape[1] - 1)]):
        array = array[:, 0]
    else:
        raise RuntimeError(f"The cosmologies are not sorted correctly")

    return array


def evaluate_grid(
    model,
    strategy,
    tfr_pattern,
    msfm_conf,
    dlss_conf,
    net_conf,
    dir_out,
    file_label=None,
):
    """Evaluate the model on the grid part of the CosmoGrid.

    Args:
        model (DeltaLossModel): Model to be evaluated.
        strategy (tf.distribute.Strategy): Distribution strategy instance within which the model has been created. This
            is used to distribute the dataset.
        tfr_pattern (str): Glob pattern of the .tfrecord files containing the data.
        msfm_conf (dict): Configuration file of the msfm pipeline.
        net_conf (dict): Configuration file of the specific model.
        dir_out (str): Output directory, this is where the evaluations will be saved.
        file_label (str, optional): Optional suffix to append to the output file names. Defaults to None.
    """
    print("\n")
    LOGGER.info(f"Starting evaluation of the grid")

    dset_kwargs = net_conf["dset"]["eval"]["grid"]

    # pipeline constants
    n_cosmos = msfm_conf["analysis"]["grid"]["n_cosmos"]
    n_patches = msfm_conf["analysis"]["n_patches"]
    n_perms_per_cosmo = msfm_conf["analysis"]["grid"]["n_perms_per_cosmo"]
    n_examples_per_cosmo = n_patches * n_perms_per_cosmo

    # multiple shape and poisson noise realizations
    n_examples_per_cosmo *= dset_kwargs["n_noise"]

    n_examples = n_cosmos * n_examples_per_cosmo
    LOGGER.info(f"There's a total of {n_examples} data vectors to be evaluated ({n_examples_per_cosmo} per cosmology)")

    # network constants
    global_batch_size = distribute.get_global_batch_size(
        strategy, net_conf["dset"]["eval"]["grid"]["local_batch_size"]
    )
    n_steps = math.ceil(n_examples / global_batch_size)

    grid_pipeline = GridPipeline(
        conf=msfm_conf, **{**dlss_conf["dset"]["general"], **dlss_conf["dset"]["eval"]["grid"]}
    )

    # like https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
    def dataset_fn(input_context):
        dset = grid_pipeline.get_dset(
            tfr_pattern=tfr_pattern,
            **dset_kwargs,
            # distribution
            input_context=input_context,
        )

        return dset

    dist_dset = strategy.distribute_datasets_from_function(dataset_fn)

    LOGGER.timer.start("eval")

    preds = []
    cosmos = []
    i_sobols = []
    i_noises = []
    i_examples = []
    for dv_batch, cosmo_batch, index_batch in LOGGER.progressbar(
        dist_dset, at_level="info", total=n_steps, desc="evaluating the grid"
    ):
        # DistributedValues of shape (local_batch_size, n_output)
        pred_batch = strategy.run(model.tf_call, args=(dv_batch,))

        # shape (global_batch_size, n_output)
        pred_batch = strategy.gather(pred_batch, axis=0)
        # shape (global_batch_size, n_params)
        cosmo_batch = strategy.gather(cosmo_batch, axis=0)
        # shape (global_batch_size,) NOTE it's important that gather takes place on the tensor (not tuple) level
        i_sobol_batch = strategy.gather(index_batch[0], axis=0)
        i_noise_batch = strategy.gather(index_batch[1], axis=0)
        i_example_batch = strategy.gather(index_batch[2], axis=0)

        preds.append(pred_batch)
        cosmos.append(cosmo_batch)
        i_sobols.append(i_sobol_batch)
        i_noises.append(i_noise_batch)
        i_examples.append(i_example_batch)

    # sort according to the sobol index
    sorted_indices = tf.argsort(tf.concat(i_sobols, axis=0), axis=0)

    # shape (n_cosmos, n_examples_per_cosmo, None)
    preds = _stack_grid_cosmos(preds, sorted_indices, n_examples_per_cosmo).numpy()
    cosmos = _stack_grid_cosmos(cosmos, sorted_indices, n_examples_per_cosmo).numpy()
    i_sobols = _stack_grid_cosmos(i_sobols, sorted_indices, n_examples_per_cosmo).numpy()
    i_noises = _stack_grid_cosmos(i_noises, sorted_indices, n_examples_per_cosmo).numpy()
    i_examples = _stack_grid_cosmos(i_examples, sorted_indices, n_examples_per_cosmo).numpy()
    LOGGER.info(f"Reshaped the results")

    # double check that the cosmologies are sorted correctly and remove the redundant axis
    cosmos = _remove_example_axis(cosmos)
    i_sobols = _remove_example_axis(i_sobols)

    out_file = _get_out_file(dir_out, file_label)
    with h5py.File(out_file, "a") as f:
        f.create_dataset(name="grid/pred", data=preds)
        f.create_dataset(name="grid/cosmo", data=cosmos)
        f.create_dataset(name="grid/i_sobol", data=i_sobols)
        f.create_dataset(name="grid/i_noise", data=i_noises)
        f.create_dataset(name="grid/i_example", data=i_examples)

    LOGGER.info(f"Evaluation of the grid has finished, saved the predictions in {out_file}")


def evaluate_fiducial(
    model,
    strategy,
    tfr_pattern,
    msfm_conf,
    dlss_conf,
    net_conf,
    dir_out,
    file_label=None,
    training_set=True,
):
    """Evaluate the model on the fiducial part of the CosmoGrid.

    Args:
        model (DeltaLossModel): Model to be evaluated.
        strategy (tf.distribute.Strategy): Distribution strategy instance within which the model has been created.
        This is used to distribute the dataset.
        tfr_pattern (str): Glob pattern of the .tfrecord files containing the data.
        msfm_conf (dict): Configuration file of the msfm pipeline.
        net_conf (dict): Configuration file of the specific model.
        dir_out (str): Output directory, this is where the evaluations will be saved.
        i_noise (int, str): Noise index. The string "all" is also allowed, then the multi noise dataset is used.
        file_label (str, optional): Optional suffix to append to the output file names. Defaults to None.
        training_set (bool, optional): Whether it's a training or validation set. This changes how the result is
            stored.
    """
    print("\n")
    LOGGER.info(f"Starting evaluation of the fiducial")

    dset_kwargs = net_conf["dset"]["eval"]["fiducial"]

    # pipeline constants
    n_cosmos = 1  # only the true fiducial
    n_patches = msfm_conf["analysis"]["n_patches"]
    n_perms_per_cosmo = msfm_conf["analysis"]["fiducial"]["n_perms_per_cosmo"]
    n_examples_per_cosmo = n_patches * n_perms_per_cosmo

    # multiple shape and poisson noise realizations
    n_examples_per_cosmo *= dset_kwargs["n_noise"]

    n_examples = n_cosmos * n_examples_per_cosmo
    LOGGER.info(f"There's a total of {n_examples} data vectors to be evaluated")

    # network constants
    global_batch_size = distribute.get_global_batch_size(
        strategy, net_conf["dset"]["eval"]["fiducial"]["local_batch_size"]
    )
    n_steps = math.ceil(n_examples / global_batch_size)

    fiducial_pipeline = FiducialPipeline(
        conf=msfm_conf, **{**dlss_conf["dset"]["general"], **dlss_conf["dset"]["eval"]["fiducial"]}
    )

    # like https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
    def dataset_fn(input_context):
        dset = fiducial_pipeline.get_dset(
            tfr_pattern=tfr_pattern,
            **dset_kwargs,
            # distribution
            input_context=input_context,
        )

        return dset

    dist_dset = strategy.distribute_datasets_from_function(dataset_fn)

    preds = []
    i_examples = []
    i_noises = []
    for dv_batch, index_batch in LOGGER.progressbar(
        dist_dset, at_level="info", total=n_steps, desc="evaluating at the fiducial"
    ):
        # DistributedValues of shape (local_batch_size, n_output)
        pred_batch = strategy.run(model.tf_call, args=(dv_batch,))

        # shape (global_batch_size, n_output)
        pred_batch = strategy.gather(pred_batch, axis=0)
        # shape (global_batch_size)
        i_example_batch = strategy.gather(index_batch[0], axis=0)
        i_noise_batch = strategy.gather(index_batch[1], axis=0)

        preds.append(pred_batch)
        i_examples.append(i_example_batch)
        i_noises.append(i_noise_batch)

    preds = tf.concat(preds, axis=0)
    i_examples = tf.concat(i_examples, axis=0)
    i_noises = tf.concat(i_noises, axis=0)
    LOGGER.info(f"Reshaped the results")

    # sort according to the example index
    sorted_indices = tf.argsort(i_examples)
    preds = tf.gather(preds, sorted_indices)
    i_examples = tf.gather(i_examples, sorted_indices)
    i_noises = tf.gather(i_noises, sorted_indices)
    LOGGER.info(f"Sorted the results")

    out_file = _get_out_file(dir_out, file_label)
    with h5py.File(out_file, "a") as f:
        if training_set:
            f.create_dataset(name="fiducial/train/pred", data=preds)
            f.create_dataset(name="fiducial/train/i_example", data=i_examples)
            f.create_dataset(name="fiducial/train/i_noise", data=i_noises)
        else:
            f.create_dataset(name="fiducial/vali/pred", data=preds)
            f.create_dataset(name="fiducial/vali/i_example", data=i_examples)
            f.create_dataset(name="fiducial/vali/i_noise", data=i_noises)

    LOGGER.info(f"Evaluation of the fiducial has finished, saved the predictions in {out_file}")
