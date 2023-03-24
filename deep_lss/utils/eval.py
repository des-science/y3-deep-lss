# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2023
Author: Arne Thomsen

Evaluate the DeepSphere graph neural networks on the CosmoGrid
"""

import numpy as np
import tensorflow as tf
import os, warnings, h5py, math, logging

from msfm import fiducial_pipeline, grid_pipeline
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


def stack_grid_cosmos(tensors, sorted_indices, n_examples_per_cosmo):
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


def remove_example_axis(array):
    """Takes in a tensor of shape (n_cosmos, n_examples_per_cosmo, None) and checks whether the value along axis 1 is
    constant to remove that redundant axis.

    Args:
        tensor (np.ndarray): Shape (n_cosmos, n_examples_per_cosmo, None)

    Raises:
        RuntimeError: If the values along the axis of length n_examples_per_cosmo are not all equal

    returns:
        array: Shape (n_cosmos, None), where the redundancy has been removed
    """
    # double check that the cosmologies are sorted correctly and remove the redundant axis
    if np.all([np.equal(array[:, i, :], array[:, i + 1, :]) for i in range(array.shape[1] - 1)]):
        array = array[:, 0, :]
    else:
        raise RuntimeError(f"The cosmologies are not sorted correctly")

    return array


def evaluate_grid(model, strategy, tfr_pattern, msfm_conf, net_conf, dir_out, file_label=None):
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

    # pipeline constants
    n_params = len(msfm_conf["analysis"]["params"])
    n_cosmos = msfm_conf["analysis"]["grid"]["n_cosmos"]
    n_patches = msfm_conf["analysis"]["n_patches"]
    n_perms_per_cosmo = msfm_conf["analysis"]["grid"]["n_perms_per_cosmo"]
    n_noise_per_example = msfm_conf["analysis"]["grid"]["n_noise_per_example"]
    n_examples_per_cosmo = n_patches * n_perms_per_cosmo * n_noise_per_example
    n_examples = n_cosmos * n_examples_per_cosmo
    LOGGER.info(f"There's a total of {n_examples} data vectors to be evaluated")

    # network constants
    global_batch_size = distribute.get_global_batch_size(
        strategy, net_conf["dset"]["eval"]["grid"]["local_batch_size"]
    )
    n_steps = math.ceil(n_examples / global_batch_size)

    # like https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
    def dataset_fn(input_context):
        dset = grid_pipeline.get_grid_dset(
            tfr_pattern=tfr_pattern,
            n_params=n_params,
            conf=msfm_conf,
            **net_conf["dset"]["eval"]["grid"],
            # distribution
            input_context=input_context,
        )
        return dset

    dist_dset = strategy.distribute_datasets_from_function(dataset_fn)

    LOGGER.timer.start("eval")

    preds = []
    cosmos = []
    sobols = []
    noises = []
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
        sobol_batch = strategy.gather(index_batch[0], axis=0)
        noise_batch = strategy.gather(index_batch[1], axis=0)

        preds.append(pred_batch)
        cosmos.append(cosmo_batch)
        sobols.append(sobol_batch)
        noises.append(noise_batch)

    # sort according to the sobol index
    sorted_indices = tf.argsort(tf.concat(sobols, axis=0), axis=0)

    # shape (n_cosmos, n_examples_per_cosmo, None)
    preds = stack_grid_cosmos(preds, sorted_indices, n_examples_per_cosmo).numpy()
    cosmos = stack_grid_cosmos(cosmos, sorted_indices, n_examples_per_cosmo).numpy()
    sobols = stack_grid_cosmos(sobols, sorted_indices, n_examples_per_cosmo).numpy()
    noises = stack_grid_cosmos(noises, sorted_indices, n_examples_per_cosmo).numpy()
    LOGGER.info(f"Reshaped the results")

    # double check that the cosmologies are sorted correctly and remove the redundant axis
    cosmos = remove_example_axis(cosmos)
    sobols = remove_example_axis(sobols)
    noises = remove_example_axis(noises)

    out_file = _get_out_file(dir_out, file_label)
    with h5py.File(out_file, "a") as f:
        f.create_dataset(name="grid/preds", data=preds)
        f.create_dataset(name="grid/cosmos", data=cosmos)
        f.create_dataset(name="grid/sobols", data=sobols)
        f.create_dataset(name="grid/noises", data=noises)

    LOGGER.info(f"Evaluation of the grid has finished, saved the predictions in {out_file}")


def evaluate_fiducial(model, strategy, tfr_pattern, msfm_conf, net_conf, dir_out, file_label=None):
    """Evaluate the model on the fiducial part of the CosmoGrid.

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
    LOGGER.info(f"Starting evaluation of the fiducial")

    # pipeline constants
    n_cosmos = 1  # only the true fiducial
    n_patches = msfm_conf["analysis"]["n_patches"]
    n_perms_per_cosmo = msfm_conf["analysis"]["fiducial"]["n_perms_per_cosmo"]
    n_examples_per_cosmo = n_patches * n_perms_per_cosmo
    n_examples = n_cosmos * n_examples_per_cosmo
    LOGGER.info(f"There's a total of {n_examples} data vectors to be evaluated")

    # network constants
    global_batch_size = distribute.get_global_batch_size(
        strategy, net_conf["dset"]["eval"]["fiducial"]["local_batch_size"]
    )
    n_steps = math.ceil(n_examples / global_batch_size)

    # like https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
    def dataset_fn(input_context):
        dset = fiducial_pipeline.get_fiducial_dset(
            tfr_pattern=tfr_pattern,
            **net_conf["dset"]["eval"]["fiducial"],
            # distribution
            input_context=input_context,
        )
        return dset

    dist_dset = strategy.distribute_datasets_from_function(dataset_fn)

    preds = []
    indices = []
    for dv_batch, index_batch in LOGGER.progressbar(
        dist_dset, at_level="info", total=n_steps, desc="evaluating at the fiducial"
    ):
        # DistributedValues of shape (local_batch_size, n_output)
        pred_batch = strategy.run(model.tf_call, args=(dv_batch,))

        # shape (global_batch_size, n_output)
        pred_batch = strategy.gather(pred_batch, axis=0)
        # shape (global_batch_size)
        index_batch = strategy.gather(index_batch[0], axis=0)

        preds.append(pred_batch)
        indices.append(index_batch)

    preds = tf.concat(preds, axis=0)
    indices = tf.concat(indices, axis=0)
    LOGGER.info(f"Reshaped the results")

    out_file = _get_out_file(dir_out, file_label)
    with h5py.File(out_file, "a") as f:
        f.create_dataset(name="fiducial/preds", data=preds)
        f.create_dataset(name="fiducial/indices", data=indices)

    LOGGER.info(f"Evaluation of the fiducial has finished, saved the predictions in {out_file}")
