# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2023
Author: Arne Thomsen

Evaluate the DeepSphere graph neural networks on the CosmoGrid
"""

import numpy as np
import tensorflow as tf
import os, warnings, h5py, math, logging, wandb
from trianglechain import TriangleChain

from msfm.fiducial_pipeline import FiducialPipeline
from msfm.grid_pipeline import GridPipeline
from msfm.utils import logger, files

from deep_lss.utils import distribute, configuration
from deep_lss.utils.distribute import HorovodStrategy

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)

# Keys in dset.common that are only used by Cls-based pipelines and must be stripped
# before constructing GridPipeline / FiducialPipeline (map-based pipelines don't accept them).
_CLS_ONLY_KEYS = frozenset({"with_cross_z", "with_cross_probe", "ggl_only"})

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

    return tensors.numpy()


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
    model, tfr_pattern, msfm_conf, dlss_conf, net_conf, dir_out, file_label=None, wandb_run=None, debug=False
):
    """Evaluate the model on the grid part of the CosmoGrid.

    Args:
        model (DeltaLossModel): Model to be evaluated.
        tfr_pattern (str): Glob pattern of the .tfrecord files containing the data.
        msfm_conf (dict): Configuration file of the msfm pipeline.
        net_conf (dict): Configuration file of the specific model.
        dir_out (str): Output directory, this is where the evaluations will be saved.
        file_label (str, optional): Optional suffix to append to the output file names. Defaults to None.
    """
    print("\n")
    LOGGER.info(f"Starting evaluation of the grid")

    dset_kwargs = {**net_conf["dset"]["eval"]["common"], **net_conf["dset"]["eval"]["grid"]}
    dset_kwargs["drop_remainder"] = True

    # network constants
    save_second_to_last_layer = net_conf["network"]["save_second_to_last_layer"]

    strategy = model.strategy

    n_side = msfm_conf["analysis"]["n_side"]
    smooth_nside = net_conf["network"].get("smooth_nside", None)
    if smooth_nside is not None and smooth_nside < n_side:
        data_vec_pix, _, _, _ = files.load_pixel_file(msfm_conf)
        _, parent_output_idx = configuration.get_smooth_nside_indices(data_vec_pix, n_side, smooth_nside)
    else:
        smooth_nside = None
        parent_output_idx = None

    grid_pipeline = GridPipeline(
        conf=msfm_conf,
        **{k: v for k, v in {**dlss_conf["dset"]["common"], **dlss_conf["dset"]["eval"]["grid"]}.items()
           if k not in _CLS_ONLY_KEYS},
    )

    local_batch_size = dset_kwargs["local_batch_size"]
    global_batch_size = distribute.get_global_batch_size(strategy, local_batch_size)

    # like https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
    def dataset_fn(input_context):
        dset = grid_pipeline.get_dset(
            tfr_pattern=tfr_pattern,
            **dset_kwargs,
            # distribution
            input_context=input_context,
            # nside downsampling
            downsample_nside=smooth_nside,
            parent_output_idx=parent_output_idx,
        )

        if debug:
            dset = dset.take(global_batch_size * 2)

        return dset

    dist_dset = strategy.distribute_datasets_from_function(dataset_fn)

    n_cosmos = msfm_conf["analysis"]["grid"]["n_cosmos"]
    n_noise = grid_pipeline.n_noise
    n_signal = grid_pipeline.n_signal
    n_examples_per_cosmo = n_noise * n_signal
    n_examples = n_cosmos * n_examples_per_cosmo
    LOGGER.info(f"There's a total of {n_examples} data vectors to be evaluated ({n_examples_per_cosmo} per cosmology)")

    n_batches = math.ceil(n_examples / global_batch_size)

    if n_examples % (strategy.num_replicas_in_sync * local_batch_size) != 0:
        LOGGER.warning(
            f"Number of examples {n_examples} is not divisible by the number of replicas "
            f"{strategy.num_replicas_in_sync} times the local batch size {local_batch_size}"
        )

    # set up a network that outputs the second to last layer too
    if save_second_to_last_layer:
        last_layer = model.network.layers[-1]
        second_to_last_layer = model.network.layers[-2]
        two_output_model = tf.keras.Model(
            inputs=model.network.input, outputs=[last_layer.output, second_to_last_layer.output]
        )

    preds = []
    second_to_last_layer = []
    cosmos = []
    i_sobols = []
    i_signals = []
    i_noises = []
    for dv_batch, _, cosmo_batch, index_batch in LOGGER.progressbar(
        dist_dset, at_level="info", total=n_batches, desc="evaluating the grid"
    ):
        # DistributedValues of shape (local_batch_size, n_output)
        if save_second_to_last_layer:
            pred_batch, second_to_last_layer_batch = strategy.run(two_output_model, args=(dv_batch,))
            second_to_last_layer_batch = strategy.gather(second_to_last_layer_batch, axis=0)
        else:
            pred_batch = strategy.run(model.tf_call, args=(dv_batch,))

        # shape (global_batch_size, n_output)
        pred_batch = strategy.gather(pred_batch, axis=0)
        # shape (global_batch_size, n_params)
        cosmo_batch = strategy.gather(cosmo_batch, axis=0)
        # shape (global_batch_size,) NOTE it's important that gather takes place on the tensor (not tuple) level
        i_sobol_batch = strategy.gather(index_batch[0], axis=0)
        i_signal_batch = strategy.gather(index_batch[1], axis=0)
        i_noise_batch = strategy.gather(index_batch[2], axis=0)

        preds.append(pred_batch)
        cosmos.append(cosmo_batch)
        i_sobols.append(i_sobol_batch)
        i_signals.append(i_signal_batch)
        i_noises.append(i_noise_batch)
        if save_second_to_last_layer:
            second_to_last_layer.append(second_to_last_layer_batch)

    # sort according to the sobol index
    sorted_indices = tf.argsort(tf.concat(i_sobols, axis=0), axis=0, stable=True)

    # shape (n_cosmos, n_examples_per_cosmo, None)
    preds = _stack_grid_cosmos(preds, sorted_indices, n_examples_per_cosmo)
    cosmos = _stack_grid_cosmos(cosmos, sorted_indices, n_examples_per_cosmo)
    i_sobols = _stack_grid_cosmos(i_sobols, sorted_indices, n_examples_per_cosmo)
    i_noises = _stack_grid_cosmos(i_noises, sorted_indices, n_examples_per_cosmo)
    i_signals = _stack_grid_cosmos(i_signals, sorted_indices, n_examples_per_cosmo)
    if save_second_to_last_layer:
        second_to_last_layer = _stack_grid_cosmos(second_to_last_layer_batch, sorted_indices, n_examples_per_cosmo)
    LOGGER.info(f"Reshaped the results")

    out_file = _get_out_file(dir_out, file_label)

    def write_out_file():
        with h5py.File(out_file, "a") as f:
            keys = [
                "grid/preds/test",
                "grid/cosmos/test",
                "grid/i_sobol/test",
                "grid/i_signal/test",
                "grid/i_noise/test",
            ]
            if save_second_to_last_layer:
                keys.append("grid/second_to_last_layer/test")
            for key in keys:
                if key in f:
                    del f[key]
            f.create_dataset(name="grid/preds/test", data=preds)
            f.create_dataset(name="grid/cosmos/test", data=cosmos)
            f.create_dataset(name="grid/i_sobol/test", data=i_sobols)
            f.create_dataset(name="grid/i_signal/test", data=i_signals)
            f.create_dataset(name="grid/i_noise/test", data=i_noises)
            if save_second_to_last_layer:
                f.create_dataset(name="grid/second_to_last_layer/test", data=second_to_last_layer)

        LOGGER.info(f"Evaluation of the grid has finished, saved the predictions in {out_file}")

    if isinstance(model.strategy, (tf.distribute.MultiWorkerMirroredStrategy, HorovodStrategy)):
        if model.is_chief():
            LOGGER.info(f"Chief here")
            write_out_file()
    else:
        write_out_file()

    if wandb_run is not None:
        artifact = wandb.Artifact(name="grid-predictions", type="predictions")
        artifact.add_file(local_path=out_file)
        wandb_run.log_artifact(artifact)

    return out_file


def evaluate_fiducial(
    model, tfr_pattern, msfm_conf, dlss_conf, net_conf, dir_out, training_set=True, file_label=None, wandb_run=None
):
    """Evaluate the model on the fiducial part of the CosmoGrid.

    Args:
        model (DeltaLossModel): Model to be evaluated.
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

    dset_kwargs = {**net_conf["dset"]["eval"]["common"], **net_conf["dset"]["eval"]["fiducial"]}

    # pipeline constants
    n_cosmos = 1  # only the true fiducial
    n_patches = msfm_conf["analysis"]["n_patches"]
    n_perms_per_cosmo = msfm_conf["analysis"]["fiducial"]["n_perms_per_cosmo"]
    n_examples_per_cosmo = n_patches * n_perms_per_cosmo

    # multiple shape and poisson noise realizations
    n_examples_per_cosmo *= dset_kwargs["noise_indices"]

    n_examples = n_cosmos * n_examples_per_cosmo
    LOGGER.info(f"There's a total of {n_examples} data vectors to be evaluated")

    # network constants
    save_second_to_last_layer = net_conf["network"]["save_second_to_last_layer"]

    strategy = model.strategy
    local_batch_size = dset_kwargs["local_batch_size"]
    global_batch_size = distribute.get_global_batch_size(strategy, local_batch_size)
    n_batches = math.ceil(n_examples / global_batch_size)

    if n_examples % (strategy.num_replicas_in_sync * local_batch_size) != 0:
        LOGGER.warning(
            f"Number of examples {n_examples} is not divisible by the number of replicas "
            f"{strategy.num_replicas_in_sync} times the local batch size {local_batch_size}"
        )

    n_side = msfm_conf["analysis"]["n_side"]
    smooth_nside = net_conf["network"].get("smooth_nside", None)
    if smooth_nside is not None and smooth_nside < n_side:
        data_vec_pix, _, _, _ = files.load_pixel_file(msfm_conf)
        _, parent_output_idx = configuration.get_smooth_nside_indices(data_vec_pix, n_side, smooth_nside)
    else:
        smooth_nside = None
        parent_output_idx = None

    fiducial_pipeline = FiducialPipeline(
        conf=msfm_conf,
        **{k: v for k, v in {**dlss_conf["dset"]["common"], **dlss_conf["dset"]["eval"]["fiducial"]}.items()
           if k not in _CLS_ONLY_KEYS},
    )

    # like https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
    def dataset_fn(input_context):
        dset = fiducial_pipeline.get_dset(
            tfr_pattern=tfr_pattern,
            **dset_kwargs,
            # distribution
            input_context=input_context,
            # nside downsampling
            downsample_nside=smooth_nside,
            parent_output_idx=parent_output_idx,
        )

        return dset

    dist_dset = strategy.distribute_datasets_from_function(dataset_fn)

    # set up a network that outputs the second to last layer too
    if save_second_to_last_layer:
        last_layer = model.network.layers[-1]
        second_to_last_layer = model.network.layers[-2]
        two_output_model = tf.keras.Model(
            inputs=model.network.input, outputs=[last_layer.output, second_to_last_layer.output]
        )

    preds = []
    second_to_last_layer = []
    i_examples = []
    i_noises = []
    for dv_batch, _, index_batch in LOGGER.progressbar(
        dist_dset, at_level="info", total=n_batches, desc="evaluating at the fiducial"
    ):
        # DistributedValues of shape (local_batch_size, n_output)
        if save_second_to_last_layer:
            pred_batch, second_to_last_layer_batch = strategy.run(two_output_model, args=(dv_batch,))
            second_to_last_layer_batch = strategy.gather(second_to_last_layer_batch, axis=0)
        else:
            pred_batch = strategy.run(model.tf_call, args=(dv_batch,))

        # shape (global_batch_size, n_output)
        pred_batch = strategy.gather(pred_batch, axis=0)
        # shape (global_batch_size)
        i_example_batch = strategy.gather(index_batch[0], axis=0)
        i_noise_batch = strategy.gather(index_batch[1], axis=0)

        preds.append(pred_batch)
        i_examples.append(i_example_batch)
        i_noises.append(i_noise_batch)
        if save_second_to_last_layer:
            second_to_last_layer.append(second_to_last_layer_batch)

    preds = tf.concat(preds, axis=0)
    i_examples = tf.concat(i_examples, axis=0)
    i_noises = tf.concat(i_noises, axis=0)
    if save_second_to_last_layer:
        second_to_last_layer = tf.concat(second_to_last_layer_batch, axis=0)
    LOGGER.info(f"Reshaped the results")

    # sort according to the example index
    sorted_indices = tf.argsort(i_examples)
    preds = tf.gather(preds, sorted_indices)
    i_examples = tf.gather(i_examples, sorted_indices)
    i_noises = tf.gather(i_noises, sorted_indices)
    if save_second_to_last_layer:
        second_to_last_layer = tf.gather(second_to_last_layer, sorted_indices)
    LOGGER.info(f"Sorted the results")

    out_file = _get_out_file(dir_out, file_label)

    def write_out_file():
        with h5py.File(out_file, "a") as f:
            if training_set:
                keys = ["fiducial/train/pred", "fiducial/train/i_example", "fiducial/train/i_noise"]
                if save_second_to_last_layer:
                    keys.append("fiducial/train/second_to_last_layer")
                for key in keys:
                    if key in f:
                        del f[key]
                f.create_dataset(name="fiducial/train/pred", data=preds)
                f.create_dataset(name="fiducial/train/i_example", data=i_examples)
                f.create_dataset(name="fiducial/train/i_noise", data=i_noises)
                if save_second_to_last_layer:
                    f.create_dataset(name="fiducial/train/second_to_last_layer", data=second_to_last_layer)
            else:
                keys = ["fiducial/vali/pred", "fiducial/vali/i_example", "fiducial/vali/i_noise"]
                if save_second_to_last_layer:
                    keys.append("fiducial/vali/second_to_last_layer")
                for key in keys:
                    if key in f:
                        del f[key]
                f.create_dataset(name="fiducial/vali/pred", data=preds)
                f.create_dataset(name="fiducial/vali/i_example", data=i_examples)
                f.create_dataset(name="fiducial/vali/i_noise", data=i_noises)
                if save_second_to_last_layer:
                    f.create_dataset(name="fiducial/vali/second_to_last_layer", data=second_to_last_layer)

        LOGGER.info(f"Evaluation of the fiducial has finished, saved the predictions in {out_file}")

    if isinstance(model.strategy, (tf.distribute.MultiWorkerMirroredStrategy, HorovodStrategy)):
        if model.is_chief():
            LOGGER.info(f"Chief here")
            write_out_file()
    else:
        write_out_file()

    if wandb_run is not None:
        artifact = wandb.Artifact(name="fiducial-predictions", type="predictions")
        artifact.add_file(local_path=out_file)
        wandb_run.log_artifact(artifact)

    return out_file


def append_obs_to_file(pred_file, label, pred):
    pred = np.squeeze(pred)

    with h5py.File(pred_file, "a") as f:
        if label in f:
            del f[label]
        f.create_dataset(name=label, data=pred)
        print(f"wrote {label} of shape {pred.shape}")


def evaluate_obs_grid(pred_file, grid_preds, grid_cosmos, i_sobols, i_signals, i_noises, n_examples=4):
    """Write one example per unique cosmology into the obs/ section, labeled by (i_sobol,i_signal,i_noise)."""
    _, first_indices = np.unique(i_sobols, return_index=True)
    for idx in first_indices[:n_examples]:
        label = f"grid_({int(i_sobols[idx])},{int(i_signals[idx])},{int(i_noises[idx])})"
        append_obs_to_file(pred_file, f"obs/preds/{label}", grid_preds[idx])
        append_obs_to_file(pred_file, f"obs/cosmos/{label}", grid_cosmos[idx])


def evaluate_obs_des(model_fn, pred_file, msfm_conf, dlss_conf):
    """Evaluate real DES Y3 catalogs through the network, with and without systematics."""
    from msfm.utils import catalog, observation

    with_lensing = dlss_conf["dset"]["common"]["with_lensing"]
    with_clustering = dlss_conf["dset"]["common"]["with_clustering"]

    wl_map = catalog.build_metacal_map_from_cat(msfm_conf)[0] if with_lensing else None
    gc_map = catalog.build_maglim_map_from_cat(msfm_conf) if with_clustering else None

    for apply_sys, label in [(True, "DESy3"), (False, "DESy3_no_sys")]:
        des_dv, _, _ = observation.forward_model_observation_map(
            wl_gamma_map=wl_map,
            gc_count_map=gc_map,
            conf=msfm_conf,
            apply_norm=True,
            with_padding=True,
            nest_in=False,
            apply_maglim_sys_map=apply_sys,
        )
        pred = model_fn(des_dv[np.newaxis])
        append_obs_to_file(pred_file, f"obs/preds/{label}", pred)


def evaluate_obs_buzzard(model_fn, pred_file, msfm_conf, dlss_conf, labels):
    """Evaluate Buzzard N-body simulation realizations through the network."""
    from msfm.utils import buzzard, observation

    with_lensing = dlss_conf["dset"]["common"]["with_lensing"]
    with_clustering = dlss_conf["dset"]["common"]["with_clustering"]

    buzzard_indices, lensing_files, clustering_files = buzzard.get_filenames(msfm_conf)

    for i, lensing_file, clustering_file in zip(buzzard_indices, lensing_files, clustering_files):
        label = f"Buzzard_{i}"
        if label not in labels:
            continue
        wl_map = buzzard.get_lensing_map(lensing_file) if with_lensing else None
        gc_map = buzzard.get_clustering_map(clustering_file) if with_clustering else None
        obs_map, _, _ = observation.forward_model_observation_map(
            wl_gamma_map=wl_map,
            gc_count_map=gc_map,
            conf=msfm_conf,
            apply_norm=True,
            with_padding=True,
            nest_in=False,
        )
        pred = model_fn(obs_map[np.newaxis])
        append_obs_to_file(pred_file, f"obs/preds/{label}", pred)


def evaluate_obs_benchmark(model_fn, pred_file, msfm_conf, dlss_conf, data_dir, obs_labels):
    """Evaluate benchmark simulations from data_dir/obs/.

    Writes obs/preds/{label}_stack (all realizations) and obs/preds/{label}_mean to pred_file,
    and obs/cosmos/{label} containing the fiducial cosmology.
    """
    from msfm.utils import parameters

    with_lensing = dlss_conf["dset"]["common"]["with_lensing"]
    with_clustering = dlss_conf["dset"]["common"]["with_clustering"]
    n_z_lensing = len(msfm_conf["survey"]["metacal"]["z_bins"]) if with_lensing else 0

    norm_lensing = msfm_conf["analysis"]["normalization"]["lensing"]
    norm_clustering = msfm_conf["analysis"]["normalization"]["clustering"]

    target_params = dlss_conf["dset"]["training"]["params"]
    fiducial_cosmo = parameters.get_fiducials(target_params, msfm_conf)

    obs_dir = os.path.join(data_dir, "obs")

    for label in obs_labels:
        obs_file = os.path.join(obs_dir, f"{label}_obs_maps.h5")
        if not os.path.exists(obs_file):
            LOGGER.warning(f"Benchmark file not found: {obs_file}, skipping.")
            continue

        out_label = label

        with h5py.File(obs_file, "r") as f:
            obs_maps = f["obs/maps"][:].astype(np.float32)

        # Select and normalise channels based on probe
        n_channels_total = obs_maps.shape[-1]
        if n_channels_total > n_z_lensing and with_lensing and not with_clustering:
            obs_maps = obs_maps[:, :, :n_z_lensing]
            obs_maps /= norm_lensing
        elif n_channels_total > n_z_lensing and with_clustering and not with_lensing:
            obs_maps = obs_maps[:, :, n_z_lensing:]
            obs_maps /= norm_clustering
        else:
            if with_lensing:
                obs_maps[:, :, :n_z_lensing] /= norm_lensing
            if with_clustering:
                obs_maps[:, :, n_z_lensing:] /= norm_clustering

        preds = np.concatenate([model_fn(obs_maps[i : i + 1]) for i in range(len(obs_maps))], axis=0)

        append_obs_to_file(pred_file, f"obs/preds/{out_label}_stack", preds)
        append_obs_to_file(pred_file, f"obs/preds/{out_label}_mean", np.mean(preds, axis=0))
        append_obs_to_file(pred_file, f"obs/cosmos/{out_label}", fiducial_cosmo)


def plot_summary_space_prior_predictive(grid_preds, obs_pred, n_rand=1_000, np_seed=12):
    rng = np.random.default_rng(np_seed)

    i_rand = rng.integers(0, grid_preds.shape[0], n_rand)

    tri = TriangleChain(size=2)
    tri.scatter(np.array(grid_preds)[i_rand], scatter_kwargs={"s": 10, "marker": "o"})
    tri.scatter(
        np.atleast_2d(obs_pred),
        scatter_kwargs={"s": 200, "marker": "*"},
        color="k",
        scatter_vline_1D=True,
        plot_histograms_1D=False,
    )
