# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2023
Author: Arne Thomsen

Evaluate the DeepSphere graph neural networks on the grid of cosmologies sampled in the CosmoGrid

Meant for the GPU nodes of the Perlmutter cluster at NERSC.
"""

import tensorflow as tf

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

import os, argparse, warnings, yaml, wandb, numpy as np, h5py

from msfm.utils import logger, files

from deep_lss.utils import configuration, distribute, evaluation
from deep_lss.models.base_model import BaseModel
from deep_lss.nets import NETWORKS

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def setup():
    description = "Train the specified network at the fiducial cosmology."
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="logging level",
    )
    parser.add_argument(
        "--dist_strategy",
        choices=[None, "mirrored", "multi_worker_mirrored", "horovod"],
        default=None,
        help="distribution strategy, use None to run locally",
    )
    parser.add_argument(
        "--fidu_train_tfr_pattern",
        type=str,
        default=None,
        help="input root dir of the fiducial data vectors (training)",
    )
    parser.add_argument(
        "--fidu_vali_tfr_pattern",
        type=str,
        default=None,
        help="input root dir of the fiducial data vectors (validation)",
    )
    parser.add_argument(
        "--grid_vali_tfr_pattern",
        type=str,
        default=None,
        help="input root dir of the grid data vectors (validation)",
    )
    parser.add_argument(
        "--dir_model",
        type=str,
        default=None,
        help="dir where the model checkpoints to be loaded are saved. If None, read from temp file",
    )
    parser.add_argument(
        "--evaluate_all_checkpoints",
        action="store_true",
        help="evaluate all checkpoints (instead of only the latest one)",
    )
    parser.add_argument("--debug", action="store_true", help="activate debug mode")
    parser.add_argument("--file_label", type=str, default=None, help="A suffix that is appended to the files")
    parser.add_argument("--wandb", action="store_true", help="log to weights & biases, otherwise log to tensorboard")
    parser.add_argument("--wandb_tags", nargs="+", type=str, default=None, help="tags for weights & biases")
    parser.add_argument("--wandb_notes", type=str, default=None, help="notes for weights & biases (longer than tags)")

    # Individual observation evaluation flags (all default off)
    parser.add_argument("--include_grid", action="store_true", help="write stride-spaced grid examples into obs/")
    parser.add_argument("--n_grid_examples", type=int, default=4)
    parser.add_argument("--include_des", action="store_true", help="evaluate DES Y3 catalogs")
    parser.add_argument("--include_buzzard", action="store_true", help="evaluate Buzzard N-body realizations")
    parser.add_argument("--buzzard_labels", nargs="+", default=["Buzzard_mean"])
    parser.add_argument("--include_mocks", action="store_true", help="evaluate mock observations from data_dir/obs/")
    parser.add_argument("--mock_labels", nargs="+", default=["fiducial_bench"])
    parser.add_argument("--data_dir", type=str, default=None, help="base data directory (needed for --include_mocks)")

    args, _ = parser.parse_known_args()

    logger.set_all_loggers_level(args.verbosity)

    # print arguments
    logger.set_all_loggers_level(args.verbosity)
    for key, value in vars(args).items():
        LOGGER.info(f"{key} = {value}")

    if args.dir_model is None:
        job_id = os.environ["SLURM_JOB_ID"]
        temp_file = f"./.env_var/id_{job_id}.txt"
        with open(temp_file, "r") as f:
            args.dir_model = f.read().strip()
        LOGGER.warning(f"Loaded the model directory {args.dir_model} from {temp_file}")

    if args.debug:
        pass
        # tf.config.run_functions_eagerly(True)
        # LOGGER.warning(f"!!!!! Running the training in test mode, TensorFlow is executed eagerly !!!!!")
        # tf.config.set_soft_device_placement(False)
        # tf.debugging.set_log_device_placement(True)
        # tf.data.experimental.enable_debug_mode()

    return args


if __name__ == "__main__":
    args = setup()
    LOGGER.timer.start("main")

    _, _ = distribute.check_devices()
    strategy = distribute.get_strategy(args.dist_strategy)

    # load the configs
    with open(os.path.join(args.dir_model, "configs.yaml"), "r") as f:
        net_conf, dlss_conf, msfm_conf = list(yaml.load_all(f, Loader=yaml.FullLoader))

    LOGGER.info(f"Loaded configs from the model directory")

    # general constants
    all_params = msfm_conf["analysis"]["params"]
    target_params = dlss_conf["dset"]["training"]["params"]
    loss_func = net_conf["run"]["loss_func"]
    n_params = len(target_params)
    LOGGER.info(f"The networks have output shape {n_params} and target {target_params}")

    # pipeline constants
    n_side = msfm_conf["analysis"]["n_side"]
    data_vec_pix, _, _, _ = files.load_pixel_file(msfm_conf)

    smooth_nside = net_conf["network"].get("smooth_nside", None)
    if smooth_nside is not None and smooth_nside < n_side:
        smooth_indices, parent_output_idx = configuration.get_smooth_nside_indices(data_vec_pix, n_side, smooth_nside)
        LOGGER.info(f"Using smooth_nside={smooth_nside}: {len(data_vec_pix)} → {len(smooth_indices)} pixels")
    else:
        smooth_nside = n_side
        smooth_indices = data_vec_pix
        parent_output_idx = None

    n_z_bins = 0
    if dlss_conf["dset"]["common"]["with_lensing"]:
        n_z_bins += len(msfm_conf["survey"]["metacal"]["z_bins"])
    if dlss_conf["dset"]["common"]["with_clustering"]:
        n_z_bins += len(msfm_conf["survey"]["maglim"]["z_bins"])

    # weights and biases
    if args.wandb:
        group_name = distribute.get_wandb_group_name(strategy)

        # TODO track the model as an artifact too so this would be consistent with training in the graph
        wandb_run = wandb.init(
            project="y3-deep-lss",
            config={"msfm": msfm_conf, "dlss": dlss_conf, "net": net_conf},
            dir=args.dir_model,
            group=group_name,
            job_type="evaluation",
            # make sure that wandb logs to the cloud
            mode="online",
            force=True,
            # to be able to log within graph mode
            sync_tensorboard=True,
            # additional metadata
            tags=args.wandb_tags,
            notes=args.wandb_notes,
        )

    smoothing_kwargs = configuration.get_smoothing_kwargs(
        loss_func, msfm_conf, dlss_conf, net_conf, dir_base=args.dir_model, mode="eval"
    )

    if loss_func == "likelihood":
        n_output = n_params + n_params * (n_params + 1) // 2
    elif loss_func == "mutual_info":
        n_output = dlss_conf["mutual_info_loss"]["dim_summary_fac"] * n_params
    elif loss_func == "delta" or loss_func == "mse":
        n_output = n_params

    # set up directories
    checkpoint_dir = os.path.abspath(os.path.join(args.dir_model, "checkpoint"))

    # create all of the variables within the strategy's scope, such that they are mirrored
    with strategy.scope():
        # load the layers
        network = NETWORKS[net_conf["network"]["name"]](
            out_features=n_output, smoothing_kwargs=smoothing_kwargs, **net_conf["network"]["kwargs"]
        ).get_layers()
        LOGGER.info(f"Loaded a network specification of type {NETWORKS[net_conf['network']['name']]}")

        # build the model, same regardless of the loss function (fiducial or grid)
        model = BaseModel(
            network=network,
            n_side=smooth_nside,
            indices=smooth_indices,
            n_neighbors=net_conf["network"]["n_neighbors"],
            input_shape=(None, len(smooth_indices), n_z_bins),
            max_batch_size=net_conf["dset"]["eval"]["grid"]["local_batch_size"],
            checkpoint_dir=checkpoint_dir,
            # always load from a checkpoint
            restore_checkpoint=True,
            strategy=strategy,
        )

    # Build a numpy-level model callable for individual observation evaluation.
    # Includes downsampling when smooth_nside < n_side.
    if parent_output_idx is not None:
        _n_pix_out = len(smooth_indices)
        _counts = np.bincount(parent_output_idx, minlength=_n_pix_out).astype(np.float32)

        def _downsample(maps):
            result = np.zeros((maps.shape[0], _n_pix_out, maps.shape[2]), dtype=maps.dtype)
            np.add.at(result, (slice(None), parent_output_idx, slice(None)), maps)
            return result / _counts[np.newaxis, :, np.newaxis]

    def _call_model(x):
        # HealpySmoothing pre-computes n_matmul_splits=2 for this pixel resolution;
        # tf.split requires the batch dim divisible by 2, so pad batch=1 → 2.
        if x.shape[0] == 1:
            x = np.concatenate([x, x], axis=0)
            return model(x, training=False).numpy()[:1]
        return model(x, training=False).numpy()

    if parent_output_idx is not None:
        model_fn = lambda x: _call_model(_downsample(x))
    else:
        model_fn = _call_model

    def evaluate_current_checkpoint(model):
        train_step = model.get_step()

        if args.file_label is None:
            file_label = train_step
        else:
            file_label = f"{train_step}_{args.file_label}"

        out_file = None

        # fiducial training
        if args.fidu_train_tfr_pattern is not None:
            out_file = evaluation.evaluate_fiducial(
                model=model,
                tfr_pattern=args.fidu_train_tfr_pattern,
                msfm_conf=msfm_conf,
                dlss_conf=dlss_conf,
                net_conf=net_conf,
                dir_out=args.dir_model,
                file_label=file_label,
                training_set=True,
            )
        else:
            LOGGER.warning(f"Skipping evaluation of the fiducial training set")

        # fiducial validation
        if args.fidu_vali_tfr_pattern is not None:
            out_file = evaluation.evaluate_fiducial(
                model=model,
                tfr_pattern=args.fidu_vali_tfr_pattern,
                msfm_conf=msfm_conf,
                dlss_conf=dlss_conf,
                net_conf=net_conf,
                dir_out=args.dir_model,
                file_label=file_label,
                training_set=False,
            )
        else:
            LOGGER.warning(f"Skipping evaluation of the fiducial validation set")

        # grid validation
        if args.grid_vali_tfr_pattern is not None:
            out_file = evaluation.evaluate_grid(
                model=model,
                tfr_pattern=args.grid_vali_tfr_pattern,
                msfm_conf=msfm_conf,
                dlss_conf=dlss_conf,
                net_conf=net_conf,
                dir_out=args.dir_model,
                file_label=file_label,
                debug=args.debug,
            )
        else:
            LOGGER.warning(f"Skipping evaluation of the grid set")

        # Individual observation evaluation (written into obs/ section of the same HDF5)
        if out_file is not None:
            if args.include_grid:
                with h5py.File(out_file, "r") as _f:
                    _gp = _f["grid/preds/test"][:]
                    _gc = _f["grid/cosmos/test"][:]
                    _isob = _f["grid/i_sobol/test"][:]
                    _isig = _f["grid/i_signal/test"][:]
                    _inoi = _f["grid/i_noise/test"][:]
                if _gp.ndim == 3:
                    _gp = np.concatenate(_gp, axis=0)
                    _gc = np.concatenate(_gc, axis=0)
                    _isob = np.concatenate(_isob, axis=0)
                    _isig = np.concatenate(_isig, axis=0)
                    _inoi = np.concatenate(_inoi, axis=0)
                evaluation.evaluate_obs_grid(out_file, _gp, _gc, _isob, _isig, _inoi, args.n_grid_examples)

            if args.include_des:
                evaluation.evaluate_obs_des(model_fn, out_file, msfm_conf, dlss_conf)

            if args.include_buzzard:
                evaluation.evaluate_obs_buzzard(model_fn, out_file, msfm_conf, dlss_conf, args.buzzard_labels)

            if args.include_mocks:
                evaluation.evaluate_obs_benchmark(
                    model_fn, out_file, msfm_conf, dlss_conf, args.data_dir, args.mock_labels
                )

        if args.wandb and out_file is not None:
            LOGGER.info(f"Logged the predictions to weights & biases")
            wandb_artifact = wandb.Artifact(name="evaluation-predictions", type="predictions")
            wandb_artifact.add_file(local_path=out_file)
            wandb_run.log_artifact(wandb_artifact)

    if args.evaluate_all_checkpoints:
        LOGGER.warning(f"Evaluating all checkpoints")

        # checkpoints = model.checkpoint_manager.checkpoints
        # TODO
        checkpoints = model.checkpoint_manager.checkpoints[10:]
        for checkpoint in checkpoints:
            # model.checkpoint_manager.checkpoint.restore(checkpoint)
            model.restore_model_from_checkpoint_dir(checkpoint)
            evaluate_current_checkpoint(model)

    else:
        LOGGER.warning(f"Evaluating only the latest checkpoint")
        evaluate_current_checkpoint(model)

    # Release TF checkpoint objects explicitly so _CheckpointRestoreCoordinatorDeleter
    # is GC'd now, before interpreter shutdown nulls out TF module-level state.
    model.checkpoint = None
    model.checkpoint_manager = None
