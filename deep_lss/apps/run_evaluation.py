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
from deep_lss.nets import NETWORKS, TRANSFORMER_NETWORKS
from deep_lss.nets.maps_plus_cls_network import MapsPlusCLSNetwork
from deep_lss.nets.transformer_networks import HealpixTransformerNetwork, TransformerMapsPlusCLSNetwork
from deep_lss.nets.regression_head import get_cls_embedding_layers, get_regression_head

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
    parser.add_argument("--n_grid_examples", type=int, default=16)
    parser.add_argument("--include_des", action="store_true", help="evaluate DES Y3 catalogs")
    parser.add_argument("--include_buzzard", action="store_true", help="evaluate Buzzard N-body realizations")
    parser.add_argument("--buzzard_labels", nargs="+", default=["Buzzard_mean"])
    parser.add_argument("--include_mocks", action="store_true", help="evaluate mock observations from data_dir/obs/")
    parser.add_argument(
        "--mock_labels",
        nargs="+",
        default=None,
        help="mock labels to evaluate; if omitted, every *_obs_maps.h5 in data_dir/obs/ is evaluated",
    )
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

    # load the configs (migrates a legacy multi-document stream if needed)
    conf = configuration.load_run_configs(os.path.join(args.dir_model, "configs.yaml"))
    net_conf = conf["net"]
    dlss_conf = conf["dlss"]
    loss_conf = conf["loss"]
    data_conf = conf["data"]
    msfm_conf = conf["msfm"]

    LOGGER.info(f"Loaded configs from the model directory")

    # general constants
    all_params = msfm_conf["analysis"]["params"]
    target_params = dlss_conf["dset"]["training"]["params"]
    loss_func = conf["run"]["loss_func"]
    n_params = len(target_params)
    LOGGER.info(f"The networks have output shape {n_params} and target {target_params}")

    # pipeline constants
    n_side = msfm_conf["analysis"]["n_side"]
    data_vec_pix, _, _, _ = files.load_pixel_file(msfm_conf)

    # the network (and pipeline downsampling) run at the finest per-probe smoothing nside; probes
    # smoothed at a coarser nside (per-probe smooth_nside mapping) are handled inside the network
    smooth_nside, smooth_indices, parent_output_idx = configuration.resolve_smooth_nside(
        net_conf, dlss_conf, msfm_conf
    )

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
        n_output = loss_conf["mutual_info_loss"]["dim_summary_fac"] * n_params
    elif loss_func == "delta" or loss_func == "mse":
        n_output = n_params

    # set up directories
    checkpoint_dir = os.path.abspath(os.path.join(args.dir_model, "checkpoint"))

    # Maps+Cls is enabled by the presence of a `cls:` block (see run_training.py).
    cls_conf = net_conf["network"].get("cls", None)
    return_cls = cls_conf is not None
    if "cls_n_bins" in net_conf["network"]:
        raise ValueError(
            "Legacy flat Cls keys (cls_n_bins / cls_transform / cls_embedding_* / asinh_default_scale) "
            "are no longer supported — move them under a nested `cls:` block in the network config "
            "(see configs/transformer/lensing/maps+cls.yaml)."
        )
    if return_cls:
        LOGGER.warning("cls block detected in net_conf['network'] — building MapsPlusCLSNetwork for evaluation")

    max_batch_size = net_conf["dset"]["eval"]["grid"]["local_batch_size"]

    is_transformer = net_conf["network"]["name"] in TRANSFORMER_NETWORKS

    # create all of the variables within the strategy's scope, such that they are mirrored
    with strategy.scope():
        if is_transformer:
            # Nested hierarchical local-window transformer. Mirror the construction in
            # run_training.py: the maps are smoothed and reordered into nested superpixel
            # blocks inside the pre-built tf.keras.Model, so no HealpyGCNN graph is built and
            # n_neighbors is irrelevant. The network is traced with dummy inputs so that
            # network.built is True before BaseModel calls network.summary().
            token_nside = net_conf["network"]["token_nside"]
            transformer_kwargs = net_conf["network"]["kwargs"]
            # Mirror run_training.py: XLA-fuse the tokenizer->transformer body. Beyond the
            # speed-up, this is required for the larger configs (many heads / nested levels) to
            # evaluate at all — the eager attention softmax otherwise overflows the CUDA kernel
            # launch limit ("invalid configuration argument"). The key lives under `network:`,
            # not in `kwargs`, so it must be forwarded explicitly.
            jit_compile_body = net_conf["network"].get("jit_compile_body", False)

            # Mirror run_training.py: head.dropout_rate is a single Dropout right before the
            # final linear layer in both paths (inactive at eval, but the build must match).
            head_conf = net_conf["network"].get("head", {}) or {}
            fused_head_layers = head_conf.get("fused_layers", []) or None
            head_dropout = head_conf.get("dropout_rate", None)

            # Mirror run_training.py: input_norm adds the EmpiricalInputNormalization layer,
            # whose statistics (measured at training time) are restored from the checkpoint.
            input_norm = bool(net_conf["network"].get("input_norm", False))

            # Mirror run_training.py: masked_attention rebuilds the same static mask constants
            # (no checkpoint variables involved).
            masked_attention = bool(net_conf["network"].get("masked_attention", False))

            if return_cls:
                _, l_min_per_pair, l_max_per_pair = configuration.get_cls_bounds_per_pair(msfm_conf, dlss_conf)
                n_cls_bins = cls_conf.get("n_bins", 16)
                cls_emb_widths = cls_conf.get("embedding_layers", [512, 512, 512, 512])
                cls_emb_dropout = cls_conf.get("embedding_dropout_rate", None)
                # dense regression head minus the leading Flatten (the fused vector is already 2-D)
                regression_head_layers = get_regression_head(
                    out_features=n_output,
                    head_type="dense",
                    dense_layers=fused_head_layers,
                    dropout_rate=head_dropout,
                )[1:]
                network = TransformerMapsPlusCLSNetwork(
                    smoothing_kwargs=smoothing_kwargs,
                    smooth_indices=smooth_indices,
                    nside=smooth_nside,
                    token_nside=token_nside,
                    in_channels=n_z_bins,
                    map_feature_dim=net_conf["network"]["map_feature_dim"],
                    transformer_kwargs=transformer_kwargs,
                    tfr_n_side=n_side,
                    n_cls_bins=n_cls_bins,
                    l_min_per_pair=l_min_per_pair,
                    l_max_per_pair=l_max_per_pair,
                    cls_embedding_layers=get_cls_embedding_layers(cls_emb_widths, dropout_rate=cls_emb_dropout),
                    regression_head_layers=regression_head_layers,
                    cls_transform=cls_conf.get("transform", "asinh_per_feature"),
                    jit_compile_body=jit_compile_body,
                    input_norm=input_norm,
                    masked_attention=masked_attention,
                )
                network(
                    (tf.zeros((2, len(smooth_indices), n_z_bins)),
                     tf.zeros((2, 3 * n_side, len(l_min_per_pair)))),
                    training=False,
                )
            else:
                network = HealpixTransformerNetwork(
                    smoothing_kwargs=smoothing_kwargs,
                    smooth_indices=smooth_indices,
                    nside=smooth_nside,
                    token_nside=token_nside,
                    in_channels=n_z_bins,
                    num_outputs=n_output,
                    transformer_kwargs=transformer_kwargs,
                    jit_compile_body=jit_compile_body,
                    head_dropout_rate=head_dropout,
                    input_norm=input_norm,
                    masked_attention=masked_attention,
                )
                network(tf.zeros((2, len(smooth_indices), n_z_bins)), training=False)

            LOGGER.info(f"Built transformer network {net_conf['network']['name']} (return_cls={return_cls})")
            model = BaseModel(
                network=network,
                n_side=None,
                indices=None,
                n_neighbors=None,
                input_shape=None,
                max_batch_size=max_batch_size,
                checkpoint_dir=checkpoint_dir,
                restore_checkpoint=True,
                strategy=strategy,
            )
        else:
            net_spec = NETWORKS[net_conf["network"]["name"]](
                out_features=n_output, smoothing_kwargs=smoothing_kwargs, **net_conf["network"]["kwargs"]
            )
            LOGGER.info(f"Loaded a network specification of type {NETWORKS[net_conf['network']['name']]}")

            if return_cls:
                _, l_min_per_pair, l_max_per_pair = configuration.get_cls_bounds_per_pair(msfm_conf, dlss_conf)
                n_cls_bins = cls_conf.get("n_bins", 16)
                cls_emb_widths = cls_conf.get("embedding_layers", [512, 512, 512, 512])
                cls_emb_dropout = cls_conf.get("embedding_dropout_rate", None)
                network = MapsPlusCLSNetwork(
                    conv_layers=net_spec.get_conv_layers(),
                    cls_embedding_layers=get_cls_embedding_layers(cls_emb_widths, dropout_rate=cls_emb_dropout),
                    regression_head_layers=net_spec.get_head_layers_no_flatten(),
                    n_side=smooth_nside,
                    tfr_n_side=n_side,
                    indices=smooth_indices,
                    n_neighbors=net_conf["network"]["n_neighbors"],
                    max_batch_size=max_batch_size,
                    initial_Fin=n_z_bins,
                    n_cls_bins=n_cls_bins,
                    l_min_per_pair=l_min_per_pair,
                    l_max_per_pair=l_max_per_pair,
                    cls_transform=cls_conf.get("transform", "asinh_per_feature"),
                )
                network.gcnn.build((max_batch_size, len(smooth_indices), n_z_bins))
                # Trace the full MapsPlusCLSNetwork so that network.built=True and BaseModel
                # can call network.summary(). gcnn.build() only builds the map branch.
                network(
                    (tf.zeros((2, len(smooth_indices), n_z_bins)), tf.zeros((2, 3 * n_side, len(l_min_per_pair)))),
                    training=False,
                )
                model = BaseModel(
                    network=network,
                    n_side=None,
                    indices=None,
                    n_neighbors=net_conf["network"]["n_neighbors"],
                    input_shape=None,
                    max_batch_size=max_batch_size,
                    checkpoint_dir=checkpoint_dir,
                    restore_checkpoint=True,
                    strategy=strategy,
                )
            else:
                network = net_spec.get_layers()
                model = BaseModel(
                    network=network,
                    n_side=smooth_nside,
                    indices=smooth_indices,
                    n_neighbors=net_conf["network"]["n_neighbors"],
                    input_shape=(None, len(smooth_indices), n_z_bins),
                    max_batch_size=max_batch_size,
                    checkpoint_dir=checkpoint_dir,
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

    if return_cls:

        def _call_model(x, cls_raw):
            # x: (B, n_pix_dv, n_ch); cls_raw: (B, n_ell, n_z_cross), precomputed consistently
            # with training by forward_model_observation_map (same alm/smoothing pipeline that
            # produces the Cls baked into the grid TFRecords) — passed in by evaluate_obs_*.
            # HealpySmoothing's n_matmul_splits requires batch dim divisible by 2.
            if x.shape[0] == 1:
                x = np.concatenate([x, x], axis=0)
                cls_raw = np.concatenate([cls_raw, cls_raw], axis=0)
                return model((x, cls_raw), training=False).numpy()[:1]
            return model((x, cls_raw), training=False).numpy()

    else:

        def _call_model(x, cls_raw=None):
            # HealpySmoothing pre-computes n_matmul_splits=2 for this pixel resolution;
            # tf.split requires the batch dim divisible by 2, so pad batch=1 → 2.
            if x.shape[0] == 1:
                x = np.concatenate([x, x], axis=0)
                return model(x, training=False).numpy()[:1]
            return model(x, training=False).numpy()

    if parent_output_idx is not None:
        model_fn = lambda x, cls_raw=None: _call_model(_downsample(x), cls_raw)
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
                data_conf=data_conf,
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
                data_conf=data_conf,
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
                data_conf=data_conf,
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
                mock_labels = args.mock_labels
                if mock_labels is None:
                    mock_labels = evaluation.discover_mock_labels(args.data_dir)
                    LOGGER.info(f"Auto-discovered {len(mock_labels)} mock(s) in {args.data_dir}/obs: {mock_labels}")
                evaluation.evaluate_obs_benchmark(model_fn, out_file, msfm_conf, dlss_conf, args.data_dir, mock_labels)

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
            model.restore_model_from_checkpoint_path(checkpoint)
            evaluate_current_checkpoint(model)

    else:
        LOGGER.warning(f"Evaluating only the latest checkpoint")
        evaluate_current_checkpoint(model)

    # Release TF checkpoint objects explicitly so _CheckpointRestoreCoordinatorDeleter
    # is GC'd now, before interpreter shutdown nulls out TF module-level state.
    model.checkpoint = None
    model.checkpoint_manager = None
