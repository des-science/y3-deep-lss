# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Train the DeepSphere graph neural networks at the fiducial cosmology and its perturbations using the information
maximizing loss to find an informative summary statistic.

Meant for the GPU nodes of the Perlmutter cluster at NERSC.
"""

import os
import sys
import threading
import warnings


def _filter_stderr():
    fd = sys.stderr.fileno()
    saved_fd = os.dup(fd)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, fd)
    os.close(write_fd)
    saved_stderr = os.fdopen(saved_fd, "w")

    def pump():
        with os.fdopen(read_fd, "r") as f:
            for line in f:
                if "gpu_timer.cc:114" not in line and "+ptx85" not in line:
                    saved_stderr.write(line)
                    saved_stderr.flush()

    t = threading.Thread(target=pump, daemon=True)
    t.start()


_filter_stderr()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["NUMBA_WARNINGS"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)

import tensorflow as tf
import horovod.tensorflow as hvd
import argparse
import yaml
import wandb
import shutil

from datetime import datetime
from time import time
from contextlib import nullcontext

from msfm.fiducial_pipeline import FiducialPipeline
from msfm.grid_pipeline import GridPipeline
from msfm.utils import logger, input_output, files, parameters

from deep_lss.utils import (
    distribute,
    configuration,
    evaluation,
    optimization,
    delta_loss,
    throughput,
    training_helpers,
)
from deep_lss.models.delta_model import DeltaLossModel
from deep_lss.models.grid_model import GridLossModel
from deep_lss.utils.distribute import HorovodStrategy
from deep_lss.nets import NETWORKS, TRANSFORMER_NETWORKS
from deep_lss.nets.composite.resnet_summary import ResNetSummaryNetwork
from deep_lss.nets.composite.transformer_summary import TransformerSummaryNetwork
from deep_lss.nets.encoders.maps.gcnn.resnet_multires import ResNetMultiResEncoder
from deep_lss.nets.layers.maps.input_normalization import compute_input_norm_stats
from deep_lss.nets.heads.regression_head import get_regression_head
from deep_lss.nets.layers.cls.embedding import get_cls_branch_kwargs

LOGGER = logger.get_logger(__file__)

# Keys present in dlss.yaml dset.common that are only meaningful for Cls (2pt) training
# and unknown to FiducialPipeline / GridPipeline — strip them before splatting into pipe_kwargs.
_CLS_ONLY_KEYS = frozenset({"with_cross_z", "with_cross_probe", "lenses_before_sources", "ggl_only"})


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
        "--loss_function",
        type=str,
        default=None,
        choices=["delta", "mse", "likelihood", "mutual_info"],
        help="loss function to train with. If omitted, read from loss_function key in the loss config.",
    )
    parser.add_argument(
        "--dist_strategy",
        choices=[None, "mirrored", "multi_worker_mirrored", "horovod"],
        default=None,
        help="distribution strategy, use None to run locally",
    )
    parser.add_argument(
        "--train_tfr_pattern",
        type=str,
        required=True,
        help="input root dir of the fiducial or grid data vectors (training)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "input data dir holding the binned-Cls cache (cls/rebinned_nb*.h5), used to fit the"
            " per-feature asinh scale for cls_transform=asinh_per_feature. If None, derived from"
            " --train_tfr_pattern (the part before /tfrecords/)."
        ),
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
        "--fidu_eval_tfr_pattern",
        type=str,
        default=None,
        help="input root dir of the fiducial data vectors (evaluation)",
    )
    parser.add_argument(
        "--grid_eval_tfr_pattern",
        type=str,
        default=None,
        help="input root dir of the grid data vectors (evaluation)",
    )
    parser.add_argument(
        "--dir_base",
        type=str,
        default=None,
        help="base dir where the models are saved. If None, a dir within the repo is generated according to the config",
    )
    parser.add_argument(
        "--dir_model",
        type=str,
        default=None,
        help="dir where the model summaries and checkpoints are saved. If None, a dir is generated according to the"
        " current date and time. This dir is appended to the dir_base as a relative path. Passing an absolute path"
        " overrides this.",
    )
    parser.add_argument(
        "--net_config",
        type=str,
        default="config/resnet_vanilla.yaml",
        help=(
            "configuration .yaml file of the model to be trained. None can only be provided if there's a config in"
            " the dir_model and restore_checkpoint is true."
        ),
    )
    parser.add_argument("--probes_config", type=str, default=None, help="probe/parameter config (configs/probes/)")
    parser.add_argument("--scales_config", type=str, default=None, help="scale-cut config (configs/scales/)")
    parser.add_argument("--loss_config", type=str, default=None, help="loss function config (configs/loss/)")
    parser.add_argument("--data_config", type=str, default=None, help="train/test split config (configs/data/)")
    parser.add_argument(
        "--msfm_config",
        type=str,
        default=None,
        help=(
            "configuration .yaml file of the multiprobe-simulation-forward-model pipeline. None means that the"
            " standard configuration file in configs/config.yaml relative to the msfm repo is loaded."
        ),
    )
    parser.add_argument(
        "--restore_checkpoint",
        action="store_true",
        help=(
            "restore the model from a checkpoint instead of initializing it from scratch."
            " Additionally, the configs are loaded from the path in this case"
        ),
    )
    parser.add_argument("--evaluate_training_set", action="store_true", help="evaluate the training set")
    parser.add_argument("--slurm_output", type=str, default=None, help="path to the slurm output file")
    parser.add_argument(
        "--n_steps",
        type=int,
        default=None,
        help="override training.n_steps from the config. Intended for throughput probes, which run the real "
        "geometry for a few thousand steps; without it a probe has to rewrite the yaml. On a RESTORED run this "
        "overrides the value saved in the run directory, which is the only way to resize a chain between jobs.",
    )
    parser.add_argument(
        "--wall_budget_seconds",
        type=float,
        default=None,
        help="override training.wall_budget_seconds: train for this many seconds instead of for a fixed number of "
        "steps, annealing the cosine to zero exactly when the budget runs out. See "
        "deep_lss.utils.throughput.WallClockBudget for why a fixed step count is hard to size correctly.",
    )
    parser.add_argument(
        "--job_budget_seconds",
        type=float,
        default=None,
        help="override training.job_budget_seconds: this job's share of the wall-clock budget. Job 1 of a chain "
        "stops here and checkpoints cleanly instead of being killed at the wall. Defaults to the whole budget.",
    )

    parser.add_argument("--debug", action="store_true", help="activate debug mode")
    parser.add_argument("--profile", action="store_true", help="run the profiler")
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision training")
    parser.add_argument(
        "--mixed_precision_dtype",
        type=str,
        default="float16",
        choices=("float16", "bfloat16"),
        help="mixed precision dtype to use when --mixed_precision is enabled",
    )
    parser.add_argument("--xla", action="store_true", help="enable XLA (Accelerated Linear Algebra) JIT compilation")
    parser.add_argument(
        "--summary_every",
        type=int,
        default=100,
        help="log step_time and global_step summaries every N training steps. This gates the only "
        "per-step host<->device sync (get_step's .numpy() at write time), so the default of 100 "
        "avoids syncing every step; set to 1 to restore the previous per-step behavior.",
    )

    parser.add_argument("--wandb", action="store_true", help="log to weights & biases, otherwise log to tensorboard")
    parser.add_argument("--wandb_tags", nargs="+", type=str, default=None, help="tags for weights & biases")
    parser.add_argument("--wandb_notes", type=str, default=None, help="notes for weights & biases (longer than tags)")
    parser.add_argument("--wandb_sweep_id", type=str, default=None, help="id of the sweep. If None, no sweep is used")

    parser.add_argument("--pasc_throughput", action="store_true")

    args, _ = parser.parse_known_args()

    if args.summary_every < 1:
        raise ValueError(f"summary_every must be >= 1, got {args.summary_every}")

    assert not (
        (args.fidu_vali_tfr_pattern is not None) and (args.grid_vali_tfr_pattern is not None)
    ), "Only one of the validation sets can be provided"

    # set up directories
    file_dir = os.path.dirname(__file__)
    args.repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))

    if args.dir_base is None:
        args.dir_base = os.path.join(args.repo_dir, "run_files")
        os.makedirs(args.dir_base, exist_ok=True)
        LOGGER.info(f"Created base directory {args.dir_base}")

    if args.slurm_output is not None:
        args.slurm_output = os.path.abspath(args.slurm_output)

    # print arguments
    logger.set_all_loggers_level(args.verbosity)
    for key, value in vars(args).items():
        LOGGER.info(f"{key} = {value}")

    # The numerical-precision policy is resolved in training() once net_conf is loaded: the
    # network config supplies the default (network.precision, defaulting to full float32) and
    # the --mixed_precision CLI flag overrides it. It must be set there, before the network
    # (incl. the HealpySmoothing sparse kernel) is built, so every layer adopts the policy.

    if args.xla:
        LOGGER.warning(
            "Using XLA jit compilation. This doesn't work in most cases, as the SparseDenseMatrixMultiplication "
            "(DeepSphere smoothing and graph convolutions) and MatrixDeterminant (delta loss) operators are not "
            "supported"
        )

        if args.dist_strategy == "mirrored":
            LOGGER.warning("XLA + MirroredStrategy freezes for unknown reasons")
        elif args.dist_strategy == "horovod":
            LOGGER.warning(
                "XLA + HorovodStrategy freezes for unknown reasons, see https://horovod.readthedocs.io/en/latest/xla.html"
            )

    if args.debug:
        tf.config.run_functions_eagerly(True)
        # tf.config.set_soft_device_placement(False)
        # tf.debugging.set_log_device_placement(True)
        # tf.data.experimental.enable_debug_mode()
        LOGGER.warning("!!!!! Running the training in test mode, TensorFlow is executed eagerly !!!!!")

    physical_devices = tf.config.list_physical_devices("GPU")
    try:
        for device in physical_devices:
            if device.device_type == "GPU":
                tf.config.experimental.set_memory_growth(device, True)
        LOGGER.info("Configured the GPUs to memory growth mode")
    except (RuntimeError, ValueError):
        # Invalid device or cannot modify virtual devices once initialized.
        LOGGER.warning(
            "Could not configure the GPUs to memory growth mode, all available GPU memory is reserved for TensorFlow"
        )

    if not args.restore_checkpoint:
        for flag in ("probes_config", "loss_config", "data_config"):
            if getattr(args, flag) is None:
                parser.error(f"--{flag} is required for a fresh run")

    return args


def training(args=None):
    LOGGER.timer.start("main")

    if args is None:
        args = setup()

    # hardware and distribution
    _, _ = distribute.check_devices()
    strategy = distribute.get_strategy(args.dist_strategy)

    # initialize a fresh model
    if not args.restore_checkpoint:
        # load the configs
        net_conf = input_output.read_yaml(os.path.join(args.repo_dir, args.net_config))
        dlss_conf = configuration.read_split_configs(args.probes_config, args.scales_config)
        loss_conf = input_output.read_yaml(args.loss_config)
        data_conf = input_output.read_yaml(args.data_config)
        msfm_conf = files.load_config(args.msfm_config)
        if args.loss_function is None:
            args.loss_function = loss_conf.get("loss_function")
        if args.loss_function is None:
            raise ValueError(
                "loss_function not set; either pass --loss_function or use a --loss_config with a loss_function key"
            )
        LOGGER.info("Loaded configs from the provided paths")

        if args.dir_model is None:
            net_name = net_conf["name"]
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            args.dir_model = f"{now}_{net_name}"
            LOGGER.info(f"Created model directory {args.dir_model}")

        # make output directory
        dir_model = os.path.join(args.dir_base, args.dir_model)
        os.makedirs(dir_model, exist_ok=True)
        LOGGER.info(f"Created output directory {dir_model}")

        # runtime metadata (kept as its own top-level block in the saved config)
        run_conf = {
            "dir_model": dir_model,
            "dir_log": args.slurm_output,
            "loss_func": args.loss_function,
            "dist_strategy": args.dist_strategy,
        }

        # save the configs as a single nested mapping
        with open(os.path.join(dir_model, "configs.yaml"), "w") as f:
            yaml.dump(
                {
                    "net": net_conf,
                    "dlss": dlss_conf,
                    "loss": loss_conf,
                    "data": data_conf,
                    "msfm": msfm_conf,
                    "run": run_conf,
                },
                f,
            )

    # restore a saved model
    elif args.restore_checkpoint and (args.dir_model is not None):
        # make output directory
        dir_model = os.path.join(args.dir_base, args.dir_model)
        os.makedirs(dir_model, exist_ok=True)
        LOGGER.info(f"Created output directory {dir_model}")

        # load the configs (migrates a legacy multi-document stream if needed)
        conf = configuration.load_run_configs(os.path.join(dir_model, "configs.yaml"))
        net_conf = conf["net"]
        dlss_conf = conf["dlss"]
        loss_conf = conf["loss"]
        data_conf = conf["data"]
        msfm_conf = conf["msfm"]

        if args.loss_function is None:
            args.loss_function = conf["run"]["loss_func"]
        LOGGER.info("Loaded configs from the model directory")

    else:
        raise ValueError("Can't restore the model from an unspecified dir_model")

    # CLI overrides of the step budget, applied to BOTH paths above. On a restored run this is the
    # only way to change the budget: the run reads configs.yaml from its own directory, so editing
    # the repo yaml between chained jobs silently does nothing.
    if args.n_steps is not None:
        LOGGER.warning(f"Overriding training.n_steps {net_conf['training']['n_steps']} -> {args.n_steps} from the CLI")
        net_conf["training"]["n_steps"] = args.n_steps
    if args.wall_budget_seconds is not None:
        LOGGER.warning(f"Overriding training.wall_budget_seconds -> {args.wall_budget_seconds} s from the CLI")
        net_conf["training"]["wall_budget_seconds"] = args.wall_budget_seconds
    if args.job_budget_seconds is not None:
        LOGGER.warning(f"Overriding training.job_budget_seconds -> {args.job_budget_seconds} s from the CLI")
        net_conf["training"]["job_budget_seconds"] = args.job_budget_seconds

    # numerical precision: net_conf["network"]["precision"] is the default (float32 = full
    # precision); the --mixed_precision CLI flag overrides it. Set the global Keras policy here,
    # before the network (incl. the HealpySmoothing sparse kernel, which casts itself to the
    # policy's compute dtype) is built inside strategy.scope() below.
    precision = net_conf["network"].get("precision", "float32")
    if args.mixed_precision:
        precision = args.mixed_precision_dtype  # CLI overrides the config
    if precision not in ("float32", "float16", "bfloat16"):
        raise ValueError(f"Unknown precision '{precision}'; expected float32, float16 or bfloat16")
    if precision != "float32":
        policy_name = f"mixed_{precision}"
        LOGGER.warning(f"Using mixed precision policy {policy_name}")
        tf.keras.mixed_precision.set_global_policy(policy_name)
        if args.loss_function == "delta":
            LOGGER.warning("Mixed precision with the delta loss is not recommended, training tends to be unstable")
    else:
        LOGGER.info("Using full float32 precision")

    # to be read by the evaluation script
    job_id = os.environ["SLURM_JOB_ID"]
    if job_id is not None:
        temp_file = f"./.env_var/id_{job_id}.txt"
        os.makedirs(os.path.dirname(temp_file), exist_ok=True)
        LOGGER.info(f"Writing the model directory to {temp_file}")
        with open(temp_file, "w") as f:
            f.write(dir_model)

    # weights and biases
    if args.wandb:
        group_name = distribute.get_wandb_group_name(strategy)

        # check if there's an existing run ID to resume
        wandb_id_file = os.path.join(dir_model, "wandb_run_id.txt")
        existing_run_id = None

        if os.path.exists(wandb_id_file) and args.restore_checkpoint:
            with open(wandb_id_file, "r") as f:
                existing_run_id = f.read().strip()
            LOGGER.info(f"Found existing wandb run ID: {existing_run_id}")

        if existing_run_id:
            wandb_run = wandb.init(
                id=existing_run_id,
                resume="allow",
                project="y3-deep-lss",
                dir=dir_model,
                group=group_name,
                job_type="training",
                # make sure that wandb logs to the cloud
                mode="online",
                force=True,
                # to be able to log within graph mode
                sync_tensorboard=True,
                # additional metadata
                tags=args.wandb_tags,
                notes=args.wandb_notes,
            )
            LOGGER.info(f"Resumed wandb run: {existing_run_id}")
        else:
            wandb_run = wandb.init(
                project="y3-deep-lss",
                dir=dir_model,
                group=group_name,
                job_type="training",
                mode="online",
                force=True,
                sync_tensorboard=True,
                tags=args.wandb_tags,
                notes=args.wandb_notes,
            )
            LOGGER.info(f"Created new wandb run: {wandb_run.id}")

            # Save the run ID for future resumption
            with open(wandb_id_file, "w") as f:
                f.write(wandb_run.id)

        if args.wandb_sweep_id is not None:
            if isinstance(strategy, HorovodStrategy):
                # only the chief gets an agent, which provides the hyperparameters
                if hvd.rank() == 0:
                    nested_hyperparam_conf = configuration.convert_dotted_to_nested_dict(wandb_run.config)
                    net_conf = configuration.update_nested_dict(net_conf, nested_hyperparam_conf["net"])

                net_conf = strategy.broadcast_object(net_conf, root_rank=0)
                LOGGER.info("Broadcast the chief/agent's hyperparameters to the other ranks")

            else:
                # in the wandb sweep config, the hyperparameters are defined like net.optimization.optimizer, while the
                # .yaml config files are structured as nested dictionaries
                nested_hyperparam_conf = configuration.convert_dotted_to_nested_dict(wandb_run.config)

                # dict.update() would discard branches that are not present in the update dict
                net_conf = configuration.update_nested_dict(net_conf, nested_hyperparam_conf["net"])

        # only update the config here instead of in the init so that possible changes by a sweep agent are included
        wandb_run.config.setdefaults({"msfm": msfm_conf, "dlss": dlss_conf, "net": net_conf})

        wandb.define_metric("train_step")
        for prefix in (
            "loss/*",
            "schedule/*",
            "learning_rate",
            "global_grad_norm*",
            "step_time",
            "data_time",
            "compute_time",
            "z_bank/*",
            "z_invariance/*",
        ):
            wandb.define_metric(prefix, step_metric="train_step")

        LOGGER.info(f"Initialized weights & biases to {dir_model}")
        LOGGER.info(f"Running with {strategy.num_replicas_in_sync} replicas")

    LOGGER.info(f"TensorFlow version {tf.__version__}")

    # set up subdirectories
    checkpoint_dir = os.path.abspath(os.path.join(dir_model, "checkpoint"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    summary_dir = os.path.abspath(os.path.join(dir_model, "summary"))
    os.makedirs(summary_dir, exist_ok=True)

    # constants: msfm
    n_side = msfm_conf["analysis"]["n_side"]
    data_vec_pix, _, _, _ = files.load_pixel_file(msfm_conf)

    # the network (and pipeline downsampling) run at the finest per-probe smoothing nside; probes
    # smoothed at a coarser nside (per-probe smooth_nside mapping) are handled inside the network
    smooth_nside, smooth_indices, parent_output_idx = configuration.resolve_smooth_nside(
        net_conf, dlss_conf, msfm_conf
    )

    # the grid tfrecords store bary_Mc raw (CosmoGrid convention, ~1e12 - 1e15) while the priors,
    # fiducials and inference all use log10(Mc) -- convert the label column(s) at load time. This
    # is the tf.data mirror of parameters.raw_to_prior_units (used on the numpy label gathers in
    # deep_lss.utils.cls_preprocessing and msi.utils.preprocessing); both are driven by the shared
    # parameters.LOG10_PARAMS list, so theta reaches the loss in prior units as the theta
    # standardization below assumes.
    def _log10_label_columns(pipeline_params):
        idxs = [pipeline_params.index(p) for p in parameters.LOG10_PARAMS if p in pipeline_params]
        if not idxs:
            return None

        def _convert(*batch):
            *rest, cosmo, index = batch
            log10 = tf.math.log(tf.constant(10.0, cosmo.dtype))
            for i in idxs:
                col = cosmo[:, i]
                assert_raw = tf.debugging.assert_greater(
                    tf.reduce_min(col),
                    tf.constant(parameters.LOG10_RAW_MIN, cosmo.dtype),
                    message="expected raw bary_Mc labels, got log10(Mc)?",
                )
                with tf.control_dependencies([assert_raw]):
                    log_col = tf.math.log(col[:, None]) / log10
                cosmo = tf.concat([cosmo[:, :i], log_col, cosmo[:, i + 1 :]], axis=1)
            return (*rest, cosmo, index)

        return _convert

    # every train/vali/adapt dataset uses the same in-network nside downsampling; specify it once
    def build_dset(pipeline, tfr_pattern, ds_kwargs, input_context=None):
        dset = pipeline.get_dset(
            tfr_pattern=tfr_pattern,
            **ds_kwargs,
            input_context=input_context,
            downsample_nside=smooth_nside if parent_output_idx is not None else None,
            parent_output_idx=parent_output_idx,
        )
        if isinstance(pipeline, GridPipeline):
            convert = _log10_label_columns(pipeline.params)
            if convert is not None:
                dset = dset.map(convert)
        return dset

    # constants: deep_lss
    params = dlss_conf["dset"]["training"]["params"]
    n_params = len(params)
    LOGGER.info(f"Training with respect to the {n_params} parameters {params}")

    with_lensing = dlss_conf["dset"]["common"]["with_lensing"]
    with_clustering = dlss_conf["dset"]["common"]["with_clustering"]
    with_cross = dlss_conf["dset"]["common"].get("with_cross", False)
    # Maps+Cls is enabled by the presence of a `cls:` block in the network config; its keys
    # (n_bins, transform, embedding_layers, embedding_dropout_rate, asinh_default_scale) configure
    # the Cls branch. cls_transform: "asinh_per_feature" (default) or "log1p_fixed".
    cls_conf = net_conf["network"].get("cls", None)
    return_cls = cls_conf is not None
    cls_transform = (cls_conf or {}).get("transform", "asinh_per_feature")
    if "cls_n_bins" in net_conf["network"]:
        raise ValueError(
            "Legacy flat Cls keys (cls_n_bins / cls_transform / cls_embedding_* / asinh_default_scale) "
            "are no longer supported — move them under a nested `cls:` block in the network config "
            "(see configs/maps/prod/transformer/lensing/maps+cls.yaml)."
        )
    if return_cls:
        LOGGER.warning(
            f"cls block detected in net_conf['network'] — the summary network will concatenate a "
            f"Cls branch onto the map features (cls_transform={cls_transform})"
        )

    # constants: network
    n_steps = net_conf["training"]["n_steps"]
    output_every = net_conf["training"]["output_every"]
    checkpoint_every = net_conf["training"]["checkpoint_every"]
    vali_every = net_conf["training"]["vali_every"]
    eval_every = net_conf["training"]["eval_every"]
    # fail-fast NaN watchdog: check the training loss every `nan_check_every` steps and abort
    # after `nan_abort_after` consecutive non-finite checks. The grad-zeroing safety net in
    # base_model turns a NaN step into a no-op, so without this a diverged run would silently
    # burn its whole allocation as a frozen network. Optional config keys (default: hard-abort).
    nan_check_every = net_conf["training"].get("nan_check_every", 100)
    nan_abort_after = net_conf["training"].get("nan_abort_after", 1)

    # Optional wall-clock training budget. When set, `n_steps` stops being the length of the run and
    # becomes only a safety cap: the loop trains until `wall_budget_seconds` of training time have
    # been spent (summed over a chain) and the cosine anneals to zero at exactly that point, so the
    # eval/inference tail is guaranteed whatever rate the run happens to achieve. Absent, everything
    # below behaves exactly as before. See deep_lss.utils.throughput.WallClockBudget.
    wall_budget_seconds = net_conf["training"].get("wall_budget_seconds", None)
    # per-job share of the budget; job 1 of a chain stops here and checkpoints instead of being
    # killed at the wall, so the handover costs nothing
    job_budget_seconds = net_conf["training"].get("job_budget_seconds", None)

    if n_steps == "auto":
        # `auto` means "however many steps the budget buys" — there is then no meaningful cap, so use
        # one large enough never to bind while still bounding the loop.
        if wall_budget_seconds is None:
            raise ValueError("training.n_steps: auto requires training.wall_budget_seconds to be set")
        n_steps = 100_000_000
        LOGGER.info("n_steps: auto — the run is bounded by its wall-clock budget alone")
    elif wall_budget_seconds is not None and n_steps <= 0:
        raise ValueError(f"training.n_steps must be a positive cap or 'auto', got {n_steps}")

    # constants: miscellaneous
    if args.loss_function == "delta":
        assert "fiducial" in args.train_tfr_pattern, "The delta loss can only be used for the fiducial dataset"
    else:
        assert "grid" in args.train_tfr_pattern, f"The {args.loss_function} loss can only be used for the grid dataset"
    training_type = "fiducial" if args.loss_function == "delta" else "grid"
    smoothing_kwargs = configuration.get_smoothing_kwargs(
        args.loss_function, msfm_conf, dlss_conf, net_conf, dir_base=dir_model
    )

    dset_kwargs = {**net_conf["dset"]["training"]["common"], **data_conf}
    noise_kwargs = {}
    if args.loss_function == "delta":
        Pipeline = FiducialPipeline
        Model = DeltaLossModel
        n_output = n_params
        dset_kwargs.update(net_conf["dset"]["training"]["fiducial"])
        local_batch_size = dset_kwargs["local_batch_size"]
        effective_local_batch_size = local_batch_size * (2 * n_params + 1)

        try:
            noise_schedule_steps = net_conf["optimization"]["noise_schedule_steps"]
        except KeyError:
            noise_schedule_steps = None
        if noise_schedule_steps is not None:
            LOGGER.warning(
                f"Using a linearly increasing noise scheduler from 0 to 1 with {noise_schedule_steps} steps"
            )
            noise_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=0.0, decay_steps=noise_schedule_steps, end_learning_rate=1.0, power=1.0
            )
            noise_scale = tf.Variable(noise_scheduler(0), trainable=False, dtype=tf.float32)
            noise_kwargs = {"shape_noise_scale": noise_scale, "poisson_noise_scale": noise_scale}

    else:
        if args.loss_function == "likelihood":
            n_output = n_params + n_params * (n_params + 1) // 2
        elif args.loss_function == "mse":
            n_output = n_params
        elif args.loss_function == "mutual_info":
            n_output = loss_conf["mutual_info_loss"]["dim_summary_fac"] * n_params
        Pipeline = GridPipeline
        Model = GridLossModel
        dset_kwargs.update(net_conf["dset"]["training"]["grid"])
        local_batch_size = dset_kwargs["local_batch_size"]
        effective_local_batch_size = local_batch_size

    try:
        n_z_bins = len(dset_kwargs["z_bin_inds"])
    except (KeyError, TypeError):
        n_z_bins = 0
        if with_lensing:
            n_z_bins += len(msfm_conf["survey"]["metacal"]["z_bins"])
        if with_clustering:
            n_z_bins += len(msfm_conf["survey"]["maglim"]["z_bins"])
        if with_cross:
            n_z_bins += len(msfm_conf["survey"]["metacal"]["z_bins"]) * len(msfm_conf["survey"]["maglim"]["z_bins"])

    # dataset
    LOGGER.info("Training set")
    pipe_kwargs = {
        k: v
        for k, v in {**dlss_conf["dset"]["common"], **dlss_conf["dset"]["training"], **noise_kwargs}.items()
        if k not in _CLS_ONLY_KEYS
    }
    pipe_kwargs["return_maps"] = True
    pipe_kwargs["return_cls"] = return_cls
    train_pipeline = Pipeline(conf=msfm_conf, **pipe_kwargs)

    # like https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
    def train_dataset_fn(input_context):
        return build_dset(train_pipeline, args.train_tfr_pattern, dset_kwargs, input_context)

    dist_dset = strategy.distribute_datasets_from_function(train_dataset_fn)
    dist_iter = iter(dist_dset)

    # A chain shares ONE budget: pick up whatever earlier jobs already spent, so the cosine continues
    # along a single global curve instead of restarting at full learning rate in job 2.
    budget = None
    if wall_budget_seconds is not None:
        prior = throughput.read_budget_state(dir_model) if args.restore_checkpoint else None
        budget = throughput.WallClockBudget(
            total_seconds=float(wall_budget_seconds),
            job_seconds=job_budget_seconds,
            consumed_seconds=(prior or {}).get("consumed_seconds", 0.0),
        )
        budget.warmup_end_seconds = (prior or {}).get("warmup_end_seconds", None)
        LOGGER.info(
            f"Wall-clock training budget: {budget.total_seconds:.0f} s total, "
            f"{budget.consumed_seconds:.0f} s already spent by earlier jobs, "
            f"{budget.job_seconds:.0f} s allowed in this job (n_steps={n_steps} is now only a cap)"
        )

    # network, create all of the variables within the strategy's scope, such that they are mirrored
    with strategy.scope():
        optimizer = optimization.get_optimizer(net_conf, args.loss_function, args.restore_checkpoint, budget=budget)

        # Validate the network name up front with a friendly error (mirrors the Cls app); the
        # bare NETWORKS[...] lookups below would otherwise raise an opaque KeyError.
        net_name = net_conf["network"]["name"]
        valid_net_names = set(NETWORKS) | set(TRANSFORMER_NETWORKS)
        if net_name not in valid_net_names:
            raise ValueError(f"Unknown network.name={net_name!r}; expected one of {sorted(valid_net_names)}")

        is_transformer = net_name in TRANSFORMER_NETWORKS

        # A mixed per-probe smooth_nside mapping (combined probes, e.g. clustering at 256) yields
        # a split_probes smoothing spec. On the GCNN path it selects ResNetMultiResEncoder: the
        # coarse probe is smoothed at its own nside and injected (concat+Dense, the transformer
        # injection idiom) after the pooling layer that already runs at that nside, instead of
        # being smoothed with the shared base kernel at full resolution.
        is_multires_gcnn = (not is_transformer) and "split_probes" in smoothing_kwargs
        if is_multires_gcnn and net_conf["network"]["name"] != "resnet":
            raise ValueError(
                "Per-probe smooth_nside on the GCNN path is only supported for network.name=resnet "
                "(ResNetMultiResEncoder)"
            )

        # `input_norm: true` standardizes the smoothed maps with statistics measured from
        # training data on fresh runs and restored from the checkpoint otherwise (see the
        # measurement block below). Supported by the transformer encoders, ResNetLayers, and
        # ResNetMultiResEncoder; adds checkpoint variables — keep it consistent with the run's
        # lineage.
        input_norm = bool(net_conf["network"].get("input_norm", False))
        # set per branch below; owns the smooth_groups/masks/load_input_norm_stats interface
        norm_owner = None

        # sparse-matmul backend for the DeepSphere graph convolutions (GCNN/resnet path only):
        # "coo" (default, the unchanged tf.sparse.sparse_dense_matmul kernel), "csr" (cuSPARSE
        # csrmm) or "gather" (XLA-friendly gather+reduce). See deepsphere.utils.make_spmm_operator.
        # Numerically equivalent to "coo" up to float32 tolerance, so it does not change the run
        # lineage. Ignored on the transformer path (no graph convolutions there).
        spmm_backend = net_conf["network"].get("spmm_backend", "csr")

        if is_transformer:
            # Nested hierarchical local-window transformer. The maps are smoothed (same
            # HealpySmoothing front-end as the GCNNs) and reordered into nested superpixel
            # blocks inside the pre-built tf.keras.Model, so no HealpyGCNN graph is built and
            # n_neighbors is irrelevant.
            token_nside = net_conf["network"]["token_nside"]
            transformer_kwargs = net_conf["network"]["kwargs"]
            # XLA-fuse the tokenizer->transformer body (smoothing stays eager). Off by default;
            # enable per-config to cut the many tiny attention/layernorm/reshape kernel launches.
            jit_compile_body = net_conf["network"].get("jit_compile_body", False)

            # `head:` block, honored by BOTH transformer paths: dropout_rate is a single Dropout
            # right before the final linear layer (after token pooling in maps-only, after fusion
            # in maps+cls); fused_layers configures the post-fusion dense stack and therefore
            # requires a `cls:` block.
            head_conf = net_conf["network"].get("head", {}) or {}
            fused_head_layers = head_conf.get("fused_layers", []) or None
            head_dropout = head_conf.get("dropout_rate", None)
            if fused_head_layers and not return_cls:
                raise ValueError(
                    "head.fused_layers is set but there is no cls: block — a maps-only run has "
                    "nothing to fuse. Remove head.fused_layers or add a cls: block."
                )
            # map_feature_dim is the width the map feature is projected to FOR the concatenation,
            # so it is fusion configuration: required with a `cls:` block, meaningless without one.
            # Maps-only it defaults to None (no projection), which is the only defensible value
            # there — the regression head opens with LayerNorm and ends in Dense(n_output), so a
            # projection would be a second linear layer with no nonlinearity between the two. That
            # default is what lets the prod pair differ by the `cls:` block and nothing else.
            if return_cls and "map_feature_dim" not in net_conf["network"]:
                raise ValueError(
                    "network.map_feature_dim is required alongside a `cls:` block — it is the map "
                    "branch's width AT the concatenation, and silently defaulting it to None would "
                    "change the balance between the two branches. Set it explicitly (e.g. 512, "
                    "matching the Cls embedding width)."
                )
            map_feature_dim = net_conf["network"].get("map_feature_dim", None)

            # `masked_attention: true` excludes footprint-masked pixels from the transformer's
            # attention/merges/pooling (static mask constants, no checkpoint variables).
            masked_attention = bool(net_conf["network"].get("masked_attention", False))

            # The Cls block decides the only structural difference between the two paths: with it
            # the network concatenates the Cls branch onto the map features, without it there is
            # nothing to fuse. One constructor, one code path; the encoder and the head are built
            # identically either way, and map_feature_dim sizes the map branch for the concat.
            cls_kwargs = get_cls_branch_kwargs(cls_conf, msfm_conf, dlss_conf, n_side, cls_transform)
            # dense regression head minus the leading readout (the map feature is already 2-D)
            regression_head_layers = get_regression_head(
                out_features=n_output,
                head_type="dense",
                dense_layers=fused_head_layers,
                dropout_rate=head_dropout,
            )[1:]
            network = TransformerSummaryNetwork(
                smoothing_kwargs=smoothing_kwargs,
                smooth_indices=smooth_indices,
                nside=smooth_nside,
                token_nside=token_nside,
                in_channels=n_z_bins,
                map_feature_dim=map_feature_dim,
                transformer_kwargs=transformer_kwargs,
                regression_head_layers=regression_head_layers,
                **cls_kwargs,
                jit_compile_body=jit_compile_body,
                input_norm=input_norm,
                masked_attention=masked_attention,
                spmm_backend=spmm_backend,
            )

            # trace so network.built=True before BaseModel.summary(). Under
            # MultiWorkerMirroredStrategy the eager call runs in the /job:localhost context
            # while the in-scope variables live on /job:worker/.../GPU:0, which the
            # jit-compiled body cannot bridge (XLA cross-device resource access) — route the
            # trace through strategy.run there.
            def _build_trace():
                maps_in = tf.zeros((2, len(smooth_indices), n_z_bins))
                if not return_cls:
                    return network(maps_in, training=False)
                cls_in = tf.zeros((2, 3 * n_side, len(cls_kwargs["l_min_per_pair"])))
                return network((maps_in, cls_in), training=False)

            if isinstance(strategy, tf.distribute.MultiWorkerMirroredStrategy):
                strategy.run(_build_trace)
            else:
                _build_trace()

            LOGGER.info(f"Built transformer network {net_conf['network']['name']} (return_cls={return_cls})")
            model = Model(
                network=network,
                n_side=None,
                indices=None,
                n_neighbors=None,
                z_bank_size=net_conf["network"]["z_bank_size"],
                max_checkpoints=net_conf["network"]["max_checkpoints"],
                optimizer=optimizer,
                input_shape=None,
                max_batch_size=effective_local_batch_size,
                checkpoint_dir=checkpoint_dir,
                summary_dir=summary_dir,
                restore_checkpoint=args.restore_checkpoint,
                strategy=strategy,
                xla=args.xla,
                summary_every=args.summary_every,
            )

        elif net_name == "resnet":
            net_spec = NETWORKS[net_conf["network"]["name"]](
                out_features=n_output,
                # multi-res: smoothing and input norm move into ResNetMultiResEncoder, which
                # consumes the split_probes spec; the spec then only provides the layer lists
                smoothing_kwargs=None if is_multires_gcnn else smoothing_kwargs,
                # only passed when enabled: input_norm is a ResNetLayers feature, and the other
                # NETWORKS specs keep their signatures (they fail loudly if it is requested)
                **({"input_norm": True} if input_norm and not is_multires_gcnn else {}),
                **({"smoothing_external": True} if is_multires_gcnn else {}),
                spmm_backend=spmm_backend,
                **net_conf["network"]["kwargs"],
            )
            norm_owner = net_spec
            LOGGER.info(f"Loaded a network specification of type {NETWORKS[net_conf['network']['name']]}")
            LOGGER.info(f"Network kwargs including regularization: {net_conf['network']['kwargs']}")
            # Build a ResNetSummaryNetwork: the HealpyGCNN map branch, with the binned log-Cls
            # branch concatenated onto it when -- and only when -- the config carries a cls: block.
            # That concatenation is the ONLY difference between the two paths; the map branch, the
            # readout and the head are built identically either way. The network is passed to
            # BaseModel pre-built, so it is used directly without re-wrapping in a HealpyGCNN.
            cls_kwargs = get_cls_branch_kwargs(cls_conf, msfm_conf, dlss_conf, n_side, cls_transform)
            map_encoder = None
            if is_multires_gcnn:
                map_encoder = ResNetMultiResEncoder(
                    smoothing_kwargs=smoothing_kwargs,
                    layers=net_spec.get_conv_layers(),
                    nside=smooth_nside,
                    n_neighbors=net_conf["network"]["n_neighbors"],
                    max_batch_size=effective_local_batch_size,
                    input_norm=input_norm,
                    spmm_backend=spmm_backend,
                    fusion=net_conf["network"].get("fusion", "concat"),
                    # top-level network key (NOT under kwargs, which is splatted into ResNetLayers);
                    # the conv build params are read from kwargs to match the body's conv basis/degree
                    injection_conv_layers=net_conf["network"].get("injection_conv_layers", 0),
                    injection_conv_kwargs={
                        "poly_degree": net_conf["network"]["kwargs"].get("poly_degree", 5),
                        "conv_type": net_conf["network"]["kwargs"].get("conv_type", "cheby"),
                    },
                    # also a top-level network key, for the same reason as injection_conv_layers
                    fusion_width=net_conf["network"].get("fusion_width", None),
                    fuse_act=net_conf["network"].get("fuse_act", None),
                )
                norm_owner = map_encoder
            network = ResNetSummaryNetwork(
                conv_layers=None if is_multires_gcnn else net_spec.get_conv_layers(),
                regression_head_layers=net_spec.get_head_layers_no_flatten(),
                n_side=None if is_multires_gcnn else smooth_nside,
                indices=None if is_multires_gcnn else smooth_indices,
                n_neighbors=net_conf["network"]["n_neighbors"],
                max_batch_size=effective_local_batch_size,
                initial_Fin=None if is_multires_gcnn else n_z_bins,
                # the Cls branch, or {} for a maps-only network -- the one difference between the paths
                **cls_kwargs,
                # optional map-branch bottleneck, mirroring the transformer; None = raw flattened
                # GCNN features (old checkpoint lineage)
                map_feature_dim=net_conf["network"].get("map_feature_dim", None),
                map_encoder=map_encoder,
                # map-branch readout: None=flatten (legacy), "mean"=mean-pool over pixels,
                # "mean_std"/"moments" add higher moments over the same pixel axis
                map_pool=net_conf["network"].get("map_pool", None),
                # apply that readout at every resolution of the map branch, not just the trunk
                map_pool_multiscale=net_conf["network"].get("map_pool_multiscale", False),
                # single-res network builds its own HealpyGCNN; multi-res owns it in map_encoder
                # (which already carries spmm_backend), so this is inert there
                spmm_backend=spmm_backend,
            )
            # HealpySmoothing is a tf.keras.Model whose build() must be called before
            # setup_grid_loss_step accesses trainable_variables. BaseModel skips this
            # because input_shape=None is passed below (a pre-built subclassed Model can't use
            # the standard build path). Build the inner GCNN directly with the map shape.
            # (multi-res: the encoder is a subclassed Model built by the trace below instead)
            if network.gcnn is not None:
                network.gcnn.build((effective_local_batch_size, len(smooth_indices), n_z_bins))
            # Trace the full ResNetSummaryNetwork so that network.built=True and BaseModel
            # can call network.summary(). gcnn.build() only builds the map branch.
            maps_trace = tf.zeros((2, len(smooth_indices), n_z_bins))
            if return_cls:
                cls_trace = tf.zeros((2, 3 * n_side, len(cls_kwargs["l_min_per_pair"])))
                network((maps_trace, cls_trace), training=False)
            else:
                network(maps_trace, training=False)
            model = Model(
                network=network,
                n_side=None,
                indices=None,
                n_neighbors=net_conf["network"]["n_neighbors"],
                z_bank_size=net_conf["network"]["z_bank_size"],
                max_checkpoints=net_conf["network"]["max_checkpoints"],
                optimizer=optimizer,
                input_shape=None,
                max_batch_size=effective_local_batch_size,
                checkpoint_dir=checkpoint_dir,
                summary_dir=summary_dir,
                restore_checkpoint=args.restore_checkpoint,
                strategy=strategy,
                xla=args.xla,
                summary_every=args.summary_every,
            )
        else:
            # Legacy layer-list specs: vision_transformer, graph_transformer, one_d_conv. They
            # expose neither get_conv_layers() nor get_head_layers_no_flatten(), so they cannot be
            # wrapped by ResNetSummaryNetwork and keep BaseModel's HealpyGCNN-wrapping path, in
            # which the regression head (readout included) is the tail of the layer list. That also
            # means they have no fusion point and therefore no Cls branch. Don't reach for these
            # for new work.
            if return_cls:
                raise ValueError(
                    f"network.name={net_name!r} is a legacy layer-list spec and has no Cls branch; "
                    "remove the cls: block, or use name=resnet / nested_transformer."
                )
            net_spec = NETWORKS[net_name](
                out_features=n_output,
                smoothing_kwargs=smoothing_kwargs,
                # only passed when enabled: the legacy specs keep their signatures and fail loudly
                # if input_norm is requested
                **({"input_norm": True} if input_norm else {}),
                spmm_backend=spmm_backend,
                **net_conf["network"]["kwargs"],
            )
            norm_owner = net_spec
            LOGGER.info(f"Loaded a legacy network specification of type {NETWORKS[net_name]}")
            LOGGER.info(f"Network kwargs including regularization: {net_conf['network']['kwargs']}")
            model = Model(
                network=net_spec.get_layers(),
                n_side=smooth_nside,
                indices=smooth_indices,
                n_neighbors=net_conf["network"]["n_neighbors"],
                # BaseModel wraps the layer list in a HealpyGCNN built with this backend
                spmm_backend=spmm_backend,
                z_bank_size=net_conf["network"]["z_bank_size"],
                max_checkpoints=net_conf["network"]["max_checkpoints"],
                optimizer=optimizer,
                input_shape=(None, len(smooth_indices), n_z_bins),
                max_batch_size=effective_local_batch_size,
                checkpoint_dir=checkpoint_dir,
                summary_dir=summary_dir,
                restore_checkpoint=args.restore_checkpoint,
                strategy=strategy,
                xla=args.xla,
                summary_every=args.summary_every,
            )

        # Fit the per-feature asinh scale for the Cls branch and load it into the (checkpointed)
        # layer. Only for fresh maps+cls runs — on restore the scale returns with the checkpoint.
        if return_cls and cls_transform == "asinh_per_feature" and not args.restore_checkpoint:
            from deep_lss.utils import cls_preprocessing

            asinh_data_dir = args.data_dir
            if asinh_data_dir is None:
                asinh_data_dir = args.train_tfr_pattern.split("/tfrecords/")[0]
                LOGGER.warning(
                    f"--data_dir not set; deriving Cls-cache dir from --train_tfr_pattern: {asinh_data_dir}"
                )
            scales_name = os.path.splitext(os.path.basename(args.scales_config))[0]
            dset_common = dlss_conf["dset"]["common"]
            scale = cls_preprocessing.compute_asinh_scale_from_cache(
                data_dir=asinh_data_dir,
                msfm_conf=msfm_conf,
                cls_n_bins=cls_conf.get("n_bins", 16),
                scales_name=scales_name,
                with_lensing=dset_common["with_lensing"],
                with_clustering=dset_common["with_clustering"],
                with_cross_z=dset_common.get("with_cross_z", True),
                with_cross_probe=dset_common.get(
                    "with_cross_probe", dset_common["with_lensing"] and dset_common["with_clustering"]
                ),
                lenses_before_sources=dset_common.get("lenses_before_sources", dset_common.get("ggl_only", False)),
                default_scale=cls_conf.get("asinh_default_scale", None),
            )
            network.cls_layer.set_scale(scale)

        # Measure the empirical input-map normalization (post-smoothing per-channel mean/std)
        # from training batches and load it into the (checkpointed) layer. Only for fresh runs —
        # on restore the statistics return with the checkpoint. Under Horovod only rank 0
        # measures (its data shard differs from the other ranks'), then broadcasts, so all
        # replicas start from identical values. The transformer encoders (single- and
        # multi-resolution, on network.map_encoder), the GCNN spec (ResNetLayers, whose layer
        # objects are the same instances that live inside the built network), and
        # ResNetMultiResEncoder all expose the same smooth_groups/masks/load_input_norm_stats
        # interface, so one code path covers every architecture and resolution (one or more
        # input-norm groups). norm_owner is set in the branches above.
        if input_norm and not args.restore_checkpoint:
            if is_transformer:
                norm_owner = network.map_encoder
            input_norm_stats = None
            if not isinstance(strategy, HorovodStrategy) or hvd.rank() == 0:
                adapt_dset = build_dset(train_pipeline, args.train_tfr_pattern, dset_kwargs)
                # 32 batches x local_batch_size maps: per-pixel mean SE ~ std/sqrt(n_maps),
                # a fixed (sim=data) offset field; the per-channel std is exact at this n
                input_norm_stats = compute_input_norm_stats(
                    norm_owner.smooth_groups, adapt_dset, n_batches=32, masks=norm_owner.masks
                )
            if isinstance(strategy, HorovodStrategy):
                input_norm_stats = strategy.broadcast_object(input_norm_stats, root_rank=0)
            norm_owner.load_input_norm_stats(input_norm_stats)

        # training step, fiducial pipeline
        if args.loss_function == "delta":
            perts = parameters.get_fiducial_perturbations(params)
            LOGGER.info(f"Training with respect to the {n_params} parameters {params} with off sets {perts}")

            model.setup_delta_loss_step(
                n_params,
                local_batch_size,
                perts,
                dim_channels=n_z_bins,
                **loss_conf["delta_loss"],
                **net_conf["optimization"]["gradient_clipping"],
            )
        # grid pipeline
        else:
            if args.loss_function == "likelihood":
                if not args.restore_checkpoint:
                    lambda_tikhonov_schedule = tf.keras.optimizers.schedules.CosineDecay(
                        loss_conf["likelihood_loss"]["lambda_tikhonov_init"],
                        loss_conf["likelihood_loss"]["lambda_tikhonov_decay_steps"],
                        alpha=0.0,
                    )
                    lambda_tikhonov = tf.Variable(lambda_tikhonov_schedule(0), trainable=False, dtype=tf.float32)
                else:
                    lambda_tikhonov = tf.Variable(0.0, trainable=False, dtype=tf.float32)
                likelihood_kwargs = {
                    "lambda_tikhonov": lambda_tikhonov,
                    "img_summary": loss_conf["likelihood_loss"]["img_summary"],
                }
            else:
                likelihood_kwargs = {}

            if args.loss_function == "mutual_info":
                mi_conf = loss_conf["mutual_info_loss"]

                # standardize theta inside the variational head (head models z = (theta - mean) / std;
                # log_prob stays in physical units via the constant log-Jacobian). The MI-bound optimum
                # is affine-invariant, so ANY reasonable affine map fixes the head conditioning (~30x
                # scale spread between Om and n_Aia under-trains the tight directions; measured +30%
                # mock FoM on the Cls pipeline 2026-07-12). Unlike the Cls app there is no gathered
                # label table here, so use analytic stats of the uniform priors the grid is
                # Sobol-sampled from. Labels arrive in prior units: the raw bary_Mc tfrecord label is
                # converted to log10(Mc) at load time in build_dset.
                theta_shift, theta_scale = None, None
                if mi_conf.get("standardize_theta", True):
                    prior_intervals = parameters.get_prior_intervals(params, conf=msfm_conf)
                    theta_shift, theta_scale = training_helpers.theta_standardization_from_prior_intervals(
                        prior_intervals
                    )
                    LOGGER.info(f"VMIM head theta standardization: shift = {theta_shift}, scale = {theta_scale}")

                mutual_info_kwargs = {
                    "dim_summary": n_output,
                    **mi_conf["regu"],
                    "mutual_info_estimator": mi_conf["estimator"],
                    "mutual_info_kwargs": {
                        "density_estimator": mi_conf.get("density_estimator", "gmm"),
                        "theta_shift": theta_shift,
                        "theta_scale": theta_scale,
                        **mi_conf["kwargs"],
                    },
                }
            else:
                mutual_info_kwargs = {}

            # A static input_signature (dim_x set) only fits BaseModel's layer-list path, where it
            # wraps the layers in a HealpyGCNN itself. Every other network here is a pre-built
            # subclassed Model — it may take a (maps, cls) tuple, and its map width is
            # len(smooth_indices), which differs from len(data_vec_pix) under downsampling — so
            # only the legacy specs get a static signature; everything else traces dynamically.
            dynamic_input = is_transformer or net_name == "resnet"
            model.setup_grid_loss_step(
                loss=args.loss_function,
                batch_size=local_batch_size,
                dim_theta=n_params,
                dim_x=None if dynamic_input else len(data_vec_pix),
                dim_channels=None if dynamic_input else n_z_bins,
                **mutual_info_kwargs,
                **likelihood_kwargs,
                **net_conf["optimization"]["gradient_clipping"],
            )

    # validation loss
    if vali_every is not None:
        vali_pipe_kwargs = {k: v for k, v in dlss_conf["dset"]["common"].items() if k not in _CLS_ONLY_KEYS}
        vali_pipe_kwargs["return_maps"] = True
        vali_pipe_kwargs["return_cls"] = return_cls
        vali_dset_kwargs = {**net_conf["dset"]["validation"]["common"], **data_conf}
        vali_dset_kwargs["drop_remainder"] = True
        n_vali_batches = net_conf["dset"]["validation"]["n_batches"]

        # fall back to the training tfrecords when no explicit validation pattern is given;
        # the split is fully determined by signal_indices + is_eval in the validation config.
        grid_vali_tfr = args.grid_vali_tfr_pattern or (args.train_tfr_pattern if training_type == "grid" else None)

        def make_validation_loop(dist_dset, step_fn, n_expected, summary_map):
            def validation_loop():
                metrics = [tf.keras.metrics.Mean(), tf.keras.metrics.Mean()]
                n_batches = 0
                n_loss_nan = 0
                for batch_tuple in LOGGER.progressbar(
                    dist_dset, at_level="debug", desc="validation", total=n_expected
                ):
                    vals = step_fn(batch_tuple)
                    n_batches += 1
                    for i, v in enumerate(vals):
                        if not tf.math.is_nan(v):
                            metrics[i].update_state(v)
                        elif i == 0:
                            n_loss_nan += 1
                # A diverged (NaN) network makes every validation batch NaN; the per-batch skip
                # above then leaves the loss metric empty and tf.keras.metrics.Mean returns 0, so
                # the assert below silently passed and training kept running to completion. Catch
                # the total-collapse case explicitly while still tolerating the odd partial batch.
                if n_batches > 0 and n_loss_nan == n_batches:
                    raise RuntimeError(
                        f"Validation loss is NaN for all {n_batches} validation batches — the "
                        "network has diverged (NaN). Aborting instead of logging a spurious 0."
                    )
                assert not tf.math.is_nan(
                    metrics[0].result()
                ), "Validation loss is NaN, check the validation batch size as this is likely due to partially empty batches"
                for key, idx in summary_map:
                    model.write_summary(key, metrics[idx].result())
                for m in metrics:
                    m.reset_states()

            return validation_loop

        if args.fidu_vali_tfr_pattern is not None:
            vali_dset_kwargs.update(net_conf["dset"]["validation"]["fiducial"])

            if args.loss_function == "delta":
                # we need the perturbations
                vali_pipe_kwargs["params"] = dlss_conf["dset"]["training"]["params"]

                # to use the correct effective batch size with respect to the perturbations
                vali_dset_kwargs["local_batch_size"] = local_batch_size

                # this is equal to the cov_det_loss term in the delta loss
                def non_regularized_loss_fn(batch):
                    return delta_loss.delta_loss(
                        batch,
                        n_params=n_params,
                        n_same=local_batch_size,
                        off_sets=perts,
                        n_output=n_params,
                        force_params_value=None,
                        jac_weight=None,
                        jac_cond_weight=None,
                        tikhonov_regu=False,
                        training=False,
                        strategy=strategy,
                    )

                @tf.function
                def vali_loss_fn(batch):
                    preds = model(batch, training=False)
                    with tf.summary.record_if(False):
                        loss = model.vali_loss_fn(preds)
                        loss_non_regu = non_regularized_loss_fn(preds)
                    return loss, loss_non_regu

            else:
                # we don't need the perturbations
                vali_pipe_kwargs["params"] = []

                if args.loss_function == "likelihood" or args.loss_function == "mse":
                    # ignore the covariance term and rescaling
                    mse = tf.keras.metrics.MeanSquaredError()

                    # as this loss is supervised
                    labels = parameters.get_fiducials(params)

                    @tf.function
                    def vali_loss_fn(batch):
                        preds = model(batch, training=False)
                        with tf.summary.record_if(False):
                            loss = model.vali_loss_fn(preds, labels)
                        loss_non_regu = mse(tf.slice(preds, begin=[0, 0], size=[-1, n_params]), labels)
                        return loss, loss_non_regu

                elif args.loss_function == "mutual_info":
                    labels = tf.constant(parameters.get_fiducials(params, conf=msfm_conf), dtype=tf.float32)
                    labels = tf.reshape(labels, shape=[-1, n_params])

                    @tf.function
                    def vali_loss_fn(batch):
                        preds = model(batch, training=False)
                        with tf.summary.record_if(False):
                            loss = model.vali_loss_fn(preds, labels)
                        loss_non_regu = loss
                        return loss, loss_non_regu

            LOGGER.info("Fiducial validation set")
            vali_fidu_pipe = FiducialPipeline(conf=msfm_conf, **vali_pipe_kwargs)

            def vali_dset_fn(input_context):
                dset = build_dset(vali_fidu_pipe, args.fidu_vali_tfr_pattern, vali_dset_kwargs, input_context)
                if n_vali_batches is not None:
                    dset = dset.take(n_vali_batches * strategy.num_replicas_in_sync).cache()

                return dset

            dist_vali_dset = strategy.distribute_datasets_from_function(vali_dset_fn)

            def vali_step_fn(batch_tuple):
                vali_batch, _, _ = batch_tuple
                total, main = strategy.run(vali_loss_fn, args=(vali_batch,))
                return (
                    strategy.reduce(tf.distribute.ReduceOp.MEAN, total, axis=None),
                    strategy.reduce(tf.distribute.ReduceOp.MEAN, main, axis=None),
                )

            validation_loop = make_validation_loop(
                dist_vali_dset,
                vali_step_fn,
                n_vali_batches,
                [("loss/vali_total", 0), ("loss/vali_main", 1)],
            )

        elif grid_vali_tfr is not None:
            vali_pipe_kwargs["params"] = dlss_conf["dset"]["eval"]["grid"]["params"]

            vali_dset_kwargs.update(net_conf["dset"]["validation"]["grid"])

            LOGGER.info("Grid validation set")
            n_vali_examples_per_replica = (
                n_vali_batches * vali_dset_kwargs["local_batch_size"] if n_vali_batches is not None else None
            )
            LOGGER.info(
                f"Grid validation: {n_vali_batches} batches × local_batch_size "
                f"{vali_dset_kwargs['local_batch_size']} = "
                f"{n_vali_examples_per_replica} examples/replica, every {vali_every} steps"
            )
            vali_grid_pipe = GridPipeline(conf=msfm_conf, **vali_pipe_kwargs)

            def vali_dset_fn(input_context):
                dset = build_dset(vali_grid_pipe, grid_vali_tfr, vali_dset_kwargs, input_context)
                if n_vali_batches is not None:
                    dset = dset.take(n_vali_batches * strategy.num_replicas_in_sync).cache()

                return dset

            # Per-parameter normalization for the vali RMSE. An unweighted mean of squared errors in
            # PHYSICAL units is dominated by the widest priors: on the lensing target
            # (Om, s8, w0, Aia, n_Aia) the prior variances span [0.009 ... 8.3], so Aia+n_Aia carry
            # ~99% of the metric and Om ~0.1% -- it tracks IA nuisance recovery, not the cosmology the
            # FoM is built on. Normalize each parameter by its prior std (so 1.0 = no better than
            # predicting the prior mean) and average only the cosmological parameters. Computed
            # independently of standardize_theta, which rescales the head rather than the metric.
            _prior_iv = parameters.get_prior_intervals(params, conf=msfm_conf)
            vali_mse_scale = tf.constant(((_prior_iv[:, 1] - _prior_iv[:, 0]) / (12.0**0.5)).astype("float32"))
            _fom_idx = training_helpers.cosmo_param_indices(params, msfm_conf["analysis"]["params"]["cosmo"])
            LOGGER.info(f"vali nrmse_cosmo over {[params[i] for i in _fom_idx]} (of {params})")
            vali_fom_idx = tf.constant(_fom_idx, dtype=tf.int32)
            _has_fom_params = len(_fom_idx) > 0

            if args.loss_function == "mutual_info":

                @tf.function
                def vali_loss_fn(x, cosmo):
                    preds = model(x, training=False)
                    with tf.summary.record_if(False):
                        loss = model.vali_loss_fn(preds, cosmo)
                    if hasattr(model, "vali_posterior_mean_fn") and _has_fom_params:
                        posterior_mean = model.vali_posterior_mean_fn(preds)
                        err = (posterior_mean - tf.cast(cosmo, tf.float32)) / vali_mse_scale
                        rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.gather(err, vali_fom_idx, axis=-1))))
                    else:
                        rmse = tf.constant(float("nan"))
                    return loss, rmse

            else:
                raise NotImplementedError("Validation for the grid dataset is not implemented yet for other losses")

            dist_vali_dset = strategy.distribute_datasets_from_function(vali_dset_fn)

            def vali_step_fn(batch_tuple):
                dv_batch, cl_batch, cosmo_batch, index_batch = batch_tuple
                x_batch = (dv_batch, cl_batch) if return_cls else dv_batch
                loss, rmse = strategy.run(vali_loss_fn, args=(x_batch, cosmo_batch))
                return (
                    strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None),
                    strategy.reduce(tf.distribute.ReduceOp.MEAN, rmse, axis=None),
                )

            # vali_loss_fn has no z-regularization, so total == main; log both keys for
            # consistency with the fiducial validation path
            validation_loop = make_validation_loop(
                dist_vali_dset,
                vali_step_fn,
                n_vali_batches,
                [("loss/vali_total", 0), ("loss/vali_nrmse_cosmo", 1)],
            )

    # A restored run resumes from the checkpointed step, so n_steps is the TOTAL step budget across
    # chained jobs (submit the follow-up with RUN_NUM=2 and --dependency=afterany): the restored
    # optimizer.iterations keeps the LR schedule, whose span is n_steps, on its curve, and the loop
    # trains only the remainder. If the budget is already exhausted the loop is empty and the run
    # falls through to the final-checkpoint logic.
    start_step = int(model.get_step()) if args.restore_checkpoint else 0
    if start_step > 0:
        LOGGER.info(f"Resuming from step {start_step} towards the total budget of {n_steps}")
        if start_step >= n_steps:
            LOGGER.warning(f"Restored step {start_step} >= n_steps {n_steps}: nothing left to train")

    LOGGER.info("Starting training")
    LOGGER.timer.start("training")

    # measure the sustained rate of this job, so the next one can be sized from a real number rather
    # than from tqdm's cumulative figure (which includes compilation and understates it badly)
    tracker = throughput.ThroughputTracker(start_step=start_step, dir_model=dir_model)
    wall_clock_schedule = getattr(optimizer, "wall_clock_schedule", None)
    if budget is not None:
        budget.start()

    t_prev = time()
    t_accum = 0.0
    t_data_accum = 0.0
    t_compute_accum = 0.0
    nan_streak = 0
    step = start_step  # the post-loop final-checkpoint check reads `step` even if the loop is empty

    for step in LOGGER.progressbar(
        range(start_step + 1, n_steps + 1), at_level="info", total=n_steps - start_step, desc="training"
    ):
        # context for profiling like https://www.tensorflow.org/guide/profiler#profiling_custom_training_loops
        # optional context like https://stackoverflow.com/a/34798330
        with tf.profiler.experimental.Trace("step", step_num=step, _r=1) if args.profile else nullcontext():
            # wall-clock cosine: the learning rate is a variable this loop drives, because the point
            # at which the allocation runs out is not a step index known in advance
            if wall_clock_schedule is not None:
                wall_clock_schedule.update(step)

            # train step
            t_data_start = time()
            if args.loss_function == "delta":
                dv_batch, _, index_batch = next(dist_iter)
                t_data_end = time()
                loss = model.delta_train_step(dv_batch)
            else:
                dv_batch, cl_batch, cosmo_batch, index_batch = next(dist_iter)
                t_data_end = time()
                x_batch = (dv_batch, cl_batch) if return_cls else dv_batch
                if getattr(model, "grid_train_step_uses_pair_ids", False):
                    loss = model.grid_train_step(x_batch, cosmo_batch, index_batch[0], index_batch[1])
                else:
                    loss = model.grid_train_step(x_batch, cosmo_batch)
            t_compute_end = time()

            # fail-fast NaN watchdog (see nan_check_every / nan_abort_after above). The
            # grad-zeroing safety net makes a NaN step a no-op, so a diverged run would otherwise
            # freeze and burn its full allocation while saving a useless checkpoint.
            if nan_check_every and (step % nan_check_every == 0):
                if not bool(tf.math.is_finite(tf.cast(loss, tf.float32))):
                    nan_streak += 1
                    LOGGER.warning(
                        f"Training loss is non-finite at step {step} "
                        f"({nan_streak}/{nan_abort_after} consecutive check(s))"
                    )
                    if nan_streak >= nan_abort_after:
                        raise RuntimeError(
                            f"Training loss non-finite for {nan_streak} consecutive check(s) "
                            f"(step {step}); aborting. Continuing would only waste the allocation "
                            "on a frozen network. Check numerical stability (precision / depth)."
                        )
                else:
                    nan_streak = 0

            # horovod
            if isinstance(model.strategy, HorovodStrategy) and step == start_step + 1:
                LOGGER.info("First step, broadcasting the variables through Horovod")
                model.horovod_broadcast_variables()

            # delta loss
            if args.loss_function == "delta" and not args.restore_checkpoint and noise_schedule_steps is not None:
                # assignment has to happen outside the tf.function
                noise_scale.assign(noise_scheduler(step))
                model.write_summary("schedule/noise_scale", noise_scale)

            # likelihood loss
            if args.loss_function == "likelihood" and not args.restore_checkpoint:
                lambda_tikhonov.assign(lambda_tikhonov_schedule(step))
                model.write_summary("schedule/lambda_tikhonov", lambda_tikhonov)

            # output
            if (output_every is not None) and (step % output_every == 0):
                _copy_log(args, dir_model)

            # checkpoint
            if (checkpoint_every is not None) and (step % checkpoint_every == 0):
                model.save_model()
                # persist the consumed budget alongside it, so a chained job resumes the same cosine
                if budget is not None:
                    throughput.write_budget_state(dir_model, budget)

            # validate
            if (vali_every is not None) and (step % vali_every == 0):
                # since at that step, everything should be already traced
                second_vali = step == 2 * vali_every
                if second_vali:
                    LOGGER.info(f"Validating the model every {vali_every} steps")
                    LOGGER.timer.start("vali")

                validation_loop()
                if model.summary_writer is not None:
                    model.summary_writer.flush()

                if second_vali:
                    LOGGER.info(
                        f"Finished validating the model after {LOGGER.timer.elapsed('vali')} and "
                        f"{n_vali_batches} steps/batches"
                    )

            # evaluate
            if (eval_every is not None) and (step % eval_every == 0):
                train_step = model.get_step()
                LOGGER.info(f"Evaluating the model after a total of {train_step} training steps")

                out_file = None

                # fiducial training
                if args.evaluate_training_set:
                    if training_type == "fiducial":
                        out_file = evaluation.evaluate_fiducial(
                            model=model,
                            tfr_pattern=args.train_tfr_pattern,
                            msfm_conf=msfm_conf,
                            dlss_conf=dlss_conf,
                            net_conf=net_conf,
                            data_conf=data_conf,
                            dir_out=dir_model,
                            file_label=train_step,
                            training_set=True,
                        )
                    elif training_type == "grid":
                        out_file = evaluation.evaluate_grid(
                            model=model,
                            tfr_pattern=args.train_tfr_pattern,
                            msfm_conf=msfm_conf,
                            dlss_conf=dlss_conf,
                            net_conf=net_conf,
                            data_conf=data_conf,
                            dir_out=dir_model,
                            file_label=train_step,
                        )
                else:
                    LOGGER.warning("Skipping evaluation of the fiducial training set")

                # fiducial evaluation
                if args.fidu_eval_tfr_pattern is not None:
                    out_file = evaluation.evaluate_fiducial(
                        model=model,
                        tfr_pattern=args.fidu_eval_tfr_pattern,
                        msfm_conf=msfm_conf,
                        dlss_conf=dlss_conf,
                        net_conf=net_conf,
                        data_conf=data_conf,
                        dir_out=dir_model,
                        file_label=train_step,
                        training_set=False,
                    )
                else:
                    LOGGER.warning("Skipping evaluation of the fiducial evaluation set")

                # grid evaluation
                if args.grid_eval_tfr_pattern is not None:
                    out_file = evaluation.evaluate_grid(
                        model=model,
                        tfr_pattern=args.grid_eval_tfr_pattern,
                        msfm_conf=msfm_conf,
                        dlss_conf=dlss_conf,
                        net_conf=net_conf,
                        data_conf=data_conf,
                        dir_out=dir_model,
                        file_label=train_step,
                    )
                else:
                    LOGGER.warning("Skipping evaluation of the grid evaluation set")

                # log here instead of inside eval to avoid partial duplicate .h5 files
                if args.wandb and (out_file is not None):
                    wandb_artifact = wandb.Artifact(
                        name=f"training-predictions-nsteps{train_step}", type="predictions"
                    )
                    wandb_artifact.add_file(local_path=out_file)
                    wandb_run.log_artifact(wandb_artifact)
                    LOGGER.info(f"Logged the predictions to weights & biases after step {step}")

            # profile
            if args.profile and step == 800:
                print("\n")
                LOGGER.info("Starting to profile")
                tf.profiler.experimental.start(model.summary_dir)
            if args.profile and step == 805:
                print("\n")
                LOGGER.info("Stopping to profile")
                tf.profiler.experimental.stop()

            if args.pasc_throughput:
                step_start = 200
                step_delta = 1000

                if step == step_start:
                    LOGGER.info("Starting to measure throughput")
                    LOGGER.timer.start("pasc_throughput")
                    t_pasc = time()

                if step == step_start + step_delta:
                    LOGGER.info(f"{step_delta} steps took {LOGGER.timer.elapsed('pasc_throughput')}")
                    delta_t_pasc = time() - t_pasc
                    global_batch_size = local_batch_size * strategy.num_replicas_in_sync
                    examples_per_s = step_delta * global_batch_size / delta_t_pasc
                    LOGGER.info(f"throughput: {examples_per_s:.2f} examples/s")

            # additional logs
            t_now = time()
            t_accum += t_now - t_prev
            t_data_accum += t_data_end - t_data_start
            t_compute_accum += t_compute_end - t_data_end
            t_prev = t_now
            tracker.update(step)
            if step % args.summary_every == 0:
                model.write_summary("step_time", t_accum / args.summary_every)
                model.write_summary("data_time", t_data_accum / args.summary_every)
                model.write_summary("compute_time", t_compute_accum / args.summary_every)
                model.write_summary("global_step", model.get_step())
                if wall_clock_schedule is not None:
                    model.write_summary("schedule/budget_fraction", budget.fraction)
                t_accum = 0.0
                t_data_accum = 0.0
                t_compute_accum = 0.0

        # wall-clock budget: stop on time rather than on a step count. Checked outside the profiler
        # context so the final step is complete, and only every `summary_every` steps because
        # `budget.elapsed` is a host-side clock read.
        if budget is not None and step % args.summary_every == 0:
            if budget.exhausted:
                LOGGER.info(
                    f"Wall-clock budget of {budget.total_seconds:.0f} s is spent at step {step}; "
                    "the cosine has annealed to its floor and training is complete"
                )
                break
            if budget.job_exhausted:
                LOGGER.info(
                    f"This job's share of {budget.job_seconds:.0f} s is spent at step {step} with "
                    f"{budget.total_seconds - budget.elapsed:.0f} s of the run's budget left; "
                    "checkpointing for the next job in the chain"
                )
                break

    if budget is not None:
        # A budget-driven run that stops because it hit `n_steps` never finished annealing, which is
        # the failure this whole mechanism exists to prevent — say so loudly rather than silently
        # producing a half-decayed model.
        if step >= n_steps and not budget.exhausted and not budget.job_exhausted:
            LOGGER.warning(
                f"Training stopped at the n_steps cap of {n_steps} with {budget.total_seconds - budget.elapsed:.0f} s "
                f"of budget unspent, so the cosine only reached {budget.fraction:.0%} of its decay. Raise n_steps: it "
                "is meant to be a loose safety cap, not the length of the run."
            )
        # the final-checkpoint logic below saves the model; only the budget needs persisting here
        throughput.write_budget_state(dir_model, budget)

    LOGGER.info(
        f"Finished training after {step - start_step} steps (total {step}) and {LOGGER.timer.elapsed('training')}"
    )
    tracker.write()

    # finalize EMA weight averaging, if enabled
    inner_optimizer = getattr(optimizer, "inner_optimizer", optimizer)
    ema_finalized = getattr(inner_optimizer, "use_ema", False)
    if ema_finalized:
        LOGGER.info("Finalizing EMA weights")
        inner_optimizer.finalize_variable_values(model.trainable_variables)

    # save everything at the end if necessary
    if ema_finalized or ((checkpoint_every is not None) and (step % checkpoint_every != 0)):
        LOGGER.info("Creating a final checkpoint")
        model.save_model()
    elif checkpoint_every is not None:
        LOGGER.info("A final checkpoint already exists")
    else:
        LOGGER.info("No checkpoint has been saved")

    if args.wandb:
        wandb.finish()
    model.delete_temp_summaries()

    LOGGER.info("Script completed successfully")
    _copy_log(args, dir_model)


def _copy_log(args, dir_out):
    if args.slurm_output is not None:
        dir_log = os.path.join(dir_out, "logs")
        os.makedirs(dir_log, exist_ok=True)

        file_log = os.path.join(dir_log, os.path.basename(args.slurm_output))
        shutil.copy(args.slurm_output, file_log)


if __name__ == "__main__":
    args = setup()

    if args.wandb_sweep_id is None:
        training(args)
    else:
        if args.dist_strategy == "horovod":
            # it doesn't hurt to initialize horovod more than once
            hvd.init()

            # only the chief gets an agent, similar to
            # https://github.com/NERSC/nersc-dl-wandb/blob/958d1c7710719b0f91ff3236a77b551d6566b952/utils/trainer.py#L91C2-L91C2
            # and https://github.com/NERSC/nersc-dl-wandb/blob/958d1c7710719b0f91ff3236a77b551d6566b952/train.py#L24
            if hvd.rank() == 0:
                wandb.agent(args.wandb_sweep_id, function=training, project="y3-deep-lss", count=1)
            # the workers get the agent's hyperparameters via broadcast
            else:
                training()
        else:
            wandb.agent(args.wandb_sweep_id, function=training, project="y3-deep-lss", count=1)
