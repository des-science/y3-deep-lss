# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Train the DeepSphere graph neural networks at the fiducial cosmology and its perturbations using the information
maximizing loss to find an informative summary statistic.

Meant for the GPU nodes of the Perlmutter cluster at NERSC.
"""

import tensorflow as tf
import horovod.tensorflow as hvd
import os, argparse, warnings, yaml, wandb, shutil

from datetime import datetime
from time import time
from contextlib import nullcontext

from msfm.fiducial_pipeline import FiducialPipeline
from msfm.grid_pipeline import GridPipeline
from msfm.utils import logger, input_output, files, parameters

from deep_lss.utils import distribute, eval, configuration
from deep_lss.models.delta_model import DeltaLossModel
from deep_lss.models.grid_model import GridLossModel
from deep_lss.utils.distribute import HorovodStrategy
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
        "--loss_function",
        type=str,
        default="delta",
        choices=["delta", "mse", "likelihood", "mutual_info"],
        help="loss function to train the network with",
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
    parser.add_argument(
        "--dlss_config",
        type=str,
        default="config/dlss_config.yaml",
        help=(
            "configuration .yaml file of this repo. None means that the standard configuration file in"
            " configs/dlss_config.yaml relative to this repo is loaded."
        ),
    )
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
    parser.add_argument("--debug", action="store_true", help="activate debug mode")
    parser.add_argument("--profile", action="store_true", help="run the profiler")
    parser.add_argument("--slurm_output", type=str, default=None, help="path to the slurm output file")
    parser.add_argument("--wandb", action="store_true", help="log to weights & biases, otherwise log to tensorboard")
    parser.add_argument("--wandb_tags", nargs="+", type=str, default=None, help="tags for weights & biases")
    parser.add_argument("--wandb_notes", type=str, default=None, help="notes for weights & biases (longer than tags)")
    parser.add_argument("--wandb_sweep_id", type=str, default=None, help="id of the sweep. If None, no sweep is used")

    args, _ = parser.parse_known_args()

    if args.loss_function == "delta":
        assert "fiducial" in args.train_tfr_pattern, f"The delta loss can only be used for the fiducial dataset"
    else:
        assert "grid" in args.train_tfr_pattern, f"The {args.loss_function} loss can only be used for the grid dataset"

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

    if args.debug:
        tf.config.run_functions_eagerly(True)
        # tf.config.set_soft_device_placement(False)
        # tf.debugging.set_log_device_placement(True)
        # tf.data.experimental.enable_debug_mode()
        LOGGER.warning(f"!!!!! Running the training in test mode, TensorFlow is executed eagerly !!!!!")

    return args


def training():
    LOGGER.timer.start("main")

    args = setup()

    # hardware and distribution
    _, _ = distribute.check_devices()
    strategy = distribute.get_strategy(args.dist_strategy)

    # initialize a fresh model
    if not args.restore_checkpoint:
        # load the configs
        net_conf = input_output.read_yaml(os.path.join(args.repo_dir, args.net_config))
        dlss_conf = input_output.read_yaml(os.path.join(args.repo_dir, args.dlss_config))
        msfm_conf = files.load_config(args.msfm_config)
        LOGGER.info(f"Loaded configs from the provided paths")

        if args.dir_model is None:
            net_name = net_conf["name"]
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            args.dir_model = f"{now}_{net_name}"
            LOGGER.info(f"Created model directory {args.dir_model}")

        # make output directory
        dir_out = os.path.join(args.dir_base, args.dir_model)
        os.makedirs(dir_out, exist_ok=True)
        LOGGER.info(f"Created output directory {dir_out}")

        # additions to the configs
        net_conf["run"] = {}
        net_conf["run"]["dir_model"] = dir_out
        net_conf["run"]["dir_log"] = args.slurm_output
        net_conf["run"]["loss_func"] = args.loss_function
        net_conf["run"]["dist_strategy"] = args.dist_strategy

        # save the configs
        with open(os.path.join(dir_out, "configs.yaml"), "w") as f:
            yaml.dump_all([net_conf, dlss_conf, msfm_conf], f)

    # restore a saved model
    elif args.restore_checkpoint and (args.dir_model is not None):
        # make output directory
        dir_out = os.path.join(args.dir_base, args.dir_model)
        os.makedirs(dir_out, exist_ok=True)
        LOGGER.info(f"Created output directory {dir_out}")

        # load the configs
        with open(os.path.join(dir_out, "configs.yaml"), "r") as f:
            net_conf, dlss_conf, msfm_conf = list(yaml.load_all(f, Loader=yaml.FullLoader))

        LOGGER.info(f"Loaded configs from the model directory")

    else:
        raise ValueError(f"Can't restore the model from an unspecified dir_model")

    # weights and biases
    if args.wandb:
        group_name = distribute.get_wandb_group_name(strategy)

        wandb_run = wandb.init(
            project="y3-deep-lss",
            dir=dir_out,
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

        if args.wandb_sweep_id is not None:
            if isinstance(strategy, HorovodStrategy):
                # only the chief gets an agent, which provides the hyperparameters
                if hvd.rank() == 0:
                    nested_hyperparam_conf = configuration.convert_dotted_to_nested_dict(wandb_run.config)
                    net_conf = configuration.update_nested_dict(net_conf, nested_hyperparam_conf["net"])

                net_conf = strategy.broadcast_object(net_conf, root_rank=0)
                LOGGER.info(f"Broadcast the chief/agent's hyperparameters to the other ranks")

            else:
                # in the wandb sweep config, the hyperparameters are defined like net.optimization.optimizer, while the
                # .yaml config files are structured as nested dictionaries
                nested_hyperparam_conf = configuration.convert_dotted_to_nested_dict(wandb_run.config)

                # dict.update() would discard branches that are not present in the update dict
                net_conf = configuration.update_nested_dict(net_conf, nested_hyperparam_conf["net"])

        # only update the config here instead of in the init so that possible changes by a sweep agent are included
        wandb_run.config.setdefaults({"msfm": msfm_conf, "dlss": dlss_conf, "net": net_conf})

        LOGGER.info(f"Initialized weights & biases to {dir_out}")

    # set up subdirectories
    checkpoint_dir = os.path.abspath(os.path.join(dir_out, "checkpoint"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    summary_dir = os.path.abspath(os.path.join(dir_out, "summary"))
    os.makedirs(summary_dir, exist_ok=True)

    # constants: msfm
    n_side = msfm_conf["analysis"]["n_side"]
    data_vec_pix, _, _, _ = files.load_pixel_file(msfm_conf)

    # constants: deep_lss
    params = dlss_conf["dset"]["training"]["params"]
    n_params = len(params)
    perts = parameters.get_fiducial_perturbations(params)
    LOGGER.info(f"Training with respect to the {n_params} parameters {params} with off sets {perts}")

    with_lensing = dlss_conf["dset"]["common"]["with_lensing"]
    with_clustering = dlss_conf["dset"]["common"]["with_clustering"]

    # constants: network
    n_steps = net_conf["training"]["n_steps"]
    output_every = net_conf["training"]["output_every"]
    checkpoint_every = net_conf["training"]["checkpoint_every"]
    eval_every = net_conf["training"]["eval_every"]

    # constants: miscellaneous
    smoothing_kwargs = configuration.get_smoothing_kwargs(
        args.loss_function, msfm_conf, dlss_conf, net_conf, dir_base=args.dir_base
    )

    dset_kwargs = net_conf["dset"]["training"]["common"]
    if args.loss_function == "delta":
        Pipeline = FiducialPipeline
        Model = DeltaLossModel
        n_output = n_params
        dset_kwargs.update(net_conf["dset"]["training"]["delta_loss"])
        local_batch_size = dset_kwargs["local_batch_size"]
        effective_local_batch_size = local_batch_size * (2 * n_params + 1)
    else:
        if args.loss_function == "likelihood":
            n_output = n_params + n_params * (n_params + 1) // 2
        else:
            n_output = n_params
        Pipeline = GridPipeline
        Model = GridLossModel
        dset_kwargs.update(net_conf["dset"]["training"]["likelihood_loss"])
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

    # dataset
    train_pipeline = Pipeline(conf=msfm_conf, **{**dlss_conf["dset"]["common"], **dlss_conf["dset"]["training"]})

    # like https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
    def dataset_fn(input_context):
        dset = train_pipeline.get_dset(
            tfr_pattern=args.train_tfr_pattern,
            **dset_kwargs,
            # distribution
            input_context=input_context,
        )

        return dset

    dist_dset = strategy.distribute_datasets_from_function(dataset_fn)
    dist_iter = iter(dist_dset)

    # network, create all of the variables within the strategy's scope, such that they are mirrored
    with strategy.scope():
        network = NETWORKS[net_conf["network"]["name"]](
            output_shape=n_output, smoothing_kwargs=smoothing_kwargs, **net_conf["network"]["kwargs"]
        ).get_layers()
        LOGGER.info(f"Loaded a network specification of type {NETWORKS[net_conf['network']['name']]}")

        model = Model(
            network=network,
            n_side=n_side,
            indices=data_vec_pix,
            n_neighbors=net_conf["network"]["n_neighbors"],
            max_checkpoints=net_conf["network"]["max_checkpoints"],
            optimizer=net_conf["optimization"]["optimizer"],
            optimizer_kwargs=net_conf["optimization"]["optimizer_kwargs"],
            input_shape=(None, len(data_vec_pix), n_z_bins),
            max_batch_size=effective_local_batch_size,
            checkpoint_dir=checkpoint_dir,
            summary_dir=summary_dir,
            restore_checkpoint=args.restore_checkpoint,
            strategy=strategy,
        )

    # training step
    if args.loss_function == "delta":
        model.setup_delta_loss_step(
            n_params,
            local_batch_size,
            perts,
            n_channels=n_z_bins,
            **dlss_conf["delta_loss"],
            **net_conf["optimization"]["gradient_clipping"],
        )
    else:
        model.setup_grid_loss_step(
            loss=args.loss_function,
            batch_size=local_batch_size,
            n_channels=n_z_bins,
            n_params=n_params,
            **net_conf["optimization"]["gradient_clipping"],
        )

    LOGGER.info(f"Starting training")
    LOGGER.timer.start("training")
    t_prev = time()

    for step in LOGGER.progressbar(range(1, n_steps + 1), at_level="info", total=n_steps, desc="training at fiducial"):
        # context for profiling like https://www.tensorflow.org/guide/profiler#profiling_custom_training_loops
        # optional context like https://stackoverflow.com/a/34798330
        with tf.profiler.experimental.Trace("step", step_num=step, _r=1) if args.profile else nullcontext():
            # train step
            if args.loss_function == "delta":
                dv_batch, _ = next(dist_iter)
                model.delta_train_step(dv_batch)
            else:
                dv_batch, cosmo_batch, _ = next(dist_iter)
                model.grid_train_step(dv_batch, cosmo_batch)

            # horovod
            if isinstance(model.strategy, HorovodStrategy) and step == 1:
                LOGGER.info(f"First step, broadcasting the variables through Horovod")
                model.horovod_broadcast_variables()

            # output
            if (output_every is not None) and (step % output_every == 0):
                _copy_log(args, dir_out)

            # checkpoint
            if (checkpoint_every is not None) and (step % checkpoint_every == 0):
                model.save_model()

            # evaluate
            if (eval_every is not None) and (step % eval_every == 0):
                train_step = model.get_step()
                LOGGER.info(f"Evaluating the model after a total of {train_step} training steps")

                out_file = None

                # fiducial training
                if args.evaluate_training_set:
                    out_file = eval.evaluate_fiducial(
                        model=model,
                        tfr_pattern=args.train_tfr_pattern,
                        msfm_conf=msfm_conf,
                        dlss_conf=dlss_conf,
                        net_conf=net_conf,
                        dir_out=dir_out,
                        file_label=train_step,
                        training_set=True,
                    )
                else:
                    LOGGER.warning(f"Skipping evaluation of the fiducial training set")

                # fiducial validation
                if args.fidu_vali_tfr_pattern is not None:
                    out_file = eval.evaluate_fiducial(
                        model=model,
                        tfr_pattern=args.fidu_vali_tfr_pattern,
                        msfm_conf=msfm_conf,
                        dlss_conf=dlss_conf,
                        net_conf=net_conf,
                        dir_out=dir_out,
                        file_label=train_step,
                        training_set=False,
                    )
                else:
                    LOGGER.warning(f"Skipping evaluation of the fiducial validation set")

                # grid validation
                if args.grid_vali_tfr_pattern is not None:
                    out_file = eval.evaluate_grid(
                        model=model,
                        tfr_pattern=args.grid_vali_tfr_pattern,
                        msfm_conf=msfm_conf,
                        dlss_conf=dlss_conf,
                        net_conf=net_conf,
                        dir_out=dir_out,
                        file_label=train_step,
                    )
                else:
                    LOGGER.warning(f"Skipping evaluation of the grid validation set")

                # log here instead of inside eval to avoid partial duplicate .h5 files
                if args.wandb and (out_file is not None):
                    LOGGER.info(f"Logged the predictions to weights & biases after step {step}")
                    wandb_artifact = wandb.Artifact(
                        name=f"training-predictions-nsteps{train_step}", type="predictions"
                    )
                    wandb_artifact.add_file(local_path=out_file)
                    wandb_run.log_artifact(wandb_artifact)

            # profile
            if args.profile and step == 200:
                print("\n")
                LOGGER.info(f"Starting to profile")
                tf.profiler.experimental.start(model.summary_dir)
            if args.profile and step == 205:
                print("\n")
                LOGGER.info(f"Stopping to profile")
                tf.profiler.experimental.stop()

            # additional logs
            t_now = time()
            model.write_summary("step_time", t_now - t_prev)
            model.write_summary("global_step", step)
            # wandb.log({"global_step": step})
            t_prev = t_now

    LOGGER.info(f"Finished training after {n_steps} steps and {LOGGER.timer.elapsed('training')}")

    # save everything at the end if necessary
    if (checkpoint_every is not None) and (step % checkpoint_every != 0):
        LOGGER.info(f"Creating a final checkpoint")
        model.save_model()
    elif checkpoint_every is not None:
        LOGGER.info(f"A final checkpoint already exists")
    else:
        LOGGER.info(f"No checkpoint has been saved")

    if args.wandb:
        wandb.finish()
    model.delete_temp_summaries()

    LOGGER.info(f"Script completed successfully")
    _copy_log(args, dir_out)


def _copy_log(args, dir_out):
    if args.slurm_output is not None:
        dir_log = os.path.join(dir_out, "logs")
        os.makedirs(dir_log, exist_ok=True)

        file_log = os.path.join(dir_log, os.path.basename(args.slurm_output))
        shutil.copy(args.slurm_output, file_log)


if __name__ == "__main__":
    args = setup()

    if args.wandb_sweep_id is None:
        training()
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
