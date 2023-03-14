# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Train the DeepSphere graph neural networks at the fiducial cosmology and its perturbations using the information
maximizing loss to find an informative summary statistic.

Meant for the GPU nodes of the Perlmutter cluster at NERSC.
TODO implement weights & biases versioning
TODO make esub compatible? The index could correspond to a neural net architecture in the hyperparameter search
"""

import tensorflow as tf
import os, argparse, warnings, yaml

from datetime import datetime
from time import time
from contextlib import nullcontext

from msfm import fiducial_pipeline
from msfm.utils import logger, input_output, analysis, parameters

from deep_lss.utils import utils, distribute, eval
from deep_lss.models.delta_model import DeltaLossModel
from deep_lss.nets import NETWORKS

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)

# os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
# os.environ["TF_GPU_THREAD_COUNT"] = "16"


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
        "--fid_tfr_pattern",
        type=str,
        default="/pscratch/sd/a/athomsen/DESY3/v2/fiducial/DESy3_fiducial_???.tfrecord",
        help="input root dir of the fiducial data vectors",
    )
    parser.add_argument(
        "--grid_tfr_pattern",
        type=str,
        default="/pscratch/sd/a/athomsen/DESY3/v2/grid/DESy3_grid_???.tfrecord",
        help="input root dir of the grid data vectors",
    )
    # TODO
    # parser.add_argument("--with_bary", action="store_true", help="include baryons")
    parser.add_argument(
        "--dir_base",
        type=str,
        default=None,
        # TODO
        help="base dir where the models are saved. If None, a dir within the repo is generated according to the config",
    )
    parser.add_argument(
        "--dir_model",
        type=str,
        default=None,
        # TODO
        help="dir where the model summaries and checkpoints are saved. If None, a dir is generated according to the"
        " current date and time. This dir is appended to the dir_base as a relative path. Passing an absolute path"
        " overrides this.",
    )
    parser.add_argument(
        "--net_config",
        type=str,
        required=True,
        help="configuration .yaml file of the model to be trained",
    )
    parser.add_argument(
        "--dlss_config",
        type=str,
        default=None,
        help="configuration .yaml file of this repo",
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
        help="restore the model from a checkpoint instead of initializing it from scratch",
    )
    parser.add_argument("--local", action="store_true", help="distribute the training")
    parser.add_argument("--debug", action="store_true", help="activate debug mode")
    parser.add_argument("--profile", action="store_true", help="run the profiler")

    args, _ = parser.parse_known_args()

    # TODO create the model directory
    logger.set_all_loggers_level(args.verbosity)

    LOGGER.debug(f"--verbosity = {args.verbosity}")
    LOGGER.debug(f"--fid_tfr_pattern = {args.fid_tfr_pattern}")
    LOGGER.debug(f"--grid_tfr_pattern = {args.grid_tfr_pattern}")
    LOGGER.debug(f"--dir_base = {args.dir_base}")
    LOGGER.debug(f"--dir_model = {args.dir_model}")
    LOGGER.debug(f"--net_config = {args.net_config}")
    LOGGER.debug(f"--dlss_config = {args.dlss_config}")
    LOGGER.debug(f"--msfm_config = {args.msfm_config}")
    LOGGER.debug(f"--restore_checkpoint = {args.restore_checkpoint}")
    LOGGER.debug(f"--local = {args.local}")
    LOGGER.debug(f"--debug = {args.debug}")
    LOGGER.debug(f"--profile = {args.profile}")

    if args.debug:
        tf.config.run_functions_eagerly(True)
        # tf.config.set_soft_device_placement(False)
        # tf.debugging.set_log_device_placement(True)
        # tf.data.experimental.enable_debug_mode()
        LOGGER.warning(f"!!!!! Running the training in test mode, TensorFlow is executed eagerly !!!!!")

    return args


def training():
    # args = setup(args)
    args = setup()
    LOGGER.timer.start("main")

    _, _ = distribute.check_devices()

    # constants: deep-large-scale-structure
    dlss_conf = utils.load_deep_lss_config(args.dlss_config)
    target_params = dlss_conf["training"]["target_params"]
    n_params = len(target_params)
    perts = parameters.get_fiducial_perturbations(target_params)
    LOGGER.info(f"Training with respect to the parameters {target_params} with off sets {perts}")

    # constants: multiprobe-simulation-forward-model
    msfm_conf = analysis.load_config(args.msfm_config)
    data_vec_pix, _, _, _, _ = analysis.load_pixel_file(msfm_conf)
    n_side = msfm_conf["analysis"]["n_side"]

    # TODO could loop over esub indices here

    # constants: network
    net_conf = input_output.read_yaml(args.net_config)
    net_name = net_conf["name"]
    n_steps = net_conf["training"]["n_steps"]
    output_every = net_conf["training"]["output_every"]
    checkpoint_every = net_conf["training"]["checkpoint_every"]
    eval_every = net_conf["training"]["eval_every"]

    # create directories
    if args.dir_base is None:
        file_dir = os.path.dirname(__file__)
        repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
        dir_base = os.path.join(repo_dir, dlss_conf["dirs"]["base"])
        os.makedirs(dir_base, exist_ok=True)
        LOGGER.info(f"Created base directory {dir_base}")

        args.dir_base = dir_base

    if args.dir_model is None:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.dir_model = f"{now}_{net_name}"
        LOGGER.info(f"Defined model directory {args.dir_model}")

    dir_out = os.path.join(args.dir_base, args.dir_model)
    os.makedirs(dir_out, exist_ok=True)
    LOGGER.info(f"Created output directory {dir_out}")

    checkpoint_dir = os.path.abspath(os.path.join(dir_out, "checkpoint"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    summary_dir = os.path.abspath(os.path.join(dir_out, "summary"))
    os.makedirs(summary_dir, exist_ok=True)

    # save the configs
    with open(os.path.join(dir_out, "configs.yaml"), "w") as f:
        yaml.dump_all([net_conf, dlss_conf, msfm_conf], f)

    # TODO not hard code
    n_z_bins = 4

    # strategy = distribute.get_strategy(not args.local, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(num_packs=1))
    strategy = distribute.get_strategy(not args.local)

    # TODO implement some noise schedule?
    # https://cosmo-gitlab.phys.ethz.ch/jafluri/arne_handover/-/blob/main/networks/train_net.py#L184

    # like https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
    def dataset_fn(input_context):
        dset = fiducial_pipeline.get_fiducial_dset(
            # dset = fiducial_pipeline.get_fiducial_multi_noise_dset(
            tfr_pattern=args.fid_tfr_pattern,
            params=target_params,
            conf=msfm_conf,
            # n_noise=n_noise_per_example,
            **net_conf["dset"]["training"],
            # distribution
            input_context=input_context,
        )
        return dset

    dist_dset = strategy.distribute_datasets_from_function(dataset_fn)
    dist_iter = iter(dist_dset)

    # create all of the variables within the strategy's scope, such that they are mirrored
    with strategy.scope():
        # load the layers
        network = NETWORKS[net_conf["model"]["name"]](
            output_shape=n_params, **net_conf["model"]["kwargs"]
        ).get_layers()
        LOGGER.info(f"Loaded a network specification of type {NETWORKS[net_conf['model']['name']]}")

        # build the model
        model = DeltaLossModel(
            network=network,
            n_side=n_side,
            indices=data_vec_pix,
            n_neighbors=net_conf["model"]["n_neighbors"],
            max_checkpoints=net_conf["model"]["max_checkpoints"],
            input_shape=(None, len(data_vec_pix), n_z_bins),
            checkpoint_dir=checkpoint_dir,
            summary_dir=summary_dir,
            restore_checkpoint=args.restore_checkpoint,
        )

    # set up the training loss
    model.setup_delta_loss_step(
        n_params,
        net_conf["dset"]["training"]["local_batch_size"],
        perts,
        n_channels=n_z_bins,
        strategy=strategy,
        **dlss_conf["training"]["delta_loss"],
    )

    LOGGER.info(f"Starting training")
    LOGGER.timer.start("training")
    t_prev = time()

    # TODO wrap in tf.function (also use tf.range in that case)?
    for step in LOGGER.progressbar(range(1, n_steps + 1), at_level="info", total=n_steps, desc="training at fiducial"):
        # context for profiling like https://www.tensorflow.org/guide/profiler#profiling_custom_training_loops
        # optional context like https://stackoverflow.com/a/34798330
        with tf.profiler.experimental.Trace("step", step_num=step, _r=1) if args.profile else nullcontext():
            # train step
            dv_batch, index_batch = next(dist_iter)
            model.delta_train_step(dv_batch)

            # output
            if (output_every is not None) and (step % output_every == 0):
                print("\n")
                LOGGER.info(f"Done with {step}/{n_steps} training steps after {LOGGER.timer.elapsed('training')}")

            # checkpoint
            if (checkpoint_every is not None) and (step % checkpoint_every == 0):
                model.save_model()

            # evaluate
            if (eval_every is not None) and (step % eval_every == 0):
                eval.evaluate_grid(
                    model=model,
                    strategy=strategy,
                    tfr_pattern=args.grid_tfr_pattern,
                    msfm_conf=msfm_conf,
                    net_conf=net_conf,
                    dir_out=dir_out,
                    step=step
                )

                eval.evaluate_fiducial(
                    model=model,
                    strategy=strategy,
                    tfr_pattern=args.fid_tfr_pattern,
                    msfm_conf=msfm_conf,
                    net_conf=net_conf,
                    dir_out=dir_out,
                    step=step
                )

            # profile
            if args.profile and step == 200:
                print("\n")
                LOGGER.info(f"Starting to profile")
                tf.profiler.experimental.start(model.summary_dir)
            if args.profile and step == 205:
                print("\n")
                LOGGER.info(f"Stopping to profile")
                tf.profiler.experimental.stop()

            # log time per step
            with model.summary_writer.as_default():
                t_now = time()
                tf.summary.scalar("step_time", t_now - t_prev)
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


# only exists for debugging purposes
if __name__ == "__main__":
    # args = [
    #     "--tfr_pattern=/Users/arne/data/DESY3/tfrecords/v2/DESy3_fiducial_000.tfrecord",
    #     "--net_config=configs/resnet_debug.yaml",
    #     "--verbosity=debug",
    #     "--distributed",
    #     # "--dir_base=/Users/arne/data/DESY3/training"
    #     # "--dir_model=/Users/arne/data/DESY3/training/2023-02-28_11-39-54_resnet_small"
    #     # "--debug"
    # ]
    training()
