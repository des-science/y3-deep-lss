# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Train the DeepSphere graph neural networks at the fiducial cosmology and its perturbations using the information
maximizing loss to find an informative summary statistic.

Meant for the GPU nodes of the Perlmutter cluster at NERSC.
TODO implement distributed training
TODO implement weights & biases versioning
"""

import tensorflow as tf
import os, argparse, warnings, yaml

from datetime import datetime

from msfm import fiducial_pipeline
from msfm.utils import logger, input_output, analysis, parameters

from deep_lss.utils import utils
from deep_lss.models.delta_model import DeltaLossModel
from deep_lss.nets import NETWORKS

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)

# TODO
# def resources(args):
#     return dict(main_memory=8192, main_time=4, main_scratch=0, main_n_cores=1)


def setup(args):
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
    parser.add_argument("--with_bary", action="store_true", help="include baryons")
    parser.add_argument(
        "--tfr_pattern",
        type=str,
        default="/pscratch/sd/a/athomsen/DESY3/v2/fiducial",
        help="input root dir of the simulations",
    )
    parser.add_argument(
        "--dir_out",
        type=str,
        default=None,
        help="dir where the models are saved. It is generated within the repo according to the date and time if set to None",
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
    parser.add_argument("--debug", action="store_true", help="activate debug mode")

    args, _ = parser.parse_known_args(args)

    # TODO create the model directory
    logger.set_all_loggers_level(args.verbosity)

    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.config.set_soft_device_placement(False)
        tf.debugging.set_log_device_placement(True)
        tf.data.experimental.enable_debug_mode()
        LOGGER.warning(f"!!!!! Running the training in test mode, TensorFlow is executed eagerly !!!!!")

    return args


def main(indices, args):
    args = setup(args)
    LOGGER.timer.start("main")

    try:
        LOGGER.info(f"Running on {len(os.sched_getaffinity(0))} cores")
    except AttributeError:
        pass

    LOGGER.warning(f"tf.config.list_physical_devices | {tf.config.list_physical_devices()}")
    try:
        LOGGER.warning(f"os.environ['CUDA_VISIBLE_DEVICES'] | {os.environ['CUDA_VISIBLE_DEVICES']}")
    except KeyError:
        pass

    # read the different configs
    dlss_conf = utils.load_deep_lss_config(args.dlss_config)
    msfm_conf = analysis.load_config(args.msfm_config)

    # define constants
    target_params = dlss_conf["training"]["target_params"]
    n_params = len(target_params)
    pert_labels = parameters.get_fiducial_perturbation_labels(target_params)
    perts = parameters.get_fiducial_perturbations(target_params)
    LOGGER.info(f"Training with respect to the parameters {target_params} with off sets {perts}")
    LOGGER.debug(f"The labels are {pert_labels}")

    n_side = msfm_conf["analysis"]["n_side"]
    data_vec_pix, _, _, _, _ = analysis.load_pixel_file(msfm_conf)

    # TODO index corresponds to a neural net architecture in the hyperparameter search 
    for index in indices:
        # TODO somehow use index to select a net config
        net_conf = input_output.read_yaml(args.net_config)

        net_name = net_conf["name"]
        n_steps = net_conf["training"]["n_steps"]
        output_every = net_conf["training"]["output_every"]
        checkpoint_every = net_conf["training"]["checkpoint_every"]
        eval_every = net_conf["training"]["eval_every"]

        # create directories
        if args.dir_out is None:
            now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

            file_dir = os.path.dirname(__file__)
            repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
            dir_out = os.path.join(repo_dir, f"run_files/{now}_{net_name}")
            os.makedirs(dir_out, exist_ok=True)
            LOGGER.info(f"Created base path {dir_out}")

            args.dir_out = dir_out

        checkpoint_dir = os.path.abspath(os.path.join(args.dir_out, "checkpoint"))
        input_output.robust_makedirs(checkpoint_dir)
        summary_dir = os.path.abspath(os.path.join(args.dir_out, "summary"))
        input_output.robust_makedirs(summary_dir)

        # save the configs
        with open(os.path.join(args.dir_out, "configs.yaml"), "w") as f:
            yaml.dump_all([net_conf, dlss_conf, msfm_conf], f)

        # TODO not hard code
        n_z_bins = 4

        # network
        batch_size = net_conf["training"]["batch_size"]

        # TODO implement some noise schedule?
        # https://cosmo-gitlab.phys.ethz.ch/jafluri/arne_handover/-/blob/main/networks/train_net.py#L184
        fiducial_dset = fiducial_pipeline.get_fiducial_dset(
            tfr_pattern=args.tfr_pattern,
            pert_labels=pert_labels,
            batch_size=batch_size,
            conf=msfm_conf,
            # relevant for performance
            n_readers=8,
            n_prefetch=tf.data.AUTOTUNE,
            file_name_shuffle_buffer=128,
            examples_shuffle_buffer=128,
        )

        network = NETWORKS[net_conf["model"]["name"]](output_shape=n_params, **net_conf["model"]["params"]).get_layers()
        LOGGER.info(f"Loaded a network of type {NETWORKS[net_conf['model']['name']]}")

        model = DeltaLossModel(
            network=network,
            n_side=n_side,
            indices=data_vec_pix,
            n_neighbors=dlss_conf["networks"]["n_neighbors"],
            input_shape=(None, len(data_vec_pix), n_z_bins),
            checkpoint_dir=checkpoint_dir,
            summary_dir=summary_dir,
            restore_checkpoint=args.restore_checkpoint,
        )

        # use default parameters for now
        model.setup_delta_loss_step(
            n_params, batch_size, perts, n_channels=n_z_bins, **dlss_conf["training"]["delta_loss"]
        )

        LOGGER.info(f"Starting training")
        counter = 0
        for data_vectors, index in fiducial_dset.take(n_steps):
            model.delta_train_step(data_vectors)

            # output
            if (output_every is not None) and (counter % output_every == 0):
                LOGGER.info(f"Done with {counter}/{n_steps} training steps")

            # checkpoint
            if (checkpoint_every is not None) and (counter % checkpoint_every == 0):
                model.save_model()

            # evaluate
            if (eval_every is not None) and (counter % eval_every == 0):
                pass

            counter += 1

        # TODO esub yield statement

        LOGGER.info(f"Finished training after {n_steps} steps")

        # save everything at the end if necessary
        if (checkpoint_every is not None) and (counter % checkpoint_every != 0):
            LOGGER.info(f"Creating a final checkpoint")
            model.save_model()
        else:
            LOGGER.info(f"A final checkpoint still exists")

        yield index

    LOGGER.info(f"Finished")
