# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2023
Author: Arne Thomsen

Evaluate the DeepSphere graph neural networks on the grid of cosmologies sampled in the CosmoGrid

Meant for the GPU nodes of the Perlmutter cluster at NERSC.
"""

import tensorflow as tf
import os, argparse, warnings

from msfm.utils import logger, input_output, analysis

from deep_lss.utils import utils, distribute, eval
from deep_lss.models.delta_model import DeltaLossModel
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
        "--fid_tfr_pattern",
        type=str,
        default="/pscratch/sd/a/athomsen/DESY3/v2/fiducial/DESy3_fiducial_???.tfrecord",
        help="input root dir of the simulations",
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
        "--dir_model", type=str, required=True, help="dir where the model checkpoints to be loaded are saved."
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
    parser.add_argument("--local", action="store_true", help="distribute the training")
    parser.add_argument("--debug", action="store_true", help="activate debug mode")

    args, _ = parser.parse_known_args()

    logger.set_all_loggers_level(args.verbosity)

    LOGGER.debug(f"--verbosity = {args.verbosity}")
    LOGGER.debug(f"--fid_tfr_pattern = {args.fid_tfr_pattern}")
    LOGGER.debug(f"--grid_tfr_pattern = {args.grid_tfr_pattern}")
    LOGGER.debug(f"--dir_model = {args.dir_model}")
    LOGGER.debug(f"--net_config = {args.net_config}")
    LOGGER.debug(f"--dlss_config = {args.dlss_config}")
    LOGGER.debug(f"--msfm_config = {args.msfm_config}")
    LOGGER.debug(f"--local = {args.local}")
    LOGGER.debug(f"--debug = {args.debug}")

    if args.debug:
        # tf.config.run_functions_eagerly(True)
        # tf.config.set_soft_device_placement(False)
        tf.debugging.set_log_device_placement(True)
        # tf.data.experimental.enable_debug_mode()
        LOGGER.warning(f"!!!!! Running the training in test mode, TensorFlow is executed eagerly !!!!!")

    return args


if __name__ == "__main__":
    args = setup()
    LOGGER.timer.start("main")

    _, _ = distribute.check_devices()

    # read the different configs
    dlss_conf = utils.load_deep_lss_config(args.dlss_config)
    msfm_conf = analysis.load_config(args.msfm_config)
    net_conf = input_output.read_yaml(args.net_config)

    # general constants
    all_params = msfm_conf["analysis"]["params"]
    target_params = dlss_conf["training"]["target_params"]
    n_output = len(target_params)
    LOGGER.info(f"The networks have output shape {n_output} and target {target_params}")

    # pipeline constants
    n_side = msfm_conf["analysis"]["n_side"]
    data_vec_pix, _, _, _, _ = analysis.load_pixel_file(msfm_conf)

    # set up directories
    checkpoint_dir = os.path.abspath(os.path.join(args.dir_model, "checkpoint"))

    # TODO not hard code
    n_z_bins = 4

    strategy = distribute.get_strategy(not args.local, cross_device_ops=tf.distribute.ReductionToOneDevice())

    # create all of the variables within the strategy's scope, such that they are mirrored
    with strategy.scope():
        # load the layers
        network = NETWORKS[net_conf["model"]["name"]](
            output_shape=n_output, **net_conf["model"]["kwargs"]
        ).get_layers()
        LOGGER.info(f"Loaded a network specification of type {NETWORKS[net_conf['model']['name']]}")

        # build the model
        model = DeltaLossModel(
            network=network,
            n_side=n_side,
            indices=data_vec_pix,
            n_neighbors=net_conf["model"]["n_neighbors"],
            input_shape=(None, len(data_vec_pix), n_z_bins),
            checkpoint_dir=checkpoint_dir,
            # always load from a checkpoint
            restore_checkpoint=True,
        )

    eval.evaluate_grid(
        model=model,
        strategy=strategy,
        tfr_pattern=args.grid_tfr_pattern,
        msfm_conf=msfm_conf,
        net_conf=net_conf,
        dir_out=args.dir_model,
    )

    eval.evaluate_fiducial(
        model=model,
        strategy=strategy,
        tfr_pattern=args.fid_tfr_pattern,
        msfm_conf=msfm_conf,
        net_conf=net_conf,
        dir_out=args.dir_model,
    )
