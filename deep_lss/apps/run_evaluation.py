# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2023
Author: Arne Thomsen

Evaluate the DeepSphere graph neural networks on the grid of cosmologies sampled in the CosmoGrid

Meant for the GPU nodes of the Perlmutter cluster at NERSC.
"""

import tensorflow as tf
import os, argparse, warnings, yaml

from msfm.utils import logger, files

from deep_lss.utils import configuration, distribute, eval
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
        "--dir_model", type=str, required=True, help="dir where the model checkpoints to be loaded are saved."
    )
    parser.add_argument("--debug", action="store_true", help="activate debug mode")
    parser.add_argument("--file_label", type=str, default=None, help="A suffix that is appended to the files")

    args, _ = parser.parse_known_args()

    logger.set_all_loggers_level(args.verbosity)

    # print arguments
    logger.set_all_loggers_level(args.verbosity)
    for key, value in vars(args).items():
        LOGGER.info(f"{key} = {value}")

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
    strategy = distribute.get_strategy(args.dist_strategy)

    # load the configs
    with open(os.path.join(args.dir_model, "configs.yaml"), "r") as f:
        net_conf, dlss_conf, msfm_conf = list(yaml.load_all(f, Loader=yaml.FullLoader))

    LOGGER.info(f"Loaded configs from the model directory")

    # general constants
    all_params = msfm_conf["analysis"]["params"]
    target_params = dlss_conf["dset"]["training"]["params"]
    n_output = len(target_params)
    LOGGER.info(f"The networks have output shape {n_output} and target {target_params}")

    # pipeline constants
    n_side = msfm_conf["analysis"]["n_side"]
    data_vec_pix, _, _, _ = files.load_pixel_file(msfm_conf)

    n_z_bins = 0
    if dlss_conf["dset"]["general"]["with_lensing"]:
        n_z_bins += len(msfm_conf["survey"]["metacal"]["z_bins"])
    if dlss_conf["dset"]["general"]["with_clustering"]:
        n_z_bins += len(msfm_conf["survey"]["maglim"]["z_bins"])

    smoothing_kwargs = configuration.get_smoothing_kwargs(msfm_conf, dlss_conf, net_conf, dir_base=args.dir_model)

    # set up directories
    checkpoint_dir = os.path.abspath(os.path.join(args.dir_model, "checkpoint"))

    # create all of the variables within the strategy's scope, such that they are mirrored
    with strategy.scope():
        # load the layers
        network = NETWORKS[net_conf["model"]["name"]](
            output_shape=n_output, smoothing_kwargs=smoothing_kwargs, **net_conf["model"]["kwargs"]
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
            strategy=strategy,
        )

    train_step = model.get_step()

    if args.file_label is None:
        file_label = train_step
    else:
        file_label = f"{train_step}_{args.file_label}"

    # fiducial training
    if args.fidu_train_tfr_pattern is not None:
        eval.evaluate_fiducial(
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
        eval.evaluate_fiducial(
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
        eval.evaluate_grid(
            model=model,
            tfr_pattern=args.grid_vali_tfr_pattern,
            msfm_conf=msfm_conf,
            dlss_conf=dlss_conf,
            net_conf=net_conf,
            dir_out=args.dir_model,
            file_label=file_label,
        )
    else:
        LOGGER.warning(f"Skipping evaluation of the grid set")
