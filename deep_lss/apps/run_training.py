# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2023
Author: Arne Thomsen

Train the DeepSphere graph neural networks at the fiducial cosmology and its perturbations using the information
maximizing loss to find an informative summary statistic.

Meant for the GPU nodes of the Perlmutter cluster at NERSC.
TODO implement distributed training
TODO implement weights & biases versioning
TODO make esub compatible? The index could correspond to a neural net architecture in the hyperparameter search
"""

import tensorflow as tf
import os, argparse, warnings, yaml, time

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


# def setup(args):
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
    parser.add_argument("--with_bary", action="store_true", help="include baryons")
    parser.add_argument(
        "--tfr_pattern",
        type=str,
        default="/pscratch/sd/a/athomsen/DESY3/v2/fiducial",
        help="input root dir of the simulations",
    )
    parser.add_argument(
        "--dir_base",
        type=str,
        default=None,
        # TODO
        help="dir where the models are saved. It is generated within the repo according to the date and time if set to None",
    )
    parser.add_argument(
        "--dir_model",
        type=str,
        default=None,
        # TODO
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
    parser.add_argument("--distributed", action="store_true", help="distribute the training")
    parser.add_argument("--debug", action="store_true", help="activate debug mode")

    # args, _ = parser.parse_known_args(args)
    args, _ = parser.parse_known_args()

    # TODO create the model directory
    logger.set_all_loggers_level(args.verbosity)

    if args.debug:
        tf.config.run_functions_eagerly(True)
        # tf.config.set_soft_device_placement(False)
        # tf.debugging.set_log_device_placement(True)
        # tf.data.experimental.enable_debug_mode()
        LOGGER.warning(f"!!!!! Running the training in test mode, TensorFlow is executed eagerly !!!!!")

    return args


# def main(args):
def main():
    # args = setup(args)
    args = setup()
    LOGGER.timer.start("main")

    # check the devices
    try:
        n_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpus = os.cpu_count()
    LOGGER.info(f"Running on {n_cpus} CPU cores")

    n_gpus = len(tf.config.list_physical_devices("GPU"))
    if n_gpus == 0:
        LOGGER.warning(f"No GPU discovered, running on CPUs only")
    else:
        LOGGER.info(f"Running on {n_gpus} GPUs")

    try:
        n_gpus_cuda = len(os.environ["CUDA_VISIBLE_DEVICES"])
        assert n_gpus == n_gpus_cuda
    except KeyError:
        LOGGER.warning(f"No CUDA enabled GPUs found")

    # read the different configs
    dlss_conf = utils.load_deep_lss_config(args.dlss_config)
    msfm_conf = analysis.load_config(args.msfm_config)

    # general constants
    target_params = dlss_conf["training"]["target_params"]
    n_params = len(target_params)
    pert_labels = parameters.get_fiducial_perturbation_labels(target_params)
    perts = parameters.get_fiducial_perturbations(target_params)
    LOGGER.info(f"Training with respect to the parameters {target_params} with off sets {perts}")
    LOGGER.debug(f"The labels are {pert_labels}")

    n_side = msfm_conf["analysis"]["n_side"]
    data_vec_pix, _, _, _, _ = analysis.load_pixel_file(msfm_conf)

    # TODO could loop over esub indices here

    net_conf = input_output.read_yaml(args.net_config)

    # network constants
    net_name = net_conf["name"]
    n_steps = net_conf["training"]["n_steps"]
    output_every = net_conf["training"]["output_every"]
    checkpoint_every = net_conf["training"]["checkpoint_every"]
    eval_every = net_conf["training"]["eval_every"]

    global_batch_size = net_conf["dset"]["global_batch_size"]
    n_readers = net_conf["dset"]["n_readers"]
    file_name_shuffle_buffer = net_conf["dset"]["file_name_shuffle_buffer"]
    examples_shuffle_buffer = net_conf["dset"]["examples_shuffle_buffer"]

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

    # TODO implement some noise schedule?
    # https://cosmo-gitlab.phys.ethz.ch/jafluri/arne_handover/-/blob/main/networks/train_net.py#L184

    # like https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
    def dataset_fn(input_context):
        dset = fiducial_pipeline.get_fiducial_dset(
            tfr_pattern=args.tfr_pattern,
            pert_labels=pert_labels,
            batch_size=global_batch_size,
            conf=msfm_conf,
            n_batches=n_steps,
            # relevant for performance
            n_readers=n_readers,
            n_prefetch=tf.data.AUTOTUNE,
            file_name_shuffle_buffer=file_name_shuffle_buffer,
            examples_shuffle_buffer=examples_shuffle_buffer,
            input_context=input_context,
        )

        return dset

    # TODO define the distribution strategy
    if (n_gpus > 1) and (args.distributed) and (not args.debug):
        cross_device_ops = tf.distribute.NcclAllReduce()
        # cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)
        n_gpus_strategy = strategy.num_replicas_in_sync
        assert n_gpus_strategy == n_gpus
        LOGGER.info(f"Training is distributed, using the MirroredStrategy")
    else:
        strategy = tf.distribute.get_strategy()
        n_gpus_strategy = 1
        LOGGER.warning(f"Training is not distributed, using the default strategy")

    with strategy.scope():
        # distribute the dataset
        dist_dset = strategy.distribute_datasets_from_function(dataset_fn)

        # load the layers
        network = NETWORKS[net_conf["model"]["name"]](
            output_shape=n_params, **net_conf["model"]["params"]
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
            summary_dir=summary_dir,
            restore_checkpoint=args.restore_checkpoint,
        )

        # set up the training loss
        model.setup_delta_loss_step(
            n_params,
            global_batch_size,
            perts,
            n_channels=n_z_bins,
            strategy=strategy,
            **dlss_conf["training"]["delta_loss"],
        )

        LOGGER.info(f"Starting training")
        counter = 0
        LOGGER.timer.start("training")
        for data_vectors, label in LOGGER.progressbar(dist_dset, at_level="info", total=n_steps):
            model.delta_train_step(data_vectors)

            # output
            if (output_every is not None) and (counter % output_every == 0):
                LOGGER.info(f"Done with {counter}/{n_steps} training steps after {LOGGER.timer.elapsed('training')}")

            # checkpoint
            if (checkpoint_every is not None) and (counter % checkpoint_every == 0):
                model.save_model()

            # evaluate
            if (eval_every is not None) and (counter % eval_every == 0):
                pass

            counter += 1

        LOGGER.info(f"Finished training after {n_steps} steps and {LOGGER.timer.elapsed('training')}")

        # save everything at the end if necessary
        if (checkpoint_every is not None) and (counter % checkpoint_every != 0):
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
    main()
