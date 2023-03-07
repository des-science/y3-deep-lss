# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2023
Author: Arne Thomsen

Evaluate the DeepSphere graph neural networks on the grid of cosmologies sampled in the CosmoGrid

Meant for the GPU nodes of the Perlmutter cluster at NERSC.
"""

import tensorflow as tf
import os, argparse, warnings, h5py, math

from msfm import grid_pipeline
from msfm.utils import logger, input_output, analysis, parameters

from deep_lss.utils import utils, distribute
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
        "--tfr_pattern",
        type=str,
        default="/pscratch/sd/a/athomsen/DESY3/v2/fiducial/DESy3_fiducial_???.tfrecord",
        help="input root dir of the simulations",
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
    LOGGER.debug(f"--tfr_pattern = {args.tfr_pattern}")
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


def stack_cosmos(tensors, n_examples_per_cosmo):
    """TODO

    Args:
        tensors (_type_): _description_
        n_examples_per_cosmo (_type_): _description_

    Returns:
        _type_: _description_
    """
    # concatenate all of the cosmologies into the first axis, shape (n_cosmos * n_examples_per_cosmo, n_output)
    tensors = tf.concat(tensors, axis=0)
    # split according to the cosmology, list of len n_cosmos with elements of shape (n_examples_per_cosmo, n_output)
    tensors = tf.split(tensors, tensors.shape[0] // n_examples_per_cosmo)
    # stack the cosmologies into the 0th axis, shape (n_cosmos, n_examples_per_cosmo, n_output)
    tensors = tf.stack(tensors, axis=0)

    return tensors


def evaluation():
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
    n_cosmos = msfm_conf["grid"]["n_cosmos"]
    n_patches = msfm_conf["analysis"]["n_patches"]
    n_perms_per_cosmo = msfm_conf["analysis"]["grid"]["n_perms_per_cosmo"]
    n_noise_per_example = msfm_conf["analysis"]["grid"]["n_noise_per_example"]
    n_examples_per_cosmo = n_patches * n_perms_per_cosmo * n_noise_per_example
    n_examples = n_cosmos * n_examples_per_cosmo
    LOGGER.info(f"There's a total of {n_examples} to be evaluated")

    n_side = msfm_conf["analysis"]["n_side"]
    data_vec_pix, _, _, _, _ = analysis.load_pixel_file(msfm_conf)

    # network constants
    net_name = net_conf["name"]
    global_batch_size = net_conf["dset"]["evaluation"]["global_batch_size"]
    n_readers = net_conf["dset"]["n_readers"]

    # set up directories
    checkpoint_dir = os.path.abspath(os.path.join(args.dir_model, "checkpoint"))
    # evals_dir = os.path.abspath(os.path.join(args.dir_model, "evals"))
    evals_file = os.path.abspath(os.path.join(args.dir_model, "evals.h5"))

    # TODO not hard code
    n_z_bins = 4

    strategy = distribute.get_strategy(not args.local)

    local_batch_size = distribute.get_local_batch_size(strategy, global_batch_size)

    n_steps = math.ceil(n_examples, global_batch_size)

    # like https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
    def dataset_fn(input_context):
        dset = grid_pipeline.get_grid_dset(
            tfr_pattern=args.tfr_pattern,
            local_batch_size=local_batch_size,
            n_params=len(all_params),
            conf=msfm_conf,
            # n_noise=3,
            # relevant for performance
            n_readers=n_readers,
            n_prefetch=tf.data.AUTOTUNE,
            # distribution
            input_context=input_context,
        )
        return dset

    dist_dset = strategy.distribute_datasets_from_function(dataset_fn)

    # create all of the variables within the strategy's scope, such that they are mirrored
    with strategy.scope():
        # load the layers
        network = NETWORKS[net_conf["model"]["name"]](
            output_shape=n_output, **net_conf["model"]["params"]
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

    LOGGER.info(f"Starting evaluation")
    LOGGER.timer.start("eval")

    step = 1
    preds = []
    cosmos = []
    sobols = []
    for dv_batch, cosmo_batch, index_batch in LOGGER.progressbar(dist_dset, at_level="info", total=n_steps):
        # DistributedValues of shape (local_batch_size, n_output)
        pred_batch = strategy.run(model, args=(dv_batch,))

        # shape (global_batch_size, n_output)
        pred_batch = strategy.gather(pred_batch, axis=0)

        print(pred_batch)

        # TODO check order as https://www.tensorflow.org/tutorials/distribute/input#caveats

        preds.append(pred_batch)
        cosmos.append(cosmo_batch)
        sobols.append(index_batch[0])

        step += 1

    LOGGER.info(
        f"Finished looping over the whole dataset training after {step} steps and {LOGGER.timer.elapsed('eval')}"
    )

    preds = stack_cosmos(preds, n_examples_per_cosmo)

    # should be (n_cosmos, n_examples_per_cosmo, n_output)
    print(preds.shape)


    cosmos = stack_cosmos(cosmos, n_examples_per_cosmo)
    sobols = stack_cosmos(sobols, n_examples_per_cosmo)
    LOGGER.info(f"Reshaped the results")


    # TODO in evals_dir
    with h5py.File(evals_file, "w") as f:
        f.create_dataset(name="pred", shape=(n_cosmos, n_examples_per_cosmo, n_output))
        f["preds"] = preds.numpy()

        f.create_dataset(name="cosmos", shape=(n_cosmos, n_examples_per_cosmo, len(all_params)))
        f["cosmos"] = cosmos.numpy()
        
        f.create_dataset(name="sobols", shape=(n_cosmos, n_examples_per_cosmo))
        f["sobols"] = sobols.numpy()

    LOGGER.info(f"Saved the resulting tensors")



if __name__ == "__main__":
    evaluation()
