import os
import numpy as np

from msfm.utils import input_output, logger, files

LOGGER = logger.get_logger(__file__)


# like https://github.com/des-science/multiprobe-simulation-forward-model/blob/main/msfm/utils/analysis.py#L21,
# but for this repo instead of the multiprobe-simulation-forward-model one
def load_deep_lss_config(conf=None):
    """Loads or passes through a config

    Args:
        conf (str, dict, optional): Can be either a string (a config.yaml is read in), a dictionary (the config is
            passed through) or None (the default config is loaded). Defaults to None.

    Raises:
        ValueError: When an invalid conf is passed

    Returns:
        dict: A configuration dictionary
    """
    # load the default config within this repo
    if conf is None:
        file_dir = os.path.dirname(__file__)
        repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
        conf = os.path.join(repo_dir, "configs/dlss_config.yaml")
        LOGGER.warning(f"Loading the default config from {conf}")
        conf = input_output.read_yaml(conf)

    # load a config specified by a path
    elif isinstance(conf, str):
        conf = input_output.read_yaml(conf)

    # pass through an existing config
    elif isinstance(conf, dict):
        pass

    else:
        raise ValueError(f"conf {conf} must be None, a str specifying the path to the .yaml file, or the read dict")

    LOGGER.info(f"Loaded the config")
    return conf


def get_smoothing_kwargs(loss_function, msfm_conf, dlss_conf, net_conf, dir_base=None, mode="training"):
    """Build a dictionary of keyword arguments for the deepsphere.healpy_layers.HealpySmoothing layer.

    Args:
        loss_function (str): One of "delta", "mse", "likelihood", "mutual_info"
        msfm_conf (dict): Multiprobe-simulation-forward-model config.
        dlss_conf (dict): Network training config.
        net_conf (dict): Network architecture config.
        dir_base (str, optional): Directory to store the smoothing kernel. Defaults to None.

    Returns:
        dict: keyword arguments for deepsphere.healpy_layers.HealpySmoothing
    """
    # msfm
    n_side = msfm_conf["analysis"]["n_side"]
    data_vec_pix, _, _, _ = files.load_pixel_file(msfm_conf)
    mask_dict = files.get_tomo_dv_masks(msfm_conf)

    # dlss
    with_lensing = dlss_conf["dset"]["common"]["with_lensing"]
    with_clustering = dlss_conf["dset"]["common"]["with_clustering"]

    if with_lensing and with_clustering:
        mask = np.concatenate([mask_dict["metacal"], mask_dict["maglim"]], axis=1)
    elif with_lensing and not with_clustering:
        mask = mask_dict["metacal"]
    elif not with_lensing and with_clustering:
        mask = mask_dict["maglim"]
    else:
        raise ValueError("At least one of with_lensing and with_clustering must be True")

    try:
        fwhm = []
        white_noise_sigma = []
        map_normalization = []
        if with_lensing:
            fwhm += dlss_conf["scale_cuts"]["lensing"]["theta_fwhm"]
            white_noise_sigma += dlss_conf["scale_cuts"]["lensing"]["white_noise_sigma"]
            map_normalization += msfm_conf["analysis"]["normalization"]["lensing"]
        if with_clustering:
            fwhm += dlss_conf["scale_cuts"]["clustering"]["theta_fwhm"]
            white_noise_sigma += dlss_conf["scale_cuts"]["clustering"]["white_noise_sigma"]
            map_normalization += msfm_conf["analysis"]["normalization"]["clustering"]

        arcmin = dlss_conf["scale_cuts"]["arcmin"]
        n_sigma_support = dlss_conf["scale_cuts"]["n_sigma_support"]

        params = dlss_conf["dset"]["training"]["params"]
        n_params = len(params)

        if dlss_conf["dset"]["common"]["apply_norm"]:
            white_noise_sigma = np.array(white_noise_sigma) / np.array(map_normalization)

        # net
        if mode == "training":
            if loss_function == "delta":
                local_batch_size = net_conf["dset"][mode]["fiducial"]["local_batch_size"]
                effective_local_batch_size = local_batch_size * (2 * n_params + 1)
            else:
                local_batch_size = net_conf["dset"][mode]["grid"]["local_batch_size"]
                effective_local_batch_size = local_batch_size
        else:
            if loss_function == "delta":
                effective_local_batch_size = net_conf["dset"]["eval"]["fiducial"]["local_batch_size"]
            else:
                effective_local_batch_size = net_conf["dset"]["eval"]["grid"]["local_batch_size"]

        smoothing_kwargs = {
            "nside": n_side,
            "indices": data_vec_pix,
            "nest": True,
            "mask": mask,
            "fwhm": fwhm,
            "arcmin": arcmin,
            "n_sigma_support": n_sigma_support,
            "max_batch_size": effective_local_batch_size,
            "white_noise_sigma": white_noise_sigma,
        }

        if dir_base is not None:
            smoothing_kwargs["data_path"] = os.path.join(dir_base, "smoothing")

    except (TypeError, KeyError):
        LOGGER.warning("Could not build smoothing_kwargs")
        smoothing_kwargs = None

    return smoothing_kwargs


def get_backend_floatx():
    """Returns the current backend float of the keras backend.

    Raises:
        ValueError: If something other than tf.float32 or tf.float64 is used.

    Returns:
        tf.floatx: either tf.float32 or tf.float64 depending on the current backend setting
    """
    import tensorflow as tf

    if tf.keras.backend.floatx() == "float32":
        return tf.float32
    elif tf.keras.backend.floatx() == "float64":
        return tf.float64
    else:
        raise ValueError(
            f"The only suppored keras backend floatx are float64 and float32 not "
            f"{tf.keras.backend.floatx()}! Please use tf.keras.backend.set_floatx to set an appropiate value."
        )


def convert_dotted_to_nested_dict(dotted_dict):
    """Convert a dictionary like {'a.b.c': 1, 'a.b.d': 2, 'a.e': 3} to a nested dictionary like
    {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}. This is needed to handle wandb configs in hyperparameter sweeps. Modified
    from ChatGPT.

    Args:
        dotted_dict (dict): Dictionary with only one level of keys, where the keys are strings with dots.

    Returns:
        dict: A dictionary where the dots have been converted into nesting.
    """

    nested_dict = {}
    for key, value in dotted_dict.items():
        keys = key.split(".")
        current_dict = nested_dict

        for k in keys[:-1]:
            current_dict = current_dict.setdefault(k, {})

        current_dict[keys[-1]] = value

    return nested_dict


def update_nested_dict(original_dict, update_dict):
    """
    Recursively updates a nested dictionary with the key-value pairs from another dictionary. Written by ChatGPT.

    Args:
        original_dict (dict): The original dictionary to be updated.
        update_dict (dict): The dictionary containing the key-value pairs to update the original dictionary.

    Returns:
        dict: The updated dictionary.

    """
    for key, value in update_dict.items():
        if key in original_dict and isinstance(original_dict[key], dict) and isinstance(value, dict):
            # recursively update nested dictionaries
            original_dict[key] = update_nested_dict(original_dict[key], value)
        else:
            # update non-dictionary values or add new key-value pairs
            original_dict[key] = value

    return original_dict
