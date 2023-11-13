import os
import tensorflow as tf

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


def get_smoothing_kwargs(msfm_conf, dlss_conf, net_conf, dir_base=None):
    """Build a dictionary of keyword arguments for the deepsphere.healpy_layers.HealpySmoothing layer.

    Args:
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
    with_lensing = dlss_conf["dset"]["general"]["with_lensing"]
    with_clustering = dlss_conf["dset"]["general"]["with_clustering"]

    if with_lensing and with_clustering:
        mask = tf.concat([mask_dict["metacal"], mask_dict["maglim"]], axis=1)
    elif with_lensing and not with_clustering:
        mask = mask_dict["metacal"]
    elif not with_lensing and with_clustering:
        mask = mask_dict["maglim"]
    else:
        raise ValueError("At least one of with_lensing and with_clustering must be True")

    fwhm = []
    if with_lensing:
        fwhm += dlss_conf["scale_cuts"]["lensing"]["theta_fwhm"]
    if with_clustering:
        fwhm += dlss_conf["scale_cuts"]["clustering"]["theta_fwhm"]

    arcmin = dlss_conf["scale_cuts"]["arcmin"]
    n_sigma_support = dlss_conf["scale_cuts"]["n_sigma_support"]

    params = dlss_conf["dset"]["training"]["params"]
    n_params = len(params)

    # net
    local_batch_size = net_conf["dset"]["training"]["local_batch_size"]
    local_delta_loss_batch_size = local_batch_size * (2 * n_params + 1)

    smoothing_kwargs = {
        "nside": n_side,
        "indices": data_vec_pix,
        "nest": True,
        "mask": mask,
        "fwhm": fwhm,
        "arcmin": arcmin,
        "n_sigma_support": n_sigma_support,
        "max_batch_size": local_delta_loss_batch_size,
    }

    if dir_base is not None:
        smoothing_kwargs["data_path"] = os.path.join(dir_base, "smoothing")

    return smoothing_kwargs
