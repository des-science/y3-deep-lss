import os

from msfm.utils import input_output, logger

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
        conf = os.path.join(repo_dir, "configs/config.yaml")
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
