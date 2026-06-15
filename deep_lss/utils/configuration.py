import os
import numpy as np

from msfm.utils import input_output, logger, files

LOGGER = logger.get_logger(__file__)


def read_split_configs(probes_config, scales_config=None, loss_config=None):
    """Build the dlss_conf dict from the orthogonal split configs.

    The split configs define disjoint top-level namespaces: probes provides ``dset.*``,
    scales provides ``scale_cuts.*``, and loss provides ``loss_function`` plus the
    ``<loss>_loss`` / ``mutual_info_loss`` blocks. Because the top-level keys do not overlap,
    a shallow ``dict.update`` is the correct merge.

    Args:
        probes_config (str): path to the probes config (required).
        scales_config (str, optional): path to the scales config. Merged in if given.
        loss_config (str, optional): path to the loss config. Merged in if given. The maps
            pipeline folds the loss into dlss_conf this way; the Cls pipeline keeps it separate
            and so omits this argument.

    Returns:
        dict: the merged dlss_conf.
    """
    dlss_conf = input_output.read_yaml(probes_config)
    if scales_config:
        dlss_conf.update(input_output.read_yaml(scales_config))
    if loss_config:
        dlss_conf.update(input_output.read_yaml(loss_config))
    LOGGER.info("Loaded the split configs")
    return dlss_conf


def get_smooth_nside_indices(indices_nside_in, nside_in, smooth_nside):
    """Derive footprint pixel indices and a parent-mapping array at smooth_nside from nside_in indices.

    For HEALPix NEST ordering, pixel j at nside_in belongs to parent pixel j // downscale at smooth_nside, where
    downscale = (nside_in / smooth_nside)^2. The returned parent_output_idx maps each nside_in pixel to its
    (0-based) row in the smooth_nside output tensor.

    Args:
        indices_nside_in (np.ndarray): 1-D array of HEALPix NEST pixel indices at nside_in.
        nside_in (int): Input HEALPix resolution parameter (power of 2).
        smooth_nside (int): Target HEALPix resolution parameter (power of 2, < nside_in).

    Returns:
        smooth_indices (np.ndarray): Sorted 1-D array of unique NEST pixel indices at smooth_nside covering the
            footprint.
        parent_output_idx (np.ndarray): 1-D int array of length len(indices_nside_in). Entry j gives the row index
            in smooth_indices that nside_in pixel j maps to.
    """
    assert nside_in % smooth_nside == 0, f"nside_in {nside_in} must be divisible by smooth_nside {smooth_nside}"
    ratio = nside_in // smooth_nside
    assert ratio & (ratio - 1) == 0, f"nside_in / smooth_nside = {ratio} must be a power of 2"
    downscale = ratio ** 2
    parent_pix = indices_nside_in // downscale
    smooth_indices = np.unique(parent_pix)
    parent_output_idx = np.searchsorted(smooth_indices, parent_pix).astype(np.int32)
    return smooth_indices, parent_output_idx


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
    with_cross = dlss_conf["dset"]["common"].get("with_cross", False)

    if with_cross:
        # mirrors the per-pixel mask used in msfm.grid_pipeline._augmentations for the cross maps:
        # AND of the two probe masks, broadcast across all n_z_cross channels.
        mask_metacal_total = np.prod(mask_dict["metacal"], axis=-1, keepdims=True)
        mask_maglim_total = np.prod(mask_dict["maglim"], axis=-1, keepdims=True)
        mask = mask_metacal_total * mask_maglim_total
    elif with_lensing and with_clustering:
        mask = np.concatenate([mask_dict["metacal"], mask_dict["maglim"]], axis=1)
    elif with_lensing and not with_clustering:
        mask = mask_dict["metacal"]
    elif not with_lensing and with_clustering:
        mask = mask_dict["maglim"]
    else:
        raise ValueError("At least one of with_lensing, with_clustering, or with_cross must be True")

    smooth_nside = net_conf["network"].get("smooth_nside", None)
    if smooth_nside is not None and smooth_nside < n_side:
        smooth_indices, parent_output_idx = get_smooth_nside_indices(data_vec_pix, n_side, smooth_nside)
        # downsample the per-channel mask to smooth_nside using per-parent averaging
        n_pix_out = len(smooth_indices)
        counts = np.bincount(parent_output_idx, minlength=n_pix_out).astype(np.float32)
        mask_smooth = np.stack(
            [np.bincount(parent_output_idx, weights=mask[:, c].astype(np.float32), minlength=n_pix_out) / counts
             for c in range(mask.shape[1])],
            axis=1,
        ).astype(np.float32)
        LOGGER.info(f"Downsampling smoothing from nside={n_side} to smooth_nside={smooth_nside}: "
                    f"{len(data_vec_pix)} → {n_pix_out} pixels")
    else:
        smooth_nside = n_side
        smooth_indices = data_vec_pix
        mask_smooth = mask

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
        if with_cross:
            # The 16 (n_z_metacal x n_z_maglim) cross bins are always derived from the lensing and clustering blocks
            # above. alm_cross = sqrt(alm_k * alm_d) → effective Gaussian beam sigma_b^2 averages, and for independent
            # zero-mean complex Gaussian white noise the cross alm has <|alm_cross|^2> = (pi/4) * sigma_k * sigma_d *
            # Omega_pix (still flat in l).
            fwhm_k = np.asarray(dlss_conf["scale_cuts"]["lensing"]["theta_fwhm"], dtype=float)
            fwhm_d = np.asarray(dlss_conf["scale_cuts"]["clustering"]["theta_fwhm"], dtype=float)
            sig_k = np.asarray(dlss_conf["scale_cuts"]["lensing"]["white_noise_sigma"], dtype=float)
            sig_d = np.asarray(dlss_conf["scale_cuts"]["clustering"]["white_noise_sigma"], dtype=float)
            # outer-product over (i_metacal, j_maglim), flattened in (i * n_z_maglim + j) order
            # to match the cross-map ordering in msfm.apps.run_grid_postprocessing
            fwhm_cross = np.sqrt((fwhm_k[:, None] ** 2 + fwhm_d[None, :] ** 2) / 2.0).ravel()
            sigma_cross = np.sqrt(np.pi / 4.0) * np.sqrt(sig_k[:, None] * sig_d[None, :]).ravel()
            fwhm += fwhm_cross.tolist()
            white_noise_sigma += sigma_cross.tolist()
            # cross maps are not normalized in msfm.grid_pipeline._augmentations
            map_normalization += [1.0] * fwhm_cross.size

        arcmin = dlss_conf["scale_cuts"]["arcmin"]
        n_sigma_support = dlss_conf["scale_cuts"]["n_sigma_support"]

        params = dlss_conf["dset"]["training"]["params"]
        n_params = len(params)

        if dlss_conf["dset"]["common"]["apply_norm"]:
            white_noise_sigma = np.array(white_noise_sigma) / np.array(map_normalization)

        # scale white noise for lower nside: sigma ∝ 1/sqrt(pixel_area) ∝ nside
        white_noise_sigma = np.array(white_noise_sigma) * (smooth_nside / n_side)

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
            "nside": smooth_nside,
            "indices": smooth_indices,
            "nest": True,
            "mask": mask_smooth,
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


def get_cls_bounds_per_pair(msfm_conf, dlss_conf):
    """Return per-cross-pair (l_min_eff, l_max_eff) bin edges from the scales config.

    For each cross pair (z1, z2):
      ``l_max_eff[j] = min(l_max[z1], l_max[z2])``  (conservative: use the tighter cut)
      ``l_min_eff[j] = max(l_min[z1], l_min[z2])``  (conservative: start where both are valid)

    These are used as per-pair bin edges in ``ClsBinningAndTransformLayer``, so the
    scale cut is baked into the binning rather than applied as a post-step.

    ``l_min`` defaults to 30 per z-bin when absent from the scales config (covers configs
    such as ``unsmoothed.yaml`` and ``8wl,40gc.yaml`` that omit the field).

    Args:
        msfm_conf (dict): Multiprobe-simulation-forward-model config.
        dlss_conf (dict): Deep-LSS training config (must contain ``scale_cuts`` key).

    Returns:
        tuple: ``(names, l_min_eff_per_pair, l_max_eff_per_pair)`` where
            - ``names``: list of str, e.g. ``["bin_0x0", "bin_0x1", …]``
            - ``l_min_eff_per_pair``: list of float, one entry per cross pair.
            - ``l_max_eff_per_pair``: list of float, one entry per cross pair.
    """
    from msfm.utils import cross_statistics

    dset_common = dlss_conf["dset"]["common"]
    with_lensing = dset_common["with_lensing"]
    with_clustering = dset_common["with_clustering"]
    n_z_lensing = len(msfm_conf["survey"]["metacal"]["z_bins"]) if with_lensing else 0
    n_z_clustering = len(msfm_conf["survey"]["maglim"]["z_bins"]) if with_clustering else 0
    with_cross_probe = dset_common.get("with_cross_probe", with_lensing and with_clustering)
    ggl_only = dset_common.get("ggl_only", False)

    _DEFAULT_L_MIN = 30

    scale_cuts = dlss_conf.get("scale_cuts", {})
    l_min_lensing = list(scale_cuts.get("lensing", {}).get("l_min", [_DEFAULT_L_MIN] * n_z_lensing))
    l_min_clustering = list(scale_cuts.get("clustering", {}).get("l_min", [_DEFAULT_L_MIN] * n_z_clustering))
    l_max_lensing = list(scale_cuts.get("lensing", {}).get("l_max", [None] * n_z_lensing))
    l_max_clustering = list(scale_cuts.get("clustering", {}).get("l_max", [None] * n_z_clustering))
    l_min_per_z = (l_min_lensing if with_lensing else []) + (l_min_clustering if with_clustering else [])
    l_max_per_z = (l_max_lensing if with_lensing else []) + (l_max_clustering if with_clustering else [])

    _, names = cross_statistics.get_cross_bin_indices(
        n_z_lensing,
        n_z_clustering,
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross_z=dset_common.get("with_cross_z", True),
        with_cross_probe=with_cross_probe,
        ggl_only=ggl_only,
    )
    n_z_cross = len(names)

    l_min_eff_per_pair = []
    l_max_eff_per_pair = []
    for name in names:
        z1_str, z2_str = name.split("_", 1)[1].split("x")
        z1, z2 = int(z1_str), int(z2_str)
        lmin1 = l_min_per_z[z1] if z1 < len(l_min_per_z) else _DEFAULT_L_MIN
        lmin2 = l_min_per_z[z2] if z2 < len(l_min_per_z) else _DEFAULT_L_MIN
        lmax1 = l_max_per_z[z1] if z1 < len(l_max_per_z) else None
        lmax2 = l_max_per_z[z2] if z2 < len(l_max_per_z) else None
        l_min_eff_per_pair.append(max(lmin1, lmin2))
        if lmax1 is None and lmax2 is None:
            raise ValueError(f"No l_max defined for pair {name} — add l_max to the scales config.")
        l_max_eff_per_pair.append(min(v for v in (lmax1, lmax2) if v is not None))

    LOGGER.warning(
        f"get_cls_bounds_per_pair: n_z_cross={n_z_cross}, "
        f"l_min_eff={l_min_eff_per_pair}, l_max_eff={l_max_eff_per_pair}"
    )
    return names, l_min_eff_per_pair, l_max_eff_per_pair


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
