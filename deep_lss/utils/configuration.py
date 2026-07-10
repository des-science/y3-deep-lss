import os
import numpy as np
import yaml

from msfm.utils import input_output, logger, files

LOGGER = logger.get_logger(__file__)

# the loss-block keys that the legacy maps configs.yaml stream folded into dlss_conf
_LEGACY_LOSS_KEYS = ("loss_function", "delta_loss", "likelihood_loss", "mutual_info_loss")


def read_split_configs(probes_config, scales_config=None):
    """Build the dlss_conf dict from the orthogonal split configs.

    The split configs define disjoint top-level namespaces: probes provides ``dset.*`` and
    scales provides ``scale_cuts.*``. Because the top-level keys do not overlap, a shallow
    ``dict.update`` is the correct merge. The loss config is kept separate (loaded as its own
    ``loss_conf``) in both apps and is intentionally not merged here.

    Args:
        probes_config (str): path to the probes config (required).
        scales_config (str, optional): path to the scales config. Merged in if given.

    Returns:
        dict: the merged dlss_conf (``dset`` + ``scale_cuts``).
    """
    dlss_conf = input_output.read_yaml(probes_config)
    if scales_config:
        dlss_conf.update(input_output.read_yaml(scales_config))
    LOGGER.info("Loaded the split configs")
    return dlss_conf


def load_run_configs(path):
    """Load a saved run's ``configs.yaml`` into the nested layout, migrating legacy streams.

    The current format is a single nested mapping with keys
    ``{net|mlp, dlss, loss, data, msfm, run}``. Older maps runs saved a positional 3-document
    stream ``[net (+run), dlss (loss merged in), msfm]`` with the train/test split living inside
    ``net["dset"]`` rather than in a separate ``data`` block. This loader normalizes that legacy
    shape so the restore/eval code only ever sees the nested layout. (Legacy 4-document cls
    streams are never reloaded, so they are not handled here.)

    Args:
        path (str): path to the saved ``configs.yaml``.

    Returns:
        dict: the nested config mapping.
    """
    with open(path, "r") as f:
        docs = list(yaml.safe_load_all(f))

    if len(docs) == 1 and isinstance(docs[0], dict) and "dlss" in docs[0]:
        return docs[0]

    # legacy 3-document maps stream
    net_conf, dlss_conf, msfm_conf = docs
    run_conf = net_conf.pop("run", {})
    loss_conf = {k: dlss_conf.pop(k) for k in _LEGACY_LOSS_KEYS if k in dlss_conf}
    eval_common = net_conf["dset"]["eval"]["common"]
    data_conf = {
        "signal_indices": eval_common.get("signal_indices", 0.8),
        "noise_indices": eval_common.get("noise_indices", None),
    }
    LOGGER.warning("Migrated a legacy 3-document configs.yaml to the nested layout")
    return {
        "net": net_conf,
        "dlss": dlss_conf,
        "loss": loss_conf,
        "data": data_conf,
        "msfm": msfm_conf,
        "run": run_conf,
    }


def load_deep_lss_config(conf):
    """Return a dlss_conf dict from either an already-loaded dict or a path.

    Compatibility shim for callers (e.g. ``msi`` Cls preprocessing) that accept a dlss_conf
    as ``dset`` + ``scale_cuts``. After the config split, the merged dict is produced by
    ``read_split_configs`` or pulled from a saved run's ``configs.yaml`` via
    ``load_run_configs(...)["dlss"]``; both yield a dict, which is passed through unchanged.
    A string/path is read as a single YAML document (legacy monolithic dlss config).

    Args:
        conf (dict | str): an already-merged dlss_conf dict, or a path to a YAML config.

    Returns:
        dict: the dlss_conf mapping.
    """
    if isinstance(conf, dict):
        return conf
    return input_output.read_yaml(conf)


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


def resolve_probe_smooth_nsides(net_conf, dlss_conf, n_side):
    """Resolve the per-probe smoothing nside from ``network.smooth_nside``.

    ``smooth_nside`` may be absent/None (native n_side for all probes), a scalar (one nside for
    all probes, the legacy form), or a mapping ``{probe: nside}`` where a missing or None entry
    means the native n_side. Values are clamped to n_side. When all active probes resolve to the
    same nside, the downstream code lowers to the single-kernel path (with pipeline downsampling),
    so the mapping form is pure config sugar in that case; only genuinely mixed nsides use the
    per-probe smoothing path.

    Args:
        net_conf (dict): Network architecture config.
        dlss_conf (dict): Deep-LSS training config (``dset.common`` probe flags).
        n_side (int): Native map resolution from the msfm config.

    Returns:
        dict: ``{probe: nside}`` for the active map probes, in channel order.
    """
    dset_common = dlss_conf["dset"]["common"]
    probes = [
        probe
        for probe, flag in [("lensing", "with_lensing"), ("clustering", "with_clustering")]
        if dset_common[flag]
    ]
    smooth_nside = net_conf["network"].get("smooth_nside", None)
    if not probes:
        if dset_common.get("with_cross", False):
            # cross-probe maps only: single-kernel path with the scalar smooth_nside
            if isinstance(smooth_nside, dict):
                raise ValueError("Per-probe smooth_nside requires the with_lensing/with_clustering map probes")
            return {"cross": min(smooth_nside or n_side, n_side)}
        raise ValueError("At least one of with_lensing, with_clustering, or with_cross must be True")

    if isinstance(smooth_nside, dict):
        probe_nsides = {probe: min(smooth_nside.get(probe) or n_side, n_side) for probe in probes}
        if dset_common.get("with_cross", False) and len(set(probe_nsides.values())) > 1:
            raise ValueError(
                "Per-probe smooth_nside with mixed nsides is not supported together with the "
                "cross-probe map channels (with_cross), since those mix both probes"
            )
    else:
        probe_nsides = {probe: min(smooth_nside or n_side, n_side) for probe in probes}
    return probe_nsides


def resolve_smooth_nside(net_conf, dlss_conf, msfm_conf):
    """Resolve the network-input geometry implied by ``network.smooth_nside``.

    The network (and the data pipeline downsampling) runs at the finest per-probe smoothing nside;
    probes below it are handled inside the smoothing layer (``PerProbeSmoothing``).

    Args:
        net_conf (dict): Network architecture config.
        dlss_conf (dict): Deep-LSS training config.
        msfm_conf (dict): Multiprobe-simulation-forward-model config.

    Returns:
        tuple: ``(smooth_nside, smooth_indices, parent_output_idx)`` where ``parent_output_idx``
            is the pipeline downsampling map (None when the network runs at the native n_side).
    """
    n_side = msfm_conf["analysis"]["n_side"]
    data_vec_pix, _, _, _ = files.load_pixel_file(msfm_conf)

    probe_nsides = resolve_probe_smooth_nsides(net_conf, dlss_conf, n_side)
    smooth_nside = max(probe_nsides.values())
    if smooth_nside < n_side:
        smooth_indices, parent_output_idx = get_smooth_nside_indices(data_vec_pix, n_side, smooth_nside)
        LOGGER.info(f"Using smooth_nside={smooth_nside}: {len(data_vec_pix)} → {len(smooth_indices)} pixels")
    else:
        smooth_indices = data_vec_pix
        parent_output_idx = None
    return smooth_nside, smooth_indices, parent_output_idx


def _downsample_mask(mask, parent_output_idx, n_pix_out):
    """Downsample a per-channel (n_pix, n_channels) mask by per-parent averaging."""
    counts = np.bincount(parent_output_idx, minlength=n_pix_out).astype(np.float32)
    return np.stack(
        [
            np.bincount(parent_output_idx, weights=mask[:, c].astype(np.float32), minlength=n_pix_out) / counts
            for c in range(mask.shape[1])
        ],
        axis=1,
    ).astype(np.float32)


def _get_effective_local_batch_size(loss_function, net_conf, mode, n_params):
    """The largest map batch the smoothing layer has to handle (sets the sparse matmul splits)."""
    if mode == "training":
        if loss_function == "delta":
            local_batch_size = net_conf["dset"][mode]["fiducial"]["local_batch_size"]
            return local_batch_size * (2 * n_params + 1)
        return net_conf["dset"][mode]["grid"]["local_batch_size"]
    if loss_function == "delta":
        return net_conf["dset"]["eval"]["fiducial"]["local_batch_size"]
    return net_conf["dset"]["eval"]["grid"]["local_batch_size"]


def _get_split_probe_specs(
    loss_function, msfm_conf, dlss_conf, net_conf, probe_nsides, n_side, data_vec_pix, mask_dict, dir_base, mode
):
    """Build the per-probe smoothing spec consumed by ``deep_lss.nets.layers.maps.smoothing.PerProbeSmoothing``.

    One ``HealpySmoothing`` kwargs dict per active probe at that probe's nside, following the same
    conventions as the single-kernel path: white noise sigma scaled by (probe_nside / n_side) and
    divided by the map normalization, masks downsampled by per-parent averaging. A probe below the
    output (finest) nside additionally carries the ``parent_output_idx`` that maps the output
    footprint to its coarse footprint, driving the in-network down/upsampling.
    """
    _PROBE_SURVEYS = {"lensing": "metacal", "clustering": "maglim"}

    output_nside = max(probe_nsides.values())
    if output_nside < n_side:
        output_indices, _ = get_smooth_nside_indices(data_vec_pix, n_side, output_nside)
    else:
        output_indices = data_vec_pix

    arcmin = dlss_conf["scale_cuts"]["arcmin"]
    n_sigma_support = dlss_conf["scale_cuts"]["n_sigma_support"]
    apply_norm = dlss_conf["dset"]["common"]["apply_norm"]
    n_params = len(dlss_conf["dset"]["training"]["params"])
    effective_local_batch_size = _get_effective_local_batch_size(loss_function, net_conf, mode, n_params)

    probe_specs = []
    for probe, probe_nside in probe_nsides.items():
        scale_conf = dlss_conf["scale_cuts"][probe]
        fwhm = list(scale_conf["theta_fwhm"])
        fwhm_base = scale_conf.get("theta_fwhm_base", None)
        mask = mask_dict[_PROBE_SURVEYS[probe]]

        white_noise_sigma = np.array(scale_conf["white_noise_sigma"], dtype=float)
        if apply_norm:
            white_noise_sigma = white_noise_sigma / np.array(msfm_conf["analysis"]["normalization"][probe])
        # scale white noise for lower nside: sigma ∝ 1/sqrt(pixel_area) ∝ nside
        white_noise_sigma = white_noise_sigma * (probe_nside / n_side)

        if probe_nside < n_side:
            probe_indices, parent_from_full = get_smooth_nside_indices(data_vec_pix, n_side, probe_nside)
            mask = _downsample_mask(mask, parent_from_full, len(probe_indices))
        else:
            probe_indices = data_vec_pix

        if probe_nside < output_nside:
            # maps the output footprint (what the network and pipeline run at) to this probe's
            # coarse footprint, for the in-network down/upsampling around the smoothing
            probe_indices_out, parent_output_idx = get_smooth_nside_indices(
                output_indices, output_nside, probe_nside
            )
            assert np.array_equal(probe_indices_out, probe_indices)
        else:
            parent_output_idx = None

        smoothing_kwargs = {
            "nside": probe_nside,
            "indices": probe_indices,
            "nest": True,
            "mask": mask,
            "fwhm": fwhm,
            "fwhm_base": fwhm_base,
            "arcmin": arcmin,
            "n_sigma_support": n_sigma_support,
            "max_batch_size": effective_local_batch_size,
            "white_noise_sigma": white_noise_sigma,
        }
        if dir_base is not None:
            smoothing_kwargs["data_path"] = os.path.join(dir_base, "smoothing")

        LOGGER.info(
            f"Split smoothing for {probe}: nside={probe_nside} (output nside={output_nside}), "
            f"fwhm={fwhm}, fwhm_base={fwhm_base}"
        )
        probe_specs.append(
            {
                "probe": probe,
                "n_channels": len(fwhm),
                "smoothing_kwargs": smoothing_kwargs,
                "parent_output_idx": parent_output_idx,
            }
        )

    return {"split_probes": probe_specs}


def get_smoothing_kwargs(loss_function, msfm_conf, dlss_conf, net_conf, dir_base=None, mode="training"):
    """Build a dictionary of keyword arguments for the deepsphere.healpy_layers.HealpySmoothing layer.

    Args:
        loss_function (str): One of "delta", "mse", "likelihood", "mutual_info"
        msfm_conf (dict): Multiprobe-simulation-forward-model config.
        dlss_conf (dict): Network training config.
        net_conf (dict): Network architecture config.
        dir_base (str, optional): Directory to store the smoothing kernel. Defaults to None.

    Returns:
        dict: keyword arguments for deepsphere.healpy_layers.HealpySmoothing, or — when
            ``network.smooth_nside`` requests mixed per-probe nsides — a ``{"split_probes": [...]}``
            spec for ``deep_lss.nets.layers.maps.smoothing.PerProbeSmoothing``.
    """
    # msfm
    n_side = msfm_conf["analysis"]["n_side"]
    data_vec_pix, _, _, _ = files.load_pixel_file(msfm_conf)
    mask_dict = files.get_tomo_dv_masks(msfm_conf)

    # dlss
    with_lensing = dlss_conf["dset"]["common"]["with_lensing"]
    with_clustering = dlss_conf["dset"]["common"]["with_clustering"]
    with_cross = dlss_conf["dset"]["common"].get("with_cross", False)

    # per-probe smoothing nsides; mixed values take the split-kernel path (one HealpySmoothing per
    # probe at its own nside), uniform values fall through to the single-kernel path below
    probe_nsides = resolve_probe_smooth_nsides(net_conf, dlss_conf, n_side)
    if len(set(probe_nsides.values())) > 1:
        return _get_split_probe_specs(
            loss_function, msfm_conf, dlss_conf, net_conf, probe_nsides, n_side, data_vec_pix, mask_dict,
            dir_base, mode,
        )

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

    smooth_nside = max(probe_nsides.values())  # uniform across probes on this path
    if smooth_nside < n_side:
        smooth_indices, parent_output_idx = get_smooth_nside_indices(data_vec_pix, n_side, smooth_nside)
        # downsample the per-channel mask to smooth_nside using per-parent averaging
        mask_smooth = _downsample_mask(mask, parent_output_idx, len(smooth_indices))
        LOGGER.info(f"Downsampling smoothing from nside={n_side} to smooth_nside={smooth_nside}: "
                    f"{len(data_vec_pix)} → {len(smooth_indices)} pixels")
    else:
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
    lenses_before_sources = dset_common.get("lenses_before_sources", dset_common.get("ggl_only", False))

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
        lenses_before_sources=lenses_before_sources,
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
