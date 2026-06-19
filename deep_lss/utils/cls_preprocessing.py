# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Cache-based preprocessing for the hard_rebinned scale cut.

Loads raw per-ell Cls from HDF5, rebins to cls_n_bins bins per pair using
per-pair sqrt-spaced bin edges (same logic as ClsBinningAndTransformLayer),
and caches the result to disk.  Probe selection is applied at load time so
a single cache file serves all probes for a given scales config.
"""

import os

import h5py
import numpy as np
import tensorflow as tf

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)

_DEFAULT_L_MIN = 30


def _build_bin_weights_all_pairs(n_ell, n_bins, n_z_lensing, n_z_clustering, l_min_z, l_max_z):
    """Build (n_ell, n_bins, n_total_pairs) averaging matrix for ALL pairs.

    Mirrors ClsBinningAndTransformLayer.__init__ but covers all (i≤j) pairs so
    a single cache serves every probe configuration.

    Args:
        n_ell: Number of ell values (= 3 * n_side).
        n_bins: Number of bins per pair.
        n_z_lensing: Number of lensing tomographic bins.
        n_z_clustering: Number of clustering tomographic bins.
        l_min_z: Per-z-bin l_min (length n_z_lensing + n_z_clustering).
        l_max_z: Per-z-bin l_max (length n_z_lensing + n_z_clustering).
    """
    from msfm.utils.power_spectra import get_cl_bins

    n_z = n_z_lensing + n_z_clustering
    n_total_pairs = n_z * (n_z + 1) // 2

    ells = np.arange(n_ell, dtype=np.float64)
    W = np.zeros((n_ell, n_bins, n_total_pairs), dtype=np.float32)

    probe_label = ["wl"] * n_z_lensing + ["gc"] * n_z_clustering
    rows = []  # collected for the summary table
    k = 0
    for i in range(n_z):
        for j in range(n_z):
            if i <= j:
                lmin_pair = max(l_min_z[i], l_min_z[j])
                lmax_pair = min(l_max_z[i], l_max_z[j])
                bin_edges = get_cl_bins(lmin_pair, lmax_pair, n_bins + 1)
                ells_per_bin = []
                for b in range(n_bins):
                    in_bin = (ells >= bin_edges[b]) & (ells < bin_edges[b + 1])
                    n_in = int(in_bin.sum())
                    ells_per_bin.append(n_in)
                    if n_in > 0:
                        W[in_bin, b, k] = 1.0 / n_in
                n_used = int((ells >= lmin_pair).sum() - (ells > lmax_pair).sum())
                rows.append((k, i, j, probe_label[i], probe_label[j], lmin_pair, lmax_pair, n_used, ells_per_bin))
                k += 1

    # Log summary table.
    type_w = max(len(f"{r[3]}×{r[4]}") for r in rows)
    header = f"  {'pair':>4}  {'zi':>2}  {'zj':>2}  {'type':<{type_w}}  {'l_min':>5}  {'l_max':>5}  {'n_ells':>6}  ells/bin"
    LOGGER.info(f"Bin weights (n_bins={n_bins}, n_ell={n_ell}, n_pairs={n_total_pairs}):")
    LOGGER.info(header)
    empty_bins = []
    for k, i, j, pi, pj, lmin_p, lmax_p, n_used, epb in rows:
        epb_str = "[" + " ".join(f"{v:2d}" for v in epb) + "]"
        n_empty = epb.count(0)
        flag = "  *** empty bins" if n_empty > 0 else ""
        LOGGER.info(f"  {k:4d}  {i:2d}  {j:2d}  {pi}×{pj:<{type_w - len(pi) - 1}}  {lmin_p:5.0f}  {lmax_p:5.0f}  {n_used:6d}  {epb_str}{flag}")
        if n_empty > 0:
            empty_bins.append((k, i, j, n_empty))
    if empty_bins:
        LOGGER.warning(f"{len(empty_bins)} pairs have empty bins — check l_min/l_max config: {[(k,i,j) for k,i,j,_ in empty_bins]}")
    return W


def _cache_path(data_dir, cls_n_bins, scales_name):
    """Return the path for a given (cls_n_bins, scales_name) combination.

    Example: {data_dir}/cls/rebinned_nb16_8wl,32gc.h5
    """
    return os.path.join(data_dir, "cls", f"rebinned_nb{cls_n_bins}_{scales_name}.h5")


def build_rebinned_cls_cache(data_dir, msfm_conf, dlss_conf, cls_n_bins, scales_name):
    """Rebin raw per-ell Cls to cls_n_bins bins per pair and cache to disk.

    The cache covers ALL (i≤j) pairs so it can be shared across probes.
    If the cache file already exists the function returns immediately.

    Args:
        data_dir: Base data directory (parent of cls/).
        msfm_conf: msfm config dict (already loaded).
        dlss_conf: dlss config dict (already loaded, must contain scale_cuts).
        cls_n_bins: Number of bins per pair.
        scales_name: Stem of the scales config filename (e.g. "8wl,32gc").
    """
    from msfm.utils import files
    from msi.utils import input_output

    cache_file = _cache_path(data_dir, cls_n_bins, scales_name)
    if os.path.exists(cache_file):
        LOGGER.info(f"Rebinned Cls cache already exists: {cache_file}")
        return

    LOGGER.warning(f"Cache not found at {cache_file} — building from raw Cls (this takes a while).")

    msfm_conf = files.load_config(msfm_conf)

    n_z_lensing = len(msfm_conf["survey"]["metacal"]["z_bins"])
    n_z_clustering = len(msfm_conf["survey"]["maglim"]["z_bins"])
    n_side = msfm_conf["analysis"]["n_side"]
    n_ell = 3 * n_side

    scale_cuts = dlss_conf.get("scale_cuts", {})
    l_min_lensing = list(scale_cuts.get("lensing", {}).get("l_min", [_DEFAULT_L_MIN] * n_z_lensing))
    l_min_clustering = list(scale_cuts.get("clustering", {}).get("l_min", [_DEFAULT_L_MIN] * n_z_clustering))
    l_max_lensing = list(scale_cuts.get("lensing", {}).get("l_max", []))
    l_max_clustering = list(scale_cuts.get("clustering", {}).get("l_max", []))
    l_min_z = l_min_lensing + l_min_clustering
    l_max_z = l_max_lensing + l_max_clustering

    if len(l_max_z) != n_z_lensing + n_z_clustering:
        raise ValueError(
            f"Expected {n_z_lensing + n_z_clustering} l_max values from scale_cuts, "
            f"got {len(l_max_z)}. Add l_max entries for all z-bins."
        )

    LOGGER.warning(f"n_ell={n_ell}, n_z_lensing={n_z_lensing}, n_z_clustering={n_z_clustering}")
    LOGGER.warning(f"l_min_z={l_min_z}")
    LOGGER.warning(f"l_max_z={l_max_z}")

    W = _build_bin_weights_all_pairs(n_ell, cls_n_bins, n_z_lensing, n_z_clustering, l_min_z, l_max_z)

    LOGGER.warning("Loading raw Cls from HDF5 (grid)...")
    file_dict = input_output.load_human_summaries(
        data_dir,
        "cls",
        return_raw_cls=True,
        return_fiducial=False,
        return_grid=True,
    )

    # grid/cls/raw shape: (n_cosmos, n_examples, n_ell, n_total_pairs)
    raw_grid = file_dict["grid/cls/raw"].astype(np.float32)

    LOGGER.warning(f"Raw grid Cls shape: {raw_grid.shape}")

    # Apply binning: (..., n_ell, n_total_pairs) x (n_ell, n_bins, n_total_pairs) -> (..., n_bins, n_total_pairs)
    LOGGER.warning("Applying rebinning (einsum)...")
    grid_binned = np.einsum("...lc,lkc->...kc", raw_grid, W)

    LOGGER.warning(f"Binned grid shape: {grid_binned.shape}")

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    LOGGER.warning(f"Saving cache to {cache_file} ...")
    with h5py.File(cache_file, "w") as f:
        f.create_dataset("grid/cls", data=grid_binned, compression="lzf")
        f.attrs["cls_n_bins"] = cls_n_bins
        f.attrs["scales_name"] = scales_name
    LOGGER.warning(f"Cache saved: {cache_file}")


def _parse_cls_indices(spec, unique_ids, is_eval):
    """Map a signal_indices or noise_indices spec to actual index values.

    None        → all unique_ids
    float 0<x<1 → train: unique_ids[:split], eval: unique_ids[split:]
    list/array  → used as-is
    """
    if spec is None:
        return unique_ids
    elif isinstance(spec, float) and 0.0 < spec < 1.0:
        split = int(spec * len(unique_ids))
        return unique_ids[split:] if is_eval else unique_ids[:split]
    else:
        return np.asarray(spec)


def get_rebinned_cls_dsets(
    data_dir,
    msfm_conf,
    dlss_conf,
    params,
    cls_n_bins,
    scales_name,
    signal_indices=0.8,
    noise_indices=None,
    with_lensing=True,
    with_clustering=True,
    with_cross_z=True,
    with_cross_probe=None,
    ggl_only=False,
    batch_size=1024,
    shuffle_buffer="full",
    prefetch=3,
    num_parallel_calls=tf.data.AUTOTUNE,
    float_type=np.float32,
    seed=None,
    return_pair_ids=False,
):
    """Load rebinned Cls from cache (building it if needed) and return TF datasets.

    Returns (cl_dset_train, cl_dset_test, out_dict) matching the interface of
    dataset.get_binned_power_spectra_dset_for_scale_cut.

    Args:
        data_dir: Base data directory (parent of cls/).
        msfm_conf: msfm config dict (already loaded).
        dlss_conf: dlss config dict (already loaded).
        params: List of cosmological parameter names for the train split.
        cls_n_bins: Number of bins per pair.
        scales_name: Stem of the scales config filename (e.g. "8wl,32gc").
        signal_indices: Fraction or list for train/eval split on cosmologies.
        noise_indices: Fraction or list for train/eval split on noise realizations.
        with_lensing, with_clustering, with_cross_z, with_cross_probe, ggl_only:
            Probe selection flags (same as existing pipeline).
        batch_size: TF dataset batch size.
        seed: Optional int seed for dset_train.shuffle(); None means unseeded (legacy behavior).
        return_pair_ids: If True, dset_train yields (cl, cosmo, i_sobol, i_signal) 4-tuples so the
            VICReg invariance term can identify positive pairs (same i_sobol/i_signal, different noise)
            within a batch. dset_test always stays a (cl, cosmo) 2-tuple. Defaults to False.
    """
    from msfm.utils import files
    from msfm.utils import cross_statistics
    from msi.utils import input_output

    msfm_conf = files.load_config(msfm_conf)

    n_z_lensing = len(msfm_conf["survey"]["metacal"]["z_bins"])
    n_z_clustering = len(msfm_conf["survey"]["maglim"]["z_bins"])

    # Ensure cache exists.
    build_rebinned_cls_cache(data_dir, msfm_conf, dlss_conf, cls_n_bins, scales_name)

    # Load binned Cls from cache: (n_cosmos, n_examples, n_cls_bins, n_total_pairs)
    cache_file = _cache_path(data_dir, cls_n_bins, scales_name)
    LOGGER.warning(f"Loading rebinned Cls from cache: {cache_file}")
    with h5py.File(cache_file, "r") as f:
        grid_cls_all = f["grid/cls"][:]  # (n_cosmos, n_examples, n_bins, n_total_pairs)

    # Apply probe selection.
    bin_indices, _ = cross_statistics.get_cross_bin_indices(
        n_z_lensing=n_z_lensing,
        n_z_clustering=n_z_clustering,
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross_z=with_cross_z,
        with_cross_probe=with_cross_probe,
        ggl_only=ggl_only,
    )
    LOGGER.info(f"Probe selection: {len(bin_indices)} pairs out of {grid_cls_all.shape[-1]} total")

    # Select pairs and flatten bins×pairs → flat vector.
    # (n_cosmos, n_examples, n_bins, n_total_pairs) -> (n_cosmos, n_examples, n_bins, n_selected)
    grid_cls = grid_cls_all[:, :, :, bin_indices]
    # (n_cosmos, n_examples, n_bins * n_selected)
    grid_cls = grid_cls.reshape(grid_cls.shape[0], grid_cls.shape[1], -1).astype(float_type)

    LOGGER.info(f"grid_cls after probe selection: {grid_cls.shape}")

    # Load cosmological parameters and indices from the original grid HDF5.
    # load_human_summaries also loads cls/binned here but we only use the metadata.
    meta_dict = input_output.load_human_summaries(
        data_dir,
        "cls",
        return_raw_cls=False,
        return_fiducial=False,
        return_grid=True,
    )
    grid_cosmos = meta_dict["grid/cosmo"]   # (n_cosmos, n_examples, n_params_all)
    grid_i_sobols = meta_dict["grid/i_sobol"]  # (n_cosmos, n_examples)
    grid_i_signals = meta_dict["grid/i_signal"]
    grid_i_noises = meta_dict["grid/i_noise"]

    # Filter cosmo to the requested training params.
    if params is not None:
        from msfm.utils import parameters as msfm_params
        all_params = msfm_params.get_parameters(None, msfm_conf)
        requested = msfm_params.get_parameters(params, msfm_conf)
        param_indices = [i for i, p in enumerate(all_params) if p in requested]
        grid_cosmos = grid_cosmos[..., param_indices]

    # Sort cosmologies by first example's Sobol index (same as existing pipeline).
    i_sort = np.argsort(grid_i_sobols, axis=0)[:, 0]
    grid_cls = grid_cls[i_sort]
    grid_cosmos = grid_cosmos[i_sort]
    grid_i_sobols_sorted = grid_i_sobols[i_sort]
    grid_i_signals = grid_i_signals[i_sort]
    grid_i_noises = grid_i_noises[i_sort]

    # Train/test split.
    unique_signal_ids = np.unique(grid_i_signals[0])
    unique_noise_ids = np.unique(grid_i_noises[0])

    train_signal_vals = _parse_cls_indices(signal_indices, unique_signal_ids, is_eval=False)
    eval_signal_vals = _parse_cls_indices(signal_indices, unique_signal_ids, is_eval=True)
    train_noise_vals = _parse_cls_indices(noise_indices, unique_noise_ids, is_eval=False)
    eval_noise_vals = _parse_cls_indices(noise_indices, unique_noise_ids, is_eval=True)

    ref_signals = grid_i_signals[0]
    ref_noises = grid_i_noises[0]
    train_mask = np.isin(ref_signals, train_signal_vals) & np.isin(ref_noises, train_noise_vals)
    eval_mask = np.isin(ref_signals, eval_signal_vals) & np.isin(ref_noises, eval_noise_vals)

    first_eval_idx = int(np.where(eval_mask)[0][0])
    grid_obs_i_sobol = grid_i_sobols_sorted[:, first_eval_idx]
    grid_obs_i_signal = grid_i_signals[:, first_eval_idx]
    grid_obs_i_noise = grid_i_noises[:, first_eval_idx]
    grid_obs_cls_raw = grid_cls[:, first_eval_idx, :].copy()
    grid_obs_cosmos = grid_cosmos[:, first_eval_idx, :].copy()

    grid_cls_train = grid_cls[:, train_mask, :]
    grid_cls_test = grid_cls[:, eval_mask, :]
    grid_cosmos_train = grid_cosmos[:, train_mask, :]
    grid_cosmos_test = grid_cosmos[:, eval_mask, :]

    _concat = lambda arr: np.concatenate([arr[i] for i in range(arr.shape[0])], axis=0)
    grid_cls_train = _concat(grid_cls_train)
    grid_cls_test = _concat(grid_cls_test)
    grid_cosmos_train = _concat(grid_cosmos_train)
    grid_cosmos_test = _concat(grid_cosmos_test)

    grid_i_sobol_test = _concat(grid_i_sobols_sorted[:, eval_mask])
    grid_i_signal_test = _concat(grid_i_signals[:, eval_mask])
    grid_i_noise_test = _concat(grid_i_noises[:, eval_mask])

    grid_i_sobol_train = _concat(grid_i_sobols_sorted[:, train_mask]).astype(np.int64)
    grid_i_signal_train = _concat(grid_i_signals[:, train_mask]).astype(np.int64)

    LOGGER.info(f"Train: {grid_cls_train.shape[0]} examples, Test: {grid_cls_test.shape[0]} examples")

    # Build TF datasets. Sign-log is applied as an augmentation (not baked into cache),
    # matching the pattern of the existing hard-cut pipeline.
    if shuffle_buffer == "full":
        shuffle_buffer = grid_cls_train.shape[0]

    def _sign_log(signal, label):
        signal = tf.math.sign(signal) * tf.math.log(tf.abs(signal) + 1e-10)
        return signal, label

    if return_pair_ids:
        # yield the per-sample (i_sobol, i_signal) ids alongside (cl, cosmo) so the VICReg
        # invariance term can find positive pairs within a batch; the sign-log only touches the signal.
        def _sign_log_with_ids(signal, label, i_sobol, i_signal):
            signal = tf.math.sign(signal) * tf.math.log(tf.abs(signal) + 1e-10)
            return signal, label, i_sobol, i_signal

        dset_train = (
            tf.data.Dataset.from_tensor_slices(
                (grid_cls_train, grid_cosmos_train, grid_i_sobol_train, grid_i_signal_train)
            )
            .cache()
            .shuffle(shuffle_buffer, seed=seed)
            .repeat()
            .batch(batch_size)
            .map(_sign_log_with_ids, num_parallel_calls=num_parallel_calls, deterministic=False)
            .prefetch(prefetch)
        )
    else:
        dset_train = (
            tf.data.Dataset.from_tensor_slices((grid_cls_train, grid_cosmos_train))
            .cache()
            .shuffle(shuffle_buffer, seed=seed)
            .repeat()
            .batch(batch_size)
            .map(_sign_log, num_parallel_calls=num_parallel_calls, deterministic=False)
            .prefetch(prefetch)
        )

    dset_test = (
        tf.data.Dataset.from_tensor_slices((grid_cls_test, grid_cosmos_test))
        .cache()
        .batch(batch_size)
        .map(_sign_log, num_parallel_calls=num_parallel_calls, deterministic=True)
        .prefetch(prefetch)
    )

    # Apply sign-log to the static eval arrays and obs Cls.
    def _np_sign_log(x):
        return np.sign(x) * np.log(np.abs(x) + 1e-10)

    out_dict = {
        "grid/cls_raw/train": grid_cls_train.copy(),
        "grid/cls_raw/test": grid_cls_test.copy(),
        "grid/cls/train": _np_sign_log(grid_cls_train),
        "grid/cls/test": _np_sign_log(grid_cls_test),
        "grid/cosmos/train": grid_cosmos_train.astype(float_type),
        "grid/cosmos/test": grid_cosmos_test.astype(float_type),
        "grid/i_sobol/test": grid_i_sobol_test,
        "grid/i_signal/test": grid_i_signal_test,
        "grid/i_noise/test": grid_i_noise_test,
        "grid/i_sobol/train": grid_i_sobol_train,
        "grid/i_signal/train": grid_i_signal_train,
        "noise/cls": None,
        "grid/i_sobols": grid_i_sobols,
        "ell_weights": None,
        "grid/obs/i_sobol": grid_obs_i_sobol,
        "grid/obs/i_signal": grid_obs_i_signal,
        "grid/obs/i_noise": grid_obs_i_noise,
        "grid/obs/cls": _np_sign_log(grid_obs_cls_raw),
        "grid/obs/cosmos": grid_obs_cosmos.astype(float_type),
    }

    return dset_train, dset_test, out_dict


def preprocess_obs_hard_rebinned(
    obs_cl=None,
    wl_gamma_map=None,
    gc_count_map=None,
    msfm_conf=None,
    dlss_conf=None,
    cls_n_bins=16,
    with_lensing=True,
    with_clustering=True,
    with_cross_z=True,
    with_cross_probe=None,
    ggl_only=False,
    apply_maglim_sys_map=True,
    **ignored_kwargs,
):
    """Apply rebinning + sign-log to a single observation for mock/DES evaluation.

    Mirrors the training preprocessing so model inputs are consistent.

    Args:
        obs_cl: Raw per-ell Cls, shape (n_ell, n_total_pairs) or (n_obs, n_ell, n_total_pairs).
                If None, wl_gamma_map / gc_count_map must be provided instead.
        wl_gamma_map, gc_count_map: HEALPix maps; per-ell Cls are computed from them.
        msfm_conf, dlss_conf: Config dicts (already loaded or file paths).
        cls_n_bins: Must match the training configuration.
    """
    from msfm.utils import files, cross_statistics
    from msfm.utils.power_spectra import get_cl_bins

    msfm_conf = files.load_config(msfm_conf)

    n_z_lensing = len(msfm_conf["survey"]["metacal"]["z_bins"])
    n_z_clustering = len(msfm_conf["survey"]["maglim"]["z_bins"])
    n_side = msfm_conf["analysis"]["n_side"]
    n_ell = 3 * n_side

    if obs_cl is None:
        from msfm.utils import observation
        _, obs_cl, _ = observation.forward_model_observation_map(
            wl_gamma_map=wl_gamma_map,
            gc_count_map=gc_count_map,
            conf=msfm_conf,
            apply_norm=False,
            with_padding=True,
            nest_in=False,
            apply_maglim_sys_map=apply_maglim_sys_map,
        )

    obs_cl = np.asarray(obs_cl, dtype=np.float32)

    scale_cuts = dlss_conf.get("scale_cuts", {})
    l_min_lensing = list(scale_cuts.get("lensing", {}).get("l_min", [_DEFAULT_L_MIN] * n_z_lensing))
    l_min_clustering = list(scale_cuts.get("clustering", {}).get("l_min", [_DEFAULT_L_MIN] * n_z_clustering))
    l_max_lensing = list(scale_cuts.get("lensing", {}).get("l_max", []))
    l_max_clustering = list(scale_cuts.get("clustering", {}).get("l_max", []))
    l_min_z = l_min_lensing + l_min_clustering
    l_max_z = l_max_lensing + l_max_clustering

    W = _build_bin_weights_all_pairs(n_ell, cls_n_bins, n_z_lensing, n_z_clustering, l_min_z, l_max_z)

    if obs_cl.shape[-1] != W.shape[-1]:
        raise ValueError(
            f"obs_cl has {obs_cl.shape[-1]} pair-columns (shape={obs_cl.shape}) "
            f"but W expects {W.shape[-1]} (shape={W.shape}); "
            f"wl_gamma_map is None: {wl_gamma_map is None}, "
            f"gc_count_map is None: {gc_count_map is None}"
        )

    # obs_cl shape: (..., n_ell, n_total_pairs)
    obs_binned = np.einsum("...lc,lkc->...kc", obs_cl, W)  # (..., n_bins, n_total_pairs)

    # Probe selection.
    bin_indices, _ = cross_statistics.get_cross_bin_indices(
        n_z_lensing=n_z_lensing,
        n_z_clustering=n_z_clustering,
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross_z=with_cross_z,
        with_cross_probe=with_cross_probe,
        ggl_only=ggl_only,
    )
    obs_selected = obs_binned[..., :, bin_indices]  # (..., n_bins, n_selected_pairs)
    obs_flat = obs_selected.reshape(obs_selected.shape[:-2] + (-1,))  # (..., n_bins*n_selected)

    result = np.sign(obs_flat) * np.log(np.abs(obs_flat) + 1e-10)
    return np.atleast_2d(result)
