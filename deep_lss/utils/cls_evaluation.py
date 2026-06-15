# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""Evaluation helpers for the Cls (power-spectrum) training pipeline.

Mirror of the map-level equivalents in evaluation.py, but operating on binned
power spectra rather than pixel maps.
"""

import os
from functools import partial

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from deep_lss.utils import evaluation
from msi.utils import preprocessing

_OBS_PREPROC_FNS = {
    "hard":             preprocessing.get_preprocessed_cl_observation_hard_cut,
    "hard_conservative": partial(preprocessing.get_preprocessed_cl_observation_hard_cut, n_extra_bins=1),
    "none":             preprocessing.get_preprocessed_cl_observation_hard_cut,
    "soft_pruned":      partial(preprocessing.get_preprocessed_cl_observation, scale_cut="soft_pruned"),
    "soft":             preprocessing.get_preprocessed_cl_observation,
}


def _obs_preprocessing_fn(scale_cut, cls_n_bins=None):
    if scale_cut == "hard_rebinned":
        from deep_lss.utils.cls_preprocessing import preprocess_obs_hard_rebinned
        return partial(preprocess_obs_hard_rebinned, cls_n_bins=cls_n_bins)
    return _OBS_PREPROC_FNS[scale_cut]


def save_loss_curve(pred_dir, pred_file, train_steps, train_losses, vali_steps, vali_losses, log_every, vali_mse=None):
    train_steps = np.array(train_steps)
    train_losses = np.array(train_losses)
    vali_steps = np.array(vali_steps)
    vali_losses = np.array(vali_losses)
    vali_mse = np.array(vali_mse) if vali_mse else np.array([])

    with h5py.File(pred_file, "a") as f:
        for key, arr in [
            ("loss/train_steps", train_steps),
            ("loss/train_losses", train_losses),
            ("loss/vali_steps", vali_steps),
            ("loss/vali_losses", vali_losses),
            ("loss/vali_mse", vali_mse),
        ]:
            if key in f:
                del f[key]
            f.create_dataset(key, data=arr)

    has_vali = len(vali_steps) > 0
    has_mse = len(vali_mse) > 0
    n_panels = 1 + int(has_mse)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    skip = max(1, 1000 // log_every)
    ax = axes[0]
    ax.plot(train_steps[skip:], train_losses[skip:], lw=0.8, label="train")
    if has_vali:
        ax.plot(vali_steps, vali_losses, lw=1.5, marker="o", ms=3, label="vali")
    ax.set_xlabel("step")
    ax.set_ylabel("MI loss")
    ax.legend()

    if has_mse:
        ax2 = axes[1]
        ax2.plot(vali_steps, vali_mse, lw=1.5, marker="o", ms=3, color="C2", label="vali MSE")
        ax2.set_xlabel("step")
        ax2.set_ylabel("posterior mean MSE")
        ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(pred_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)
    print(f"Saved loss curve to {pred_dir}/loss_curve.png")


def evaluate_mock_cls(
    label,
    model,
    pred_file,
    data_dir,
    msfm_conf,
    dlss_conf,
    params,
    batch_size,
    with_lensing,
    with_clustering,
    with_cross_z,
    with_cross_probe,
    ggl_only,
    apply_log=True,
    ell_weighting=None,
    scale_cut="hard",
    cls_n_bins=None,
):
    from msfm.utils import parameters as msfm_params

    obs_file = os.path.join(data_dir, "obs", f"{label}_obs_maps.h5")
    out_label = label
    if not os.path.exists(obs_file):
        print(f"WARNING: mock file not found: {obs_file}, skipping")
        return
    print(f"Evaluating {out_label}...")
    with h5py.File(obs_file, "r") as f_in:
        obs_cls_raw = f_in["obs/cls_raw"][:]
    print(f"obs_cls_raw.shape = {obs_cls_raw.shape}")

    obs_cl = _obs_preprocessing_fn(scale_cut, cls_n_bins=cls_n_bins)(
        obs_cl=obs_cls_raw,
        msfm_conf=msfm_conf,
        dlss_conf=dlss_conf,
        base_dir=data_dir,
        nest_in=False,
        with_lensing=with_lensing,
        with_clustering=with_clustering,
        with_cross_z=with_cross_z,
        with_cross_probe=with_cross_probe,
        ggl_only=ggl_only,
        apply_log=apply_log,
        standardize=False,
        ell_weighting=ell_weighting,
        make_plot=False,
    )
    obs_cl = np.squeeze(obs_cl)

    fidu_preds = np.concatenate(
        [
            model(tf.constant(obs_cl[i : i + batch_size], dtype=tf.float32), training=False).numpy()
            for i in range(0, len(obs_cl), batch_size)
        ],
        axis=0,
    )

    fiducial_cosmo = msfm_params.get_fiducials(params, msfm_conf)

    evaluation.append_obs_to_file(pred_file, f"obs/preds/{out_label}_stack", fidu_preds)
    evaluation.append_obs_to_file(pred_file, f"obs/preds/{out_label}_mean", np.mean(fidu_preds, axis=0))
    evaluation.append_obs_to_file(pred_file, f"obs/cosmos/{out_label}", fiducial_cosmo)
    print(f"Saved {out_label} ({len(fidu_preds)} realizations) to {pred_file}")


def evaluate_des_y3(
    model,
    msfm_conf,
    dlss_conf,
    data_dir,
    pred_file,
    with_lensing,
    with_clustering,
    with_cross_z,
    with_cross_probe,
    ggl_only=False,
    apply_log=True,
    ell_weighting=None,
    scale_cut="hard",
    cls_n_bins=None,
):
    from msfm.utils import catalog

    use_wl = with_lensing or with_cross_probe
    use_gc = with_clustering or with_cross_probe

    # Build both maps regardless of probe selection: preprocess_obs_hard_rebinned
    # rebins the full (lensing+clustering) set of pairs and applies probe
    # selection afterwards, so obs_cl must cover all pairs just like the grid cache.
    print("Building DES Y3 maps from catalogs...")
    wl_gamma_map, _ = catalog.build_metacal_map_from_cat(msfm_conf)
    gc_count_map = catalog.build_maglim_map_from_cat(msfm_conf)

    lensing_variants = [(wl_gamma_map, "")]
    if use_wl:
        wl_gamma_map_no_psi_rot, _ = catalog.build_metacal_map_from_cat(
            msfm_conf, apply_shear_rotation=False, debug=False
        )
        lensing_variants.append((wl_gamma_map_no_psi_rot, "_no_psi_rot"))

    sys_variants = [(True, "DESy3")]
    if use_gc:
        sys_variants.append((False, "DESy3_no_sys"))

    print("Computing DES Y3 binned Cls...")
    for wl_map, rot_suffix in lensing_variants:
        for apply_sys, sys_label in sys_variants:
            des_cl = _obs_preprocessing_fn(scale_cut, cls_n_bins=cls_n_bins)(
                wl_gamma_map=wl_map,
                gc_count_map=gc_count_map,
                msfm_conf=msfm_conf,
                dlss_conf=dlss_conf,
                base_dir=data_dir,
                nest_in=False,
                with_lensing=with_lensing,
                with_clustering=with_clustering,
                with_cross_z=with_cross_z,
                with_cross_probe=with_cross_probe,
                ggl_only=ggl_only,
                apply_log=apply_log,
                standardize=False,
                ell_weighting=ell_weighting,
                make_plot=False,
                apply_maglim_sys_map=apply_sys,
            )

            des_pred = model(tf.constant(des_cl, dtype=tf.float32), training=False).numpy()
            evaluation.append_obs_to_file(pred_file, f"obs/preds/{sys_label}{rot_suffix}", des_pred)
