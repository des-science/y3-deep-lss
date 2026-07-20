# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Small pure helpers shared by the two training drivers (apps/run_training.py and
apps/run_cls_training+evaluation.py) so the two apps compute the VMIM-head theta
standardization and the cosmology-only validation metric identically.

Kept deliberately narrow: only logic that is byte-identical across the two apps lives here.
The optimizer / LR-schedule setup is NOT shared (the map app routes through
optimization.get_optimizer with a nested config schema and cosine/warmup schedules, while the
Cls app is flat-config with plateau/constant schedules and ReduceLROnPlateau), and the
mutual_info_kwargs assembly is left per-app (the two build it with different structure/defaults).
"""

import numpy as np


def theta_standardization_from_samples(samples):
    """Return (shift, scale) = per-parameter mean/std of physical training labels, float32.

    The VMIM head standardizes theta (z = (theta - shift) / scale); the MI-bound optimum is
    affine-invariant, so this is pure conditioning. `samples` is a ``(N, n_params)`` array (or
    anything reshapeable to it). Used by the Cls app, which has a gathered label table.
    """
    s = np.asarray(samples, dtype=np.float32)
    s = s.reshape(-1, s.shape[-1])
    return s.mean(axis=0), s.std(axis=0)


def theta_standardization_from_prior_intervals(prior_intervals):
    """Return (shift, scale) from analytic uniform-prior stats: mean and width/sqrt(12), float32.

    `prior_intervals` is a ``(n_params, 2)`` array of ``[low, high]`` bounds. Used by the map app,
    which has no gathered label table (the grid is Sobol-sampled from these uniform priors).
    """
    iv = np.asarray(prior_intervals)
    shift = iv.mean(axis=1).astype("float32")
    scale = ((iv[:, 1] - iv[:, 0]) / (12.0**0.5)).astype("float32")
    return shift, scale


def cosmo_param_indices(params, cosmo_names):
    """Indices of the cosmological parameters within `params`.

    The validation nrmse is averaged over only these (the wide IA-nuisance priors otherwise
    dominate an unweighted physical-units error). `cosmo_names` is msfm_conf["analysis"]["params"]["cosmo"].
    """
    return [i for i, p in enumerate(params) if p in cosmo_names]
