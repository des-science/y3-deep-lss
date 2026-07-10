# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2025
Author: Arne Thomsen

Empirical per-channel input normalization for the smoothed HEALPix maps.

``EmpiricalInputNormalization`` standardizes the smoothed maps per channel with a scalar
mean/std measured from training data at the start of a fresh run and frozen into the checkpoint,
then re-applies the footprint mask so masked pixels stay exactly zero. Channels of very different
physical scale (lensing shear vs clustering counts) thus enter the downstream network balanced.
``compute_input_norm_stats`` measures those per-channel statistics from a stream of training maps.

Kept independent of any particular network body (used by the transformer networks in
``transformer_networks.py``) so it can be reused.
"""

import numpy as np
import tensorflow as tf

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


class EmpiricalInputNormalization(tf.keras.layers.Layer):
    """Per-channel standardization of the smoothed input maps: ``(x - mean) * inv_std * mask``.

    Only ``mean`` and ``inv_std`` are learned-from-data state: they are measured once, at the start
    of a fresh training run, from a few hundred training maps AFTER the smoothing front-end
    (``compute_input_norm_stats``), so they describe exactly what the transformer sees under the
    active scales config. They live in non-trainable variables and therefore persist in the
    checkpoint: resumes and evaluation restore them exactly, and the DES data map is normalized with
    the identical simulation-derived constants — a fixed, invertible part of the forward model, with
    no per-sample statistics that could differ between simulations and data (or delete cosmological
    signal such as the map variance).

    - ``mean`` and ``inv_std`` are per channel only (scalars). All pixels of a channel are
      statistically identical in expectation (apart from the survey mask), so a per-pixel mean or
      std is unjustified — it would be a noisy estimate of a near-constant and would whiten out
      real depth-variation structure. Subtracting ``mean`` removes the DC of the unnormalized
      clustering count maps (mean ~ ``n_gal`` per pixel); dividing by the per-channel std
      equalizes the order-of-magnitude imbalance between probes/bins. Both are pooled over the
      active footprint pixels and over maps, so the footprint is standardized to ~unit variance.
    - ``mask`` is the ``(n_pix, n_channels)`` survey footprint, the SAME mask handed to the
      ``HealpySmoothing`` front-end. It is a constructor input, not measured or checkpointed — it
      is fixed config-derived geometry, reconstructed identically on every resume/eval exactly like
      the smoothing mask, so it stays a plain constant (out of the parameter count) rather than a
      variable. It is re-applied AFTER the affine step so out-of-footprint pixels — which the
      nonzero ``mean`` would otherwise move to ``-mean * inv_std`` — stay exactly zero, and is
      binarized to a 0/1 indicator: the smoothing already applied any fractional apodization, so
      this second multiply must only restore zeros, never weight the footprint again.
    """

    def __init__(self, n_channels, mask, **kwargs):
        # fixed float32: runs on the fp32-smoothed maps, before the cast to the body dtype
        super().__init__(dtype="float32", **kwargs)
        self.mean = tf.Variable(tf.zeros((n_channels,), dtype=tf.float32), trainable=False, name="input_norm_mean")
        self.inv_std = tf.Variable(tf.ones((n_channels,), dtype=tf.float32), trainable=False, name="input_norm_inv_std")
        # constant (not a checkpointed variable): shared config geometry, like HealpySmoothing.mask
        self.mask = None if mask is None else tf.constant(np.asarray(mask) > 0, dtype=tf.float32)

    def load_stats(self, mean, inv_std):
        self.mean.assign(mean)
        self.inv_std.assign(inv_std)

    def call(self, x):
        x = (x - self.mean) * self.inv_std
        if self.mask is not None:
            x = x * self.mask
        return x


def compute_input_norm_stats(smoothing, dset, n_batches, mask=None):
    """Measure the ``EmpiricalInputNormalization`` per-channel statistics from training maps.

    Streams ``n_batches`` batches from ``dset`` (elements with the maps first, as yielded by the
    msfm pipelines), applies the network's own smoothing front-end, and accumulates per-pixel
    first/second moments in float64. ``mask`` is the footprint the maps are standardized over (the
    input-norm layer's own mask): statistics are pooled over its active pixels AND over maps, so
    each channel's footprint is standardized to ~unit variance while the masked fraction never
    dilutes the scale. When ``mask`` is None the footprint falls back to the pixels that ever vary
    (exactly zero in every map == outside the footprint).

    Returns:
        tuple: ``(mean, inv_std)`` float32 arrays of shape ``(n_channels,)``.
    """
    n_maps = 0
    moment1 = None
    moment2 = None
    for element in dset.take(n_batches):
        maps = element[0]
        if smoothing is not None:
            maps = smoothing(maps, training=False)
        maps = np.asarray(maps, dtype=np.float64)
        if moment1 is None:
            moment1 = maps.sum(axis=0)
            moment2 = np.square(maps).sum(axis=0)
        else:
            moment1 += maps.sum(axis=0)
            moment2 += np.square(maps).sum(axis=0)
        n_maps += maps.shape[0]
    assert n_maps > 1, f"Got only {n_maps} maps from {n_batches} batches — cannot measure input normalization"

    # active footprint: the passed survey mask, else the pixels that ever vary (exactly zero in
    # every map == outside the footprint)
    active = np.asarray(mask) > 0 if mask is not None else moment2 > 0.0
    n_active = active.sum(axis=0)
    assert np.all(n_active > 0), f"Channels without any active pixels: n_active per channel = {n_active}"

    # per-channel scalars pooled over the active pixels and over maps
    count = n_maps * n_active
    mean = (moment1 * active).sum(axis=0) / count
    channel_var = np.maximum((moment2 * active).sum(axis=0) / count - np.square(mean), 0.0)
    assert np.all(channel_var > 0.0), f"Degenerate (zero-variance) channel(s): channel_var = {channel_var}"
    inv_std = 1.0 / np.sqrt(channel_var)

    LOGGER.warning(
        f"Input normalization measured from {n_maps} maps: per-channel std = "
        f"{np.array2string(np.sqrt(channel_var), precision=4)}, per-channel mean = "
        f"{np.array2string(mean, precision=4)}, "
        f"active pixel fraction = {np.array2string(n_active / active.shape[0], precision=3)}"
    )
    return mean.astype(np.float32), inv_std.astype(np.float32)
