# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2025
Author: Arne Thomsen

Power-spectrum (Cls) binning + transform layer used by the maps + Cls composite networks
(``deep_lss.nets.composite.resnet_maps_plus_cls`` and
``deep_lss.nets.composite.transformer_maps_plus_cls``).
"""

import numpy as np
import tensorflow as tf

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


class ClsBinningAndTransformLayer(tf.keras.layers.Layer):
    """Non-trainable layer that bins raw per-ell Cls with per-pair scale-cut bin edges.

    The TFRecords store per-ell Cls of shape ``(batch, n_ell, n_z_cross)`` where
    ``n_ell = 3 * n_side``.  For each cross pair ``c`` this layer uses its own
    sqrt-spaced bin edges ``[l_min_per_pair[c], l_max_per_pair[c]]`` (derived from
    the scales config) so that the scale cut is baked into the binning — no
    post-binning masking is needed.

    The output shape is ``(batch, n_bins * n_z_cross)`` — a fixed-size vector with
    exactly ``n_bins`` bins per pair — after one of two transforms selected by
    ``cls_transform``:

      * ``"asinh_per_feature"`` (default): ``asinh(x / s)`` with a per-feature (per ``(pair, ell-bin)``
        column) scale ``s`` grounded in the data. ``s`` is set once via ``set_scale()`` from
        the median ``|C_l|`` of the cached binned Cls, mirroring the Cls-only ``AsinhScaleLayer``.
      * ``"log1p_fixed"``: ``sign(x) * log1p(|x| / 1e-10)`` with a fixed knee.

    All weights are non-trainable and stored as ``tf.Variable`` so they are saved /
    restored with the model checkpoint and broadcast correctly under MirroredStrategy —
    this includes the per-feature ``cls_scale`` in the asinh mode, so the same transform
    is applied at training, evaluation and any later restore without a sidecar file.
    """

    def __init__(self, n_ell, n_bins, l_min_per_pair, l_max_per_pair, cls_transform="asinh_per_feature", **kwargs):
        """
        Args:
            n_ell (int): Number of ell values stored in the TFRecords (= 3 * n_side).
            n_bins (int): Number of bins per cross pair.
            l_min_per_pair (list[float]): Per-pair lower bin edge. Length = n_z_cross.
            l_max_per_pair (list[float]): Per-pair upper bin edge (= l_max_eff from scale cut).
                Length = n_z_cross.
            cls_transform (str): "asinh_per_feature" (per-feature asinh(x/s); call set_scale() to
                load the fitted scale) or "log1p_fixed" (fixed 1e-10 knee).
        """
        super().__init__(**kwargs)
        from msfm.utils.power_spectra import get_cl_bins

        if cls_transform not in ("log1p_fixed", "asinh_per_feature"):
            raise ValueError(f"Unknown cls_transform={cls_transform!r}")
        self.cls_transform = cls_transform

        n_z_cross = len(l_min_per_pair)
        assert len(l_max_per_pair) == n_z_cross

        ells = np.arange(n_ell, dtype=np.float64)

        # (n_ell, n_bins, n_z_cross) — per-pair averaging matrices
        W = np.zeros((n_ell, n_bins, n_z_cross), dtype=np.float32)
        for c, (lmin_c, lmax_c) in enumerate(zip(l_min_per_pair, l_max_per_pair)):
            bin_edges_c = get_cl_bins(lmin_c, lmax_c, n_bins + 1)
            for k in range(n_bins):
                in_bin = (ells >= bin_edges_c[k]) & (ells < bin_edges_c[k + 1])
                if in_bin.sum() > 0:
                    W[in_bin, k, c] = 1.0 / in_bin.sum()
        self.bin_weight = tf.Variable(W, trainable=False, name="bin_weight")

        self.n_cls_flat = n_bins * n_z_cross

        # Per-feature asinh scale. Created (initialised to ones) whenever the asinh transform is
        # selected so the checkpoint always has a matching variable to save/restore; the fitted
        # values are loaded later via set_scale() (fresh training) or restored from the checkpoint.
        if self.cls_transform == "asinh_per_feature":
            self.cls_scale = tf.Variable(np.ones(self.n_cls_flat, dtype=np.float32), trainable=False, name="cls_scale")
        else:
            self.cls_scale = None

        LOGGER.warning(
            f"ClsBinningAndTransformLayer: n_bins={n_bins}, n_z_cross={n_z_cross}, "
            f"output_dim={self.n_cls_flat}, cls_transform={self.cls_transform}"
        )
        for c, (lmin_c, lmax_c) in enumerate(zip(l_min_per_pair, l_max_per_pair)):
            LOGGER.info(f"  Cls pair {c:2d}: l_min={lmin_c}, l_max={lmax_c}")

    def set_scale(self, scale):
        """Load the fitted per-feature asinh scale (length n_cls_flat) into the stored variable."""
        if self.cls_scale is None:
            raise RuntimeError("set_scale() is only valid for cls_transform='asinh_per_feature'")
        scale = np.asarray(scale, dtype=np.float32)
        assert scale.shape == (self.n_cls_flat,), f"scale shape {scale.shape} != ({self.n_cls_flat},)"
        self.cls_scale.assign(scale)

    def call(self, cls, training=None):
        """Bin with per-pair bin edges and transform raw per-ell Cls.

        Args:
            cls: Float tensor ``(batch, n_ell, n_z_cross)``.

        Returns:
            Float tensor ``(batch, n_bins * n_z_cross)``, transformed per ``cls_transform``, cast
            back to the input dtype so it concatenates with the (mixed-precision) map branch.
        """
        # Under a mixed_bfloat16/float16 policy the raw Cls arrive downcast to the compute dtype,
        # but bin_weight / cls_scale are float32. The Cls span a huge dynamic range (~1e-11..1e-5),
        # so do the binning and transform in float32, then cast the (O(1)) output back to the input
        # dtype for the downstream LayerNorm / concat with the map features.
        in_dtype = cls.dtype
        cls = tf.cast(cls, self.bin_weight.dtype)
        # (batch, n_ell, n_z_cross) × (n_ell, n_bins, n_z_cross) → (batch, n_bins, n_z_cross)
        cls_binned = tf.einsum("blc,lkc->bkc", cls, self.bin_weight)
        cls_flat = tf.reshape(cls_binned, (tf.shape(cls_binned)[0], -1))  # (batch, n_bins*n_z_cross)
        if self.cls_transform == "asinh_per_feature":
            # Per-feature asinh(x/s): data-grounded, sign-preserving, invertible (x = s*sinh(y)).
            out = tf.math.asinh(cls_flat / self.cls_scale)
        else:
            # Signed log transform: compresses dynamic range, sign-preserving and invertible
            # (x = sign(y)*1e-10*expm1(|y|)).
            out = tf.math.sign(cls_flat) * tf.math.log1p(tf.abs(cls_flat) / 1e-10)
        return tf.cast(out, in_dtype)
