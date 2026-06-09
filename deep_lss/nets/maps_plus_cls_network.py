# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2025
Author: Arne Thomsen

Network that concatenates DeepSphere map features with binned angular power spectra (Cls).

Architecture:
  1. HealpyGCNN processes the HEALPix maps → flatten → map_norm (LN)       (map branch)
  2. ClsBinningAndTransformLayer bins + gathers + sign-log-transforms Cls
     → cls_norm (LN) → cls_embedding MLP (Dense→LN × N)                    (Cls branch)
  3. Concatenate both branches
  4. regression_head (LN + hidden Dense layers + output)
"""

import numpy as np
import tensorflow as tf

from deepsphere import HealpyGCNN

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
    exactly ``n_bins`` bins per pair — after a ``sign(x) * log(|x| + ε)`` transform.

    All weights are non-trainable and stored as ``tf.Variable`` so they are saved /
    restored with the model checkpoint and broadcast correctly under MirroredStrategy.
    """

    def __init__(self, n_ell, n_bins, l_min_per_pair, l_max_per_pair, **kwargs):
        """
        Args:
            n_ell (int): Number of ell values stored in the TFRecords (= 3 * n_side).
            n_bins (int): Number of bins per cross pair.
            l_min_per_pair (list[float]): Per-pair lower bin edge. Length = n_z_cross.
            l_max_per_pair (list[float]): Per-pair upper bin edge (= l_max_eff from scale cut).
                Length = n_z_cross.
        """
        super().__init__(**kwargs)
        from msfm.utils.power_spectra import get_cl_bins

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

        LOGGER.warning(
            f"ClsBinningAndTransformLayer: n_bins={n_bins}, n_z_cross={n_z_cross}, "
            f"output_dim={self.n_cls_flat}"
        )
        for c, (lmin_c, lmax_c) in enumerate(zip(l_min_per_pair, l_max_per_pair)):
            LOGGER.info(f"  Cls pair {c:2d}: l_min={lmin_c}, l_max={lmax_c}")

    def call(self, cls, training=None):
        """Bin with per-pair bin edges and sign-log-transform raw per-ell Cls.

        Args:
            cls: Float tensor ``(batch, n_ell, n_z_cross)``.

        Returns:
            Float tensor ``(batch, n_bins * n_z_cross)``, sign-log-transformed.
        """
        # (batch, n_ell, n_z_cross) × (n_ell, n_bins, n_z_cross) → (batch, n_bins, n_z_cross)
        cls_binned = tf.einsum("blc,lkc->bkc", cls, self.bin_weight)
        cls_flat = tf.reshape(cls_binned, (tf.shape(cls_binned)[0], -1))  # (batch, n_bins*n_z_cross)
        # Signed log transform: compresses dynamic range, preserves sign of cross-spectra
        return tf.math.sign(cls_flat) * tf.math.log(tf.abs(cls_flat) + 1e-10)


class MapsPlusCLSNetwork(tf.keras.Model):
    """Maps + Cls combined network.

    Processes HEALPix maps with a DeepSphere HealpyGCNN, then concatenates the
    Cls branch (per-pair binned, sign-log-transformed, encoded by a small MLP)
    to the flattened GCNN output before the regression head.  Each branch is
    independently LayerNorm'd; the Cls embedding further processes the Cls
    features before fusion.
    """

    def __init__(
        self,
        conv_layers,
        cls_embedding_layers,
        regression_head_layers,
        n_side,
        tfr_n_side,
        indices,
        n_neighbors,
        max_batch_size,
        initial_Fin,
        n_cls_bins,
        l_min_per_pair,
        l_max_per_pair,
    ):
        """
        Args:
            conv_layers (list): Graph-convolution layers (ResNetLayers.get_conv_layers()).
            cls_embedding_layers (list): MLP layers that encode the Cls branch before fusion
                (get_cls_embedding_layers()).  Pass ``[]`` to skip the embedding.
            regression_head_layers (list): Dense head layers without the leading Flatten
                (ResNetLayers.get_head_layers_no_flatten()).
            n_side (int): HEALPix n_side of the input maps (after any downsampling/smoothing)
                used to build the GCNN graph.
            tfr_n_side (int): Native HEALPix n_side of the TFRecords, i.e. the simulation
                resolution. The Cls stored in the TFRecords are not downsampled, so
                ``n_ell = 3 * tfr_n_side``.
            indices (np.ndarray): 1-D array of HEALPix NEST pixel indices in the footprint.
            n_neighbors (int): Number of neighbours for the HealpyGCNN graph.
            max_batch_size (int): Pre-allocated max batch size for sparse-dense matmul splits.
            initial_Fin (int): Number of input map channels (z-bins).
            n_cls_bins (int): Number of ell bins per cross pair.
            l_min_per_pair (list[float]): Per-pair lower bin edge (from scales config).
            l_max_per_pair (list[float]): Per-pair upper bin edge = l_max_eff (from scales config).
        """
        super().__init__()

        self.gcnn = HealpyGCNN(
            nside=n_side,
            indices=indices,
            layers=conv_layers,
            n_neighbors=n_neighbors,
            max_batch_size=max_batch_size,
            initial_Fin=initial_Fin,
        )

        self.cls_layer = ClsBinningAndTransformLayer(
            n_ell=3 * tfr_n_side,
            n_bins=n_cls_bins,
            l_min_per_pair=l_min_per_pair,
            l_max_per_pair=l_max_per_pair,
        )

        # Separate LayerNorm per branch so the high-dimensional map features and the
        # compact Cls features are independently normalised before the embedding / concatenation.
        self.map_norm = tf.keras.layers.LayerNormalization(axis=-1, name="map_norm")
        self.cls_norm = tf.keras.layers.LayerNormalization(axis=-1, name="cls_norm")

        self.cls_embedding_layers = cls_embedding_layers
        self.regression_head_layers = regression_head_layers

        dense_widths = [l.units for l in cls_embedding_layers if hasattr(l, "units")]
        cls_out_dim = dense_widths[-1] if dense_widths else self.cls_layer.n_cls_flat
        LOGGER.warning(
            f"MapsPlusCLSNetwork: n_cls_bins={n_cls_bins}, n_z_cross={len(l_max_per_pair)}, "
            f"cls_flat_dim={self.cls_layer.n_cls_flat}, "
            f"cls_emb_dim={cls_out_dim} ({'embedding' if cls_embedding_layers else 'no embedding'})"
        )

    def call(self, inputs, training=False):
        """Forward pass.

        Args:
            inputs (tuple): ``(maps, cls)`` where
                - maps: float tensor ``(batch, n_pix, n_channels)``
                - cls:  float tensor ``(batch, n_ell, n_z_cross)``  (raw per-ell values)
            training (bool): Keras training flag.

        Returns:
            tf.Tensor: Summary statistics, shape ``(B, out_features)``.
        """
        maps, cls = inputs

        # Map branch: GCNN → flatten → normalise
        x = self.gcnn(maps, training=training)                  # (batch, n_pix_reduced, n_ch)
        x_flat = tf.reshape(x, (tf.shape(x)[0], -1))           # (B, n_map_flat)
        x_flat = self.map_norm(x_flat, training=training)

        # Cls branch: per-pair bin + log transform → normalise → embed
        cls_flat = self.cls_layer(cls, training=training)       # (B, n_bins * n_z_cross)
        cls_flat = self.cls_norm(cls_flat, training=training)
        for layer in self.cls_embedding_layers:
            cls_flat = layer(cls_flat, training=training)       # (B, emb_width) after last Dense+LN

        # Concatenate and pass through the regression head
        x = tf.concat([x_flat, cls_flat], axis=-1)
        for layer in self.regression_head_layers:
            x = layer(x, training=training)
        return x
