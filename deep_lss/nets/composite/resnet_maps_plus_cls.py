# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2025
Author: Arne Thomsen

Network that concatenates DeepSphere map features with binned angular power spectra (Cls).
The DeepSphere GCNN map branch is built from ``ResNetLayers`` conv/head layer lists (the only
encoder exposing ``get_conv_layers`` / ``get_head_layers_no_flatten``), hence ``ResNet`` in the
name; the transformer-branch counterpart is
``deep_lss.nets.composite.transformer_maps_plus_cls.TransformerMapsPlusCLSNetwork``.

Architecture:
  1. HealpyGCNN processes the HEALPix maps → flatten → map_norm (LN)       (map branch)
  2. ClsBinningAndTransformLayer bins + gathers + sign-log-transforms Cls
     → cls_norm (LN) → cls_embedding MLP (Dense→LN × N)                    (Cls branch)
  3. Concatenate both branches
  4. regression_head (LN + hidden Dense layers + output)
"""

import tensorflow as tf

from deepsphere import HealpyGCNN

from msfm.utils import logger

from deep_lss.nets.layers.cls.binning import ClsBinningAndTransformLayer

LOGGER = logger.get_logger(__file__)


class ResNetMapsPlusCLSNetwork(tf.keras.Model):
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
        cls_transform="asinh_per_feature",
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
            cls_transform (str): Cls transform, forwarded to ClsBinningAndTransformLayer
                ("asinh_per_feature" or "log1p_fixed").
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
            cls_transform=cls_transform,
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
            f"ResNetMapsPlusCLSNetwork: n_cls_bins={n_cls_bins}, n_z_cross={len(l_max_per_pair)}, "
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
