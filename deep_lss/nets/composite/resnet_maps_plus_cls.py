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
  1. HealpyGCNN processes the HEALPix maps → flatten
     [→ linear Dense(map_feature_dim)] → map_norm (LN)                     (map branch)
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
        conv_layers=None,
        cls_embedding_layers=None,
        regression_head_layers=None,
        n_side=None,
        tfr_n_side=None,
        indices=None,
        n_neighbors=None,
        max_batch_size=None,
        initial_Fin=None,
        n_cls_bins=None,
        l_min_per_pair=None,
        l_max_per_pair=None,
        cls_transform="asinh_per_feature",
        map_feature_dim=None,
        map_encoder=None,
        spmm_backend="csr",
    ):
        """
        Args:
            conv_layers (list): Graph-convolution layers (ResNetLayers.get_conv_layers()).
                Mutually exclusive with ``map_encoder``.
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
            map_feature_dim (int, optional): Bottleneck width of the map branch at the fusion
                point, mirroring the transformer composite: a linear Dense projects the flattened
                GCNN features (~1e5-dim) down to this width before map_norm, so both branches
                meet the concatenation at comparable dimensionality. ``None`` (default) keeps the
                legacy behavior of concatenating the raw flattened features. Note: object-based
                checkpointing is structure-sensitive — a checkpoint trained without the projection
                can only be restored with ``map_feature_dim`` unset, and vice versa.
            map_encoder (tf.keras.Model, optional): Prebuilt map branch returning
                ``(B, n_pix_reduced, n_ch)`` conv features — the multi-resolution
                ``ResNetMultiResEncoder`` (per-probe ``smooth_nside``). Mutually exclusive with
                ``conv_layers``/``n_side``/``indices``/``initial_Fin``; smoothing and input norm
                then live inside the encoder. Like ``map_feature_dim``, this is a separate
                checkpoint lineage: a multi-res checkpoint restores only with the matching
                ``smooth_nside`` mapping set, and vice versa.
        """
        super().__init__()

        if (map_encoder is None) == (conv_layers is None):
            raise ValueError("pass exactly one of conv_layers (single-res GCNN) or map_encoder (multi-res)")
        if map_encoder is not None and any(a is not None for a in (n_side, indices, initial_Fin)):
            raise ValueError("map_encoder owns the map branch — n_side/indices/initial_Fin must be None")

        # keep the single-res attribute creation order unchanged (object-based checkpointing is
        # structure-sensitive; a None attribute is untracked, so the map_encoder path adds no
        # variables to the single-res object graph and vice versa)
        self.gcnn = (
            HealpyGCNN(
                nside=n_side,
                indices=indices,
                layers=conv_layers,
                n_neighbors=n_neighbors,
                max_batch_size=max_batch_size,
                initial_Fin=initial_Fin,
                spmm_backend=spmm_backend,
            )
            if map_encoder is None
            else None
        )
        self.map_encoder = map_encoder

        self.cls_layer = ClsBinningAndTransformLayer(
            n_ell=3 * tfr_n_side,
            n_bins=n_cls_bins,
            l_min_per_pair=l_min_per_pair,
            l_max_per_pair=l_max_per_pair,
            cls_transform=cls_transform,
        )

        # Optional map-branch bottleneck: linear, like the transformer's final Dense(num_outputs)
        # projection; the LayerNorm below then normalises the projected features.
        self.map_projection = (
            tf.keras.layers.Dense(map_feature_dim, name="map_projection") if map_feature_dim is not None else None
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
            f"ResNetMapsPlusCLSNetwork: map_feature_dim="
            f"{map_feature_dim if map_feature_dim is not None else 'None (raw flattened GCNN features)'}, "
            f"n_cls_bins={n_cls_bins}, n_z_cross={len(l_max_per_pair)}, "
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

        # Map branch: GCNN (or multi-res encoder) → flatten → (project) → normalise
        map_branch = self.map_encoder if self.map_encoder is not None else self.gcnn
        x = map_branch(maps, training=training)                 # (batch, n_pix_reduced, n_ch)
        x_flat = tf.reshape(x, (tf.shape(x)[0], -1))           # (B, n_map_flat)
        if self.map_projection is not None:
            x_flat = self.map_projection(x_flat)                # (B, map_feature_dim)
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
