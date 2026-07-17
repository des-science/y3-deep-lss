# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2025
Author: Arne Thomsen

Maps + Cls composite built on the nested hierarchical local-window transformer, mirroring
``deep_lss.nets.composite.resnet_maps_plus_cls.ResNetMapsPlusCLSNetwork`` but with the DeepSphere
GCNN map branch replaced by the transformer. It reuses the smoothing/input-norm/tokenizer helpers
defined alongside the maps-only ``HealpixTransformerNetwork`` in
``deep_lss.nets.encoders.maps.transformer.network``.
"""

import tensorflow as tf

from msfm.utils import logger

from deep_lss.nets.encoders.maps.transformer.network import build_map_encoder
from deep_lss.nets.layers.cls.binning import ClsBinningAndTransformLayer

LOGGER = logger.get_logger(__file__)


class TransformerMapsPlusCLSNetwork(tf.keras.Model):
    """Maps + Cls nested transformer, mirroring ``ResNetMapsPlusCLSNetwork``.

    Map branch:  smoothing -> tokenizer -> transformer (-> map_feature_dim) -> map_norm
    Cls branch:  ClsBinningAndTransformLayer -> cls_norm -> cls embedding MLP
    Fusion:      concat -> regression head -> (B, out_features)
    """

    def __init__(
        self,
        smoothing_kwargs,
        smooth_indices,
        nside,
        token_nside,
        in_channels,
        map_feature_dim,
        transformer_kwargs,
        tfr_n_side,
        n_cls_bins,
        l_min_per_pair,
        l_max_per_pair,
        cls_embedding_layers,
        regression_head_layers,
        jit_compile_body=False,
        cls_transform="asinh_per_feature",
        input_norm=False,
        masked_attention=False,
    ):
        super().__init__()

        # Map branch: single- or multi-resolution encoder returning the (B, map_feature_dim)
        # feature. The post-fusion head dropout is applied here (not inside the transformer), so the
        # encoder's own head dropout is None. build_map_encoder dispatches on the smoothing spec and
        # rejects masked_attention on the multi-resolution (split_probes) path.
        self.map_encoder = build_map_encoder(
            smoothing_kwargs=smoothing_kwargs,
            smooth_indices=smooth_indices,
            nside=nside,
            token_nside=token_nside,
            in_channels=in_channels,
            num_outputs=map_feature_dim,
            transformer_kwargs=transformer_kwargs,
            jit_compile_body=jit_compile_body,
            head_dropout_rate=None,
            input_norm=input_norm,
            masked_attention=masked_attention,
        )

        self.cls_layer = ClsBinningAndTransformLayer(
            n_ell=3 * tfr_n_side,
            n_bins=n_cls_bins,
            l_min_per_pair=l_min_per_pair,
            l_max_per_pair=l_max_per_pair,
            cls_transform=cls_transform,
        )

        # Independent LayerNorm per branch before fusion (as in ResNetMapsPlusCLSNetwork).
        self.map_norm = tf.keras.layers.LayerNormalization(axis=-1, name="map_norm")
        self.cls_norm = tf.keras.layers.LayerNormalization(axis=-1, name="cls_norm")

        self.cls_embedding_layers = cls_embedding_layers
        self.regression_head_layers = regression_head_layers

        LOGGER.warning(
            f"TransformerMapsPlusCLSNetwork: map_feature_dim={map_feature_dim}, "
            f"n_cls_bins={n_cls_bins}, n_z_cross={len(l_max_per_pair)}, "
            f"cls_flat_dim={self.cls_layer.n_cls_flat}"
        )

    def call(self, inputs, training=False):
        maps, cls = inputs

        # Map branch: smoothing (fp32) -> input norm (fp32) -> cast -> tokenizer -> transformer,
        # single- or multi-resolution, via the shared encoder -> (B, map_feature_dim) -> normalise.
        x = self.map_encoder(maps, training=training)
        x = self.map_norm(x, training=training)

        # Cls branch: per-pair bin + log transform -> normalise -> embed
        cls_flat = self.cls_layer(cls, training=training)
        cls_flat = self.cls_norm(cls_flat, training=training)
        for layer in self.cls_embedding_layers:
            cls_flat = layer(cls_flat, training=training)

        # Concatenate and pass through the regression head
        out = tf.concat([x, cls_flat], axis=-1)
        for layer in self.regression_head_layers:
            out = layer(out, training=training)
        return out
