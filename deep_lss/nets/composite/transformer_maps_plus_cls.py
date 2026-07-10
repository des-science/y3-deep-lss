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

from deep_lss.nets.encoders.maps.transformer.network import (
    _build_fp32_smoothing,
    _input_norm_footprint,
    _make_transformer_body,
    _masked_attention_token_valid,
    HealpixNestedTokenizer,
)
from deep_lss.nets.encoders.maps.transformer.healpix_transformer import (
    HealpixNestedHierarchicalLocalWindowTransformer,
)
from deep_lss.nets.layers.maps.input_normalization import EmpiricalInputNormalization
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

        # sparse smoothing stays in float32 (no fast bf16 cuSPARSE kernel) — see _build_fp32_smoothing
        self.smoothing = _build_fp32_smoothing(smoothing_kwargs)
        if self.smoothing is None:
            LOGGER.warning("No smoothing layer is included in the transformer network")

        # standardize the smoothed fp32 maps with per-channel statistics (fresh runs measure them
        # via compute_input_norm_stats; resumes/evaluation restore the two checkpointed mean/inv_std
        # variables) — adds variables, so toggling changes the checkpoint lineage. The footprint
        # mask is the same config geometry handed to the smoothing front-end (see the layer).
        self.input_norm = (
            EmpiricalInputNormalization(in_channels, _input_norm_footprint(smoothing_kwargs)) if input_norm else None
        )

        self.tokenizer = HealpixNestedTokenizer(smooth_indices, nside, token_nside, in_channels)
        # masked_attention: exclude masked pixels from the transformer's attention/merges/
        # pooling instead of feeding them through as zeros (see _masked_attention_token_valid).
        # Mask constants only — no variables, so toggling keeps the checkpoint lineage.
        self.transformer = HealpixNestedHierarchicalLocalWindowTransformer(
            num_pixels=self.tokenizer.num_pixels,
            nside=nside,
            nside_down=token_nside,
            in_channels=in_channels,
            num_outputs=map_feature_dim,
            token_valid=_masked_attention_token_valid(masked_attention, smoothing_kwargs, self.tokenizer),
            **transformer_kwargs,
        )
        # smoothing (sparse, XLA-incompatible) stays eager; the tokenizer->transformer body
        # is optionally fused with XLA — see _make_transformer_body.
        self._body = _make_transformer_body(self.tokenizer, self.transformer, jit_compile_body)
        # compute dtype of the body (bf16 under a mixed policy, else float32); the fp32-smoothed
        # maps are cast to it before the body so the transformer runs in the mixed-precision dtype.
        self._body_compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype

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

        # Map branch: smoothing (fp32) -> input norm (fp32) -> cast -> tokenizer -> transformer -> normalise
        x = maps
        if self.smoothing is not None:
            x = self.smoothing(x, training=training)
        if self.input_norm is not None:
            x = self.input_norm(x)
        x = tf.cast(x, self._body_compute_dtype)
        x = self._body(x, training=training)  # (B, map_feature_dim)
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
