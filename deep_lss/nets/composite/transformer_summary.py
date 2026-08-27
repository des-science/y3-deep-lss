# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2025
Author: Arne Thomsen

The nested-transformer summary network — the ONE network for both the maps-only and the maps+Cls
path, which differ only in whether a Cls branch is concatenated onto the map features. The GCNN
counterpart is ``deep_lss.nets.composite.resnet_summary.ResNetSummaryNetwork``. The map branch is
the shared encoder built by ``deep_lss.nets.encoders.maps.transformer.network.build_map_encoder``
(single- or multi-resolution), which already mean-pools its tokens, so there is no readout choice
here and no ``map_pool``.

Architecture:
  1. smoothing -> input norm -> tokenizer -> transformer -> mean pool over tokens
     [-> linear Dense(map_feature_dim)]                                             (map branch)
  2. ONLY with a Cls branch:
       a. map_norm (LN) over the map features
       b. ClsBinningAndTransformLayer -> cls_norm (LN) -> cls embedding MLP         (Cls branch)
       c. Concatenate both branches
  3. regression head (LN + hidden Dense layers + dropout + output) -> (B, out_features)

``map_norm`` belongs to the fusion, not to the map branch: it balances the map features against
the Cls features before the concatenation, which is meaningful only when there are two branches.
With no Cls the head's own leading LayerNormalization already normalizes the map features.

The transformer's internal head dropout is never used (``head_dropout_rate=None``); the ``head:``
block's dropout lives in the regression head on both paths.

``map_feature_dim`` is the width the map feature is projected to for the concatenation, exactly as
on the GCNN; ``None`` means no projection. Leave it ``None`` maps-only: the regression head already
opens with LayerNorm and ends in ``Dense(out_features)``, so a projection would be a second linear
layer with no nonlinearity between the two, buying nothing but parameters (nside 512 / base 32 /
double: a 1024 -> 512 crush in front of a Dense to ~10 outputs). With a Cls branch the width is the
point — it balances the two branches at the concat — and there it earns its place.
"""

import tensorflow as tf

from msfm.utils import logger

from deep_lss.nets.encoders.maps.transformer.network import build_map_encoder
from deep_lss.nets.layers.cls.binning import ClsBinningAndTransformLayer

LOGGER = logger.get_logger(__file__)


class TransformerSummaryNetwork(tf.keras.Model):
    """Nested-transformer summary network, with or without a Cls branch.

    The Cls arguments (``tfr_n_side``, ``n_cls_bins``, ``l_min_per_pair``, ``l_max_per_pair``,
    ``cls_embedding_layers``) are all-or-nothing: pass every one for maps+Cls, none for maps-only.
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
        regression_head_layers,
        tfr_n_side=None,
        n_cls_bins=None,
        l_min_per_pair=None,
        l_max_per_pair=None,
        cls_embedding_layers=None,
        jit_compile_body=False,
        cls_transform="asinh_per_feature",
        input_norm=False,
        masked_attention=False,
        spmm_backend="csr",
    ):
        super().__init__()

        # The Cls branch is all-or-nothing: a half-configured call would otherwise build a network
        # that trains fine and quietly ignores the Cls the pipeline is still yielding.
        cls_args = {
            "tfr_n_side": tfr_n_side,
            "n_cls_bins": n_cls_bins,
            "l_min_per_pair": l_min_per_pair,
            "l_max_per_pair": l_max_per_pair,
            "cls_embedding_layers": cls_embedding_layers,
        }
        self.return_cls = any(v is not None for v in cls_args.values())
        if self.return_cls and (missing := [k for k, v in cls_args.items() if v is None]):
            raise ValueError(
                f"incomplete Cls branch: {missing} left as None while the others were given. Pass all "
                f"of {sorted(cls_args)} for a maps+Cls network, or none of them for a maps-only one."
            )

        # Map branch: single- or multi-resolution encoder returning the (B, map_feature_dim)
        # feature, or the raw pooled feature when map_feature_dim is None (see the module
        # docstring). The head dropout is applied in the regression head (not inside the transformer),
        # so the encoder's own head dropout is None. build_map_encoder dispatches on the smoothing
        # spec and rejects masked_attention on the multi-resolution (split_probes) path.
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
            spmm_backend=spmm_backend,
        )

        self.cls_layer = (
            ClsBinningAndTransformLayer(
                n_ell=3 * tfr_n_side,
                n_bins=n_cls_bins,
                l_min_per_pair=l_min_per_pair,
                l_max_per_pair=l_max_per_pair,
                cls_transform=cls_transform,
            )
            if self.return_cls
            else None
        )

        # Independent LayerNorm per branch before fusion (as in ResNetSummaryNetwork). Both belong
        # to the fusion, so a maps-only network skips them and lets the regression head's own
        # leading LayerNorm do the job instead of stacking two back to back.
        self.map_norm = tf.keras.layers.LayerNormalization(axis=-1, name="map_norm") if self.return_cls else None
        self.cls_norm = tf.keras.layers.LayerNormalization(axis=-1, name="cls_norm") if self.return_cls else None

        self.cls_embedding_layers = cls_embedding_layers
        self.regression_head_layers = regression_head_layers

        if self.return_cls:
            LOGGER.warning(
                f"TransformerSummaryNetwork: map_feature_dim={map_feature_dim}, "
                f"n_cls_bins={n_cls_bins}, n_z_cross={len(l_max_per_pair)}, "
                f"cls_flat_dim={self.cls_layer.n_cls_flat}"
            )
        else:
            LOGGER.warning(
                f"TransformerSummaryNetwork: map_feature_dim={map_feature_dim or 'None (no projection)'}, "
                "NO Cls branch (maps only) — no fusion, no map_norm"
            )

    def call(self, inputs, training=False):
        """Forward pass.

        Args:
            inputs: with a Cls branch, the tuple ``(maps, cls)``; maps-only, the ``maps`` tensor on
                its own.
            training (bool): Keras training flag.

        Returns:
            tf.Tensor: Summary statistics, shape ``(B, out_features)``.
        """
        maps, cls = inputs if self.return_cls else (inputs, None)

        # Map branch: smoothing (fp32) -> input norm (fp32) -> cast -> tokenizer -> transformer,
        # single- or multi-resolution, via the shared encoder -> (B, map_feature_dim).
        x = self.map_encoder(maps, training=training)

        if self.return_cls:
            # Fusion: normalise each branch on its own, then concatenate.
            x = self.map_norm(x, training=training)

            # Cls branch: per-pair bin + log transform -> normalise -> embed
            cls_flat = self.cls_layer(cls, training=training)
            cls_flat = self.cls_norm(cls_flat, training=training)
            for layer in self.cls_embedding_layers:
                cls_flat = layer(cls_flat, training=training)

            x = tf.concat([x, cls_flat], axis=-1)

        # Regression head (opens with its own LayerNorm)
        for layer in self.regression_head_layers:
            x = layer(x, training=training)
        return x
