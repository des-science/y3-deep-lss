# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created November 2023
Author: Arne Thomsen
"""

import tensorflow as tf
from deepsphere import healpy_layers

from deep_lss.nets.heads.regression_head import get_regression_head
from deep_lss.nets.layers.maps.global_attention import HealpyGlobalAttention
from deep_lss.nets.layers.maps.input_normalization import EmpiricalInputNormalization
from msfm.utils import logger

LOGGER = logger.get_logger(__file__)

# graph-convolution basis for the graph-conv pooling slots and the residual body.
# All three are polynomial filters in the graph Laplacian (isotropic, rotation-equivariant);
# they differ only in the polynomial basis. "bernstein" (BernNet, arXiv:2106.10994) uses the
# Bernstein basis, whose non-negative coefficients give a better-conditioned / more stable
# parametrization of the learned spectral response. Selected via network.kwargs.conv_type.
_CONV_HELPERS = {
    "cheby": healpy_layers.HealpyChebyshev,
    "mono": healpy_layers.HealpyMonomial,
    "bernstein": healpy_layers.HealpyBernstein,
}
# per-basis string code consumed by Healpy_ResidualLayer for the residual body
_CONV_BASIS_CODES = {"cheby": "CHEBY", "mono": "MONO", "bernstein": "BERN"}


class ResNetLayers:
    """Class used to build the layers of the ResNet network, which was used as the fiducial architecture in Janis'
    KiDS1000 analysis.
    """

    def __init__(
        self,
        out_features,
        # convolutions
        base_channels=32,
        pool_layers=3,
        pool_widen=True,
        conv_layers=2,
        conv_widen=False,
        channel_multiplier=2,
        residual_layers=6,
        residual_block_type="residual",
        mlp_ratio=4,
        layer_scale_init=1e-6,
        drop_path_rate=0.0,
        conv_type="cheby",
        global_attention=None,
        residual_attention=None,
        residual_attention_every=1,
        body_dropout_rate=None,
        # regression head
        head_type="dense",
        dense_layers=None,
        dropout_rate=None,
        # misc
        poly_degree=5,
        norm_kwargs={},
        activation=tf.nn.relu,
        smoothing_kwargs=None,
        input_norm=False,
        smoothing_external=False,
        spmm_backend="csr",
    ) -> None:
        """Class used to build the layers of the ResNet network, which was used as the fiducial architecture in Janis'
        KiDS1000 analysis.

        Args:
            out_features (int, optional): Output shape of the regression head. This determines the size of the learned
                summary statistics. Defaults to 6.
            base_channels (int, optional): Number of channels after the first layer of the network. Width then grows
                by ``channel_multiplier`` on every widening downsampling stage (see ``pool_widen``/``conv_widen``).
                Defaults to 32.
            pool_layers (int, optional): Number of pure pooling stages -- strided ``HealpyPseudoConv`` layers that
                downsample the neighboring Healpix pixels (nside halved per stage) without a graph convolution. These
                layers are fairly cheap and their number effectively determines how expensive the following graph
                convolutions are (they run at the coarser nside). Defaults to 3.
            pool_widen (bool, optional): If True, each pooling stage also multiplies the channel count by
                ``channel_multiplier`` (the classic pool-and-widen schedule). If False the pooling stages keep the
                width at ``base_channels``. Defaults to True.
            conv_layers (int, optional): Number of graph-conv pooling stages. Each stage is a real graph convolution
                (basis set by ``conv_type``) + LayerNorm + a strided ``HealpyPseudoConv``, so it both convolves and
                downsamples -- the same downsampling role as ``pool_layers`` but with an added convolution. Defaults
                to 2.
            conv_widen (bool, optional): If True, each graph-conv pooling stage ALSO multiplies the channel count by
                ``channel_multiplier`` (handled identically to ``pool_widen``), coupling a real graph convolution to
                every channel-widening downsampling step -- a graph-U-Net schedule where genuine
                (rotation-equivariant) convolution happens at every resolution level. If False the graph-conv stages
                keep the width constant and only add convolution (the default decoupled schedule). NOTE: the pure
                ``pool_layers`` run FIRST (at the highest nside), so keep >=1-2 of them to pool off the top
                resolutions before the first real conv. A real conv at nside 512/256 is mainly a COMPUTE cost (it
                processes 4x/16x the pixels of nside 128); it does NOT trip the ``nnz(L)*output.shape[1] > 2^31``
                SPARSE_LIMIT unless the ``coo`` backend is used -- that ceiling is coo-only (cuSPARSE ``csr`` csrmm
                has no such limit; see deepsphere.utils). Defaults to False.
            channel_multiplier (int, optional): Integer factor by which the channel count grows on each widening
                downsampling stage. The deep-body width is ``base_channels * channel_multiplier^(#widening stages)``,
                where a widening stage is any ``pool_layers`` stage (if ``pool_widen``) plus any ``conv_layers`` stage
                (if ``conv_widen``). Widening is applied AFTER each stage's layers, so ``base_channels`` stays the
                width after the first layer. Defaults to 2.
            residual_layers (int, optional): Number of residual layers. These are the main graph convolutions
                (channel-preserving). Defaults to 6.
            residual_block_type (str, optional): Which block to use for the ``residual_layers`` body.
                "residual" (default) is the classic ``Healpy_ResidualLayer`` (two full ChebK convs +
                LayerNorm + skip, basis per ``conv_type``). "convnext" is a ConvNeXt-style block
                (``Healpy_ConvNeXtLayer``, arXiv:2201.03545): a single depthwise ChebK conv (one
                spectral filter per channel, spatial mixing only) + LayerNorm + an inverted-bottleneck
                pointwise GELU MLP (channel mixing) + LayerScale + DropPath + skip. The ConvNeXt block
                has ONE graph conv per block vs the residual block's two (~half the sparse ``L @ x``
                cost) and, because its depthwise kernel is ``[C, K]``, the polynomial order
                ``poly_degree`` only adds ``C * K`` params (no ``C^2`` blowup) — so large receptive
                fields are cheap. It is Chebyshev-only (depthwise has no Monomial/Bernstein path), so
                ``conv_type`` then only affects the ``conv_layers`` pooling stages, and the block uses
                GELU regardless of ``activation``. Defaults to "residual".
            mlp_ratio (float, optional): ConvNeXt block only — hidden width of the pointwise MLP as a
                multiple of the channel count. Defaults to 4.
            layer_scale_init (float, optional): ConvNeXt block only — LayerScale init for the residual
                branch (near-identity at start); ``None`` disables LayerScale. Defaults to 1e-6.
            drop_path_rate (float, optional): stochastic-depth rate on the residual branch of EITHER
                block type (0.0 = off). Applied flat across the ``residual_layers`` (no timm-style linear
                0→rate ramp). DropPath is a per-sample Bernoulli mask with no trainable variables, so it
                is checkpoint-lineage preserving: a run may be turned on or off without invalidating an
                existing checkpoint. On the classic block it masks the branch before the skip, which is
                the exact stochastic-depth form here because that block is built with ``activation=None``
                (the pure ``x + branch`` residual); see ``GCNN_ResidualLayer`` for the post-activation
                caveat. Defaults to 0.0.
            conv_type (str, optional): Polynomial basis for the graph convolutions in the graph-conv pooling stages
                AND the residual body: "cheby" (Chebyshev, default), "mono" (Monomial), or "bernstein"
                (Bernstein basis, BernNet arXiv:2106.10994 — non-negative coefficients, better-conditioned learned
                spectral response). All are isotropic/rotation-equivariant polynomial filters in the graph Laplacian;
                only the basis differs. The pure ``pool_layers`` HealpyPseudoConv layers are unaffected. Defaults to
                "cheby".
            global_attention (dict, optional): if given, append a HealpyGlobalAttention block (a global
                self-attention transformer encoder over the coarsest-nside pixel tokens) at the END of the
                graph-conv body, before the regression head. The dict is passed as keyword arguments to
                HealpyGlobalAttention (num_heads, key_dim, mlp_ratio, n_layers, dropout_rate,
                positional_embedding, layer_scale_init, activation); ``{}`` uses all its defaults. This adds one
                all-to-all mixing stage to the otherwise purely-local graph-conv receptive field, mirroring the
                global-attention stage of the HEALPix nested transformer. Defaults to None (no attention block).
            residual_attention (dict, optional): if given, INTERLEAVE ``HealpyGlobalAttention`` block(s) INSIDE
                the residual body — one block after every ``residual_attention_every``-th residual layer —
                instead of (or in addition to) the single ``global_attention`` tail. The dict is the same
                HealpyGlobalAttention kwargs shape as ``global_attention``. Motivation (combined-probe gap):
                the transformer distributes content-dependent, all-to-all mixing THROUGH its shared
                post-injection body, whereas the GCNN body is purely-local additive graph conv with at most
                one attention tail; interleaving lets the graph convs act on globally-mixed features (a
                CoAtNet-style conv↔attention hybrid). In the multi-resolution combined path the whole residual
                body lives in ``ResNetMultiResEncoder.gcnn_post`` (post lensing+clustering fusion, at the
                coarse nside), so these blocks mix the two fused probes. Channel-preserving and Fout-free like
                the tail block, so ``HealpyGCNN``/``split_layers_at_nside`` route it through passthrough
                without disturbing the nside bookkeeping. Defaults to None (no interleaved attention).
            residual_attention_every (int, optional): insert one interleaved ``residual_attention`` block
                after every Nth residual layer (1-indexed; e.g. ``residual_layers=5`` with ``every=2`` places
                blocks after residual layers 2 and 4). Only used when ``residual_attention`` is given. Must be
                >= 1. Defaults to 1 (a block after every residual layer).
            body_dropout_rate (float, optional): If set, insert a ``SpatialDropout1D(body_dropout_rate)`` after
                EACH residual-body block (and after its interleaved attention block, if any). This is the ONLY
                dropout inside the map trunk -- the map encoder is otherwise unregularized (``dropout_rate``
                builds only the regression head, which the maps+cls composite reaches after the GCNN body).
                ``SpatialDropout1D`` drops whole feature-map channels across all footprint pixels (the conv
                analogue of dropout: neighboring HEALPix pixels are correlated, so element-wise dropout is a
                weak regularizer). Carries NO trainable variables, so toggling it does not change the
                object-based checkpoint variable set (a body-dropout run and a ``None`` run share weight
                structure and can be restored across each other). Motivation: regularize the trunk that mines
                the fine-scale non-Gaussian features where CosmoGrid-vs-DES misspecification lives -- but note
                misspecification is signless, so judge by coverage (SBC/TARP/HPD) + DES PPC, not FoM alone.
                Defaults to None (no body dropout -- byte-identical to the current graph).
            head_type (str, optional): Type of regression head to be used, allowed are "dense" and "conv. Defaults to
                "dense".
            dropout_rate (float, optional): Dropout rate within the regression head. Defaults to None, then it's not
                included.
            poly_degree (int, optional): Degree of the polynomials within the Chebyshev convolutions. Defaults to 5.
            norm_kwargs (dict, optional): Keyword arguments to be passed to the normalization layers. Defaults to {}.
            activation (callable, optional): Non-linear activation function to be used throughout. Defaults to
                tf.nn.relu.
            smoothing_kwargs (dict, optional): Keyword arguments to be passed to the smoothing layer. Defaults to None,
                then no smoothing is performed within the network.
            input_norm (bool, optional): Standardize the smoothed maps with an EmpiricalInputNormalization
                layer placed right after the smoothing front-end (same placement as the transformer map
                encoders): per-channel mean/inv_std measured from training data on fresh runs
                (``compute_input_norm_stats`` via the ``smooth_groups``/``masks``/``load_input_norm_stats``
                interface below) and restored from the checkpoint otherwise. Adds checkpoint variables —
                a checkpoint trained with one setting can only be restored with the same one. Requires
                ``smoothing_kwargs`` (channel count and footprint mask). Defaults to False.
            smoothing_external (bool, optional): Set by ``ResNetMultiResEncoder`` when it builds
                this spec with ``smoothing_kwargs=None`` because smoothing (and input norm) live in
                the encoder instead — silences the missing-smoothing warning. Defaults to False.
            spmm_backend (str, optional): sparse-matmul backend for the (single-res) smoothing layer
                built here ("coo"/"csr"/"gather"; see deepsphere.utils.make_spmm_operator). Defaults
                to "csr" (cuSPARSE; numerically equivalent to "coo" and faster). The graph
                convolutions get their backend from the wrapping HealpyGCNN's own ``spmm_backend``
                (set by BaseModel / the maps+cls composite), not from here.
        """
        self.layers = []

        if conv_type not in _CONV_HELPERS:
            raise ValueError(f"conv_type must be one of {sorted(_CONV_HELPERS)}, got {conv_type!r}")
        if not isinstance(channel_multiplier, int) or channel_multiplier < 1:
            raise ValueError(f"channel_multiplier must be a positive integer, got {channel_multiplier!r}")
        if residual_block_type not in ("residual", "convnext"):
            raise ValueError(f"residual_block_type must be 'residual' or 'convnext', got {residual_block_type!r}")
        if residual_attention is not None and (
            not isinstance(residual_attention_every, int) or residual_attention_every < 1
        ):
            raise ValueError(f"residual_attention_every must be an int >= 1, got {residual_attention_every!r}")
        conv_helper = _CONV_HELPERS[conv_type]
        conv_basis_code = _CONV_BASIS_CODES[conv_type]

        self.smoothing_layer = None
        self.input_norm_layer = None
        self._input_norm_mask = None

        if smoothing_kwargs is not None:
            if "split_probes" in smoothing_kwargs:
                raise ValueError(
                    "Per-probe smooth_nside (split_probes) is not consumed by ResNetLayers directly — "
                    "run_training dispatches it to ResNetMultiResEncoder, which owns the smoothing and "
                    "builds this spec with smoothing_kwargs=None"
                )
            self.smoothing_layer = healpy_layers.HealpySmoothing(**smoothing_kwargs, spmm_backend=spmm_backend)
            self.layers.append(self.smoothing_layer)
        elif smoothing_external:
            LOGGER.info("Smoothing (and input norm) handled externally by ResNetMultiResEncoder")
        else:
            LOGGER.warning("No smoothing layer is included in the network")

        if input_norm:
            if smoothing_kwargs is None:
                raise ValueError(
                    "input_norm requires smoothing_kwargs (per-channel fwhm for the channel count and the "
                    "footprint mask)"
                )
            # same config geometry as the smoothing front-end (cf. _input_norm_footprint for the
            # transformer encoders); the layer binarizes it to a 0/1 indicator itself
            self._input_norm_mask = smoothing_kwargs.get("mask")
            self.input_norm_layer = EmpiricalInputNormalization(len(smoothing_kwargs["fwhm"]), self._input_norm_mask)
            self.layers.append(self.input_norm_layer)

        # pure pooling stages: strided HealpyPseudoConv, downsampling (nside halved) without a graph conv.
        # Width grows by channel_multiplier per stage when pool_widen (applied after each append, so
        # base_channels stays the width after the first layer).
        n_channels = base_channels
        for _ in range(pool_layers):
            self.layers.append(healpy_layers.HealpyPseudoConv(p=1, Fout=n_channels, activation=activation))
            if pool_widen:
                n_channels *= channel_multiplier

        # graph-conv pooling stages: graph conv (basis per conv_type) + LayerNorm + strided HealpyPseudoConv.
        # Width is handled identically to the pooling stages: grows by channel_multiplier per stage when
        # conv_widen (graph-U-Net schedule), otherwise constant.
        for _ in range(conv_layers):
            self.layers.append(conv_helper(K=poly_degree, Fout=n_channels, activation=activation))
            self.layers.append(tf.keras.layers.LayerNormalization(**{"axis": -1, **norm_kwargs}))
            self.layers.append(healpy_layers.HealpyPseudoConv(p=1, Fout=n_channels, activation=activation))
            if conv_widen:
                n_channels *= channel_multiplier

        # residual body (channel-preserving). "residual" is the classic two-ChebK residual block;
        # "convnext" is the depthwise-separable ConvNeXt block (one graph conv + pointwise MLP).
        # When residual_attention is given, a HealpyGlobalAttention block is interleaved after every
        # residual_attention_every-th residual layer (see the arg docs): distributed content-dependent
        # mixing through the body, so the graph convs act on globally-mixed features.
        for i in range(residual_layers):
            if residual_block_type == "convnext":
                self.layers.append(
                    healpy_layers.Healpy_ConvNeXtLayer(
                        K=poly_degree,
                        mlp_ratio=mlp_ratio,
                        layer_scale_init=layer_scale_init,
                        drop_path_rate=drop_path_rate,
                        activation="gelu",
                        norm_kwargs=norm_kwargs,
                        use_bias=True,
                    )
                )
            else:
                self.layers.append(
                    healpy_layers.Healpy_ResidualLayer(
                        conv_basis_code,
                        layer_kwargs={"K": poly_degree, "activation": activation, "use_bias": True},
                        # the delta loss is only compatible with layer and not batch normalization
                        use_bn=True,
                        bn_kwargs=norm_kwargs,
                        norm_type="layer_norm",
                        # block-level activation stays None (it lives inside layer_kwargs), so the skip is
                        # the pure `x + branch` form and stochastic depth is exact here
                        drop_path_rate=drop_path_rate,
                    )
                )
            if residual_attention is not None and (i + 1) % residual_attention_every == 0:
                self.layers.append(HealpyGlobalAttention(**residual_attention))
            # optional trunk regularization: channel-wise (SpatialDropout1D) dropout after each block.
            # No trainable variables, so it is lineage-preserving; rides HealpyGCNN's passthrough branch
            # (no Fout / no nside change), exactly like the LayerNorms in the conv-pooling stages.
            if body_dropout_rate is not None:
                self.layers.append(tf.keras.layers.SpatialDropout1D(body_dropout_rate))

        # optional global self-attention over the coarsest-nside pixel tokens, at the end of the
        # conv body (before the head). HealpyGCNN routes it through its passthrough branch and does
        # not count it toward the nside reduction, so it stays a channel-preserving tail; it is part
        # of the conv-layer snapshot so the maps+cls composite / multi-res encoder pick it up too.
        if global_attention is not None:
            self.layers.append(HealpyGlobalAttention(**global_attention))

        # snapshot conv-only layers before the regression head is appended
        self._conv_layers = list(self.layers)

        # regression head
        regression_head_layers = get_regression_head(
            out_features=out_features,
            head_type=head_type,
            dense_layers=dense_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            poly_degree=poly_degree,
            norm_kwargs=norm_kwargs,
        )
        # head without the leading Flatten — ResNetSummaryNetwork owns the readout on BOTH paths
        self._head_type = head_type
        self._head_layers_no_flatten = regression_head_layers[1:] if head_type == "dense" else regression_head_layers
        self.layers.extend(regression_head_layers)

    def get_layers(self):
        return self.layers

    # --- empirical input-norm interface, mirroring the transformer map encoders
    # (HealpixMapEncoder) so run_training measures the statistics through a single code path.
    # The layer objects below are the SAME instances that live in self.layers, so loading the
    # statistics through this spec reaches the layer inside the built network / composite.

    def smooth_groups(self, maps, training=False):
        """Smoothed maps (fp32) as a one-element list, used to measure the input-norm statistics."""
        return [maps if self.smoothing_layer is None else self.smoothing_layer(maps, training=training)]

    @property
    def masks(self):
        """Footprint mask as a one-element list (aligned with ``smooth_groups``)."""
        return [self._input_norm_mask]

    def load_input_norm_stats(self, stats):
        """Load the single ``(mean, inv_std)`` group into the input-norm layer."""
        ((mean, inv_std),) = stats
        self.input_norm_layer.load_stats(mean, inv_std)

    def get_conv_layers(self):
        """Return only the graph-convolution layers, without the regression head."""
        return self._conv_layers

    def get_head_layers_no_flatten(self):
        """Return the regression head layers without the leading readout (Flatten/pool) layer.

        Used by ResNetSummaryNetwork, which owns the readout — flatten or ``map_pool`` — on both
        the maps-only and the maps+Cls path and hands this list a ``(B, d)`` vector.

        Raises:
            ValueError: for ``head_type='conv'``. That head is a graph convolution followed by a
                mean over the pixel axis, i.e. it IS a readout, so it cannot follow one; it only
                composes with the layer-list path (``get_layers()``), which no live config uses.
        """
        if self._head_type != "dense":
            raise ValueError(
                f"head_type={self._head_type!r} has no readout-free form: the conv head ends in its own "
                "mean over the pixel axis and cannot run on the already-reduced (B, d) vector that "
                "ResNetSummaryNetwork produces. Use head_type='dense'."
            )
        return self._head_layers_no_flatten
