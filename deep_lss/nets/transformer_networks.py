# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2025
Author: Arne Thomsen

Integration layer that turns the nested hierarchical local-window transformer
(``healpix_transformer.HealpixNestedHierarchicalLocalWindowTransformer``) into a
drop-in network for the y3-deep-lss training pipeline.

Two pieces are added on top of the raw transformer:

  1. ``HealpixNestedTokenizer`` — applies the *same* input handling the DeepSphere
     GCNNs rely on. The maps are first Gaussian-smoothed by ``HealpySmoothing``
     (prepended in the network wrappers below), then this layer reorders the
     footprint pixels ``(B, P, C)`` into superpixel-major, NEST-contiguous blocks
     ``(B, N * 4^L, C)`` so they tile complete nested ``4^L`` superpixels — the shape
     the transformer's ``batch_flat_to_nested`` expects. Missing children (partial
     superpixels) are zero-filled; with ``token_nside >= msfm n_side_down`` the
     footprint is already complete and no padding is used.

  2. ``EmpiricalInputNormalization`` (optional, ``network.input_norm: true``; defined in
     ``input_normalization.py``) — standardizes the smoothed maps per channel with a scalar
     mean/std measured from training data at the start of a fresh run and frozen into the
     checkpoint, then re-applies the footprint mask so masked pixels stay exactly zero. Channels
     of very different physical scale (lensing shear vs clustering counts) thus enter the
     transformer balanced.

  3. ``HealpixTransformerNetwork`` (maps only) and ``TransformerMapsPlusCLSNetwork``
     (maps + binned Cls, mirroring ``MapsPlusCLSNetwork``) — pre-built ``tf.keras.Model``s
     that BaseModel uses directly (passed with ``n_side=None``), so no HealpyGCNN graph
     is constructed.
"""

import numpy as np
import healpy as hp
import tensorflow as tf

from deepsphere import healpy_layers

from msfm.utils import logger

from .healpix_transformer import HealpixNestedHierarchicalLocalWindowTransformer
from .maps_plus_cls_network import ClsBinningAndTransformLayer
from .split_smoothing import PerProbeSmoothing
from .input_normalization import EmpiricalInputNormalization

LOGGER = logger.get_logger(__file__)


def _build_fp32_smoothing(smoothing_kwargs):
    """Construct the smoothing front-end in float32 regardless of the active global policy.

    ``HealpySmoothing`` reads ``tf.keras.mixed_precision.global_policy()`` at construction to pick
    the dtype of its sparse kernel. Under a bf16/fp16 mixed-precision policy that makes the (eager)
    ``tf.sparse.sparse_dense_matmul`` run in low precision, which has no fast cuSPARSE kernel and is
    ~10x slower (benchmarked). Forcing float32 keeps the sparse smoothing fast; the network casts
    the smoothed maps to the transformer body's compute dtype afterwards, so the body still gets the
    bf16 speed/memory benefit. This mirrors how smoothing is kept outside the XLA region — the
    sparse op is the one component that does not follow the rest of the network.

    A ``{"split_probes": [...]}`` spec (mixed per-probe smooth_nside, see
    ``configuration.get_smoothing_kwargs``) builds a ``PerProbeSmoothing`` with one kernel per
    probe instead of a single ``HealpySmoothing``.
    """
    if not smoothing_kwargs:
        return None
    prev_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy("float32")
    try:
        if "split_probes" in smoothing_kwargs:
            return PerProbeSmoothing(smoothing_kwargs["split_probes"])
        return healpy_layers.HealpySmoothing(**smoothing_kwargs)
    finally:
        tf.keras.mixed_precision.set_global_policy(prev_policy)


def _input_norm_footprint(smoothing_kwargs):
    """Per-channel survey footprint ``(n_pix, n_channels)`` at the smoothing output resolution.

    Returns the same mask the smoothing front-end applies — its single ``mask`` for a plain
    ``HealpySmoothing``, or, for the ``split_probes`` (``PerProbeSmoothing``) case, the per-probe
    masks upsampled to the output nside and concatenated in channel order (matching how
    ``PerProbeSmoothing`` upsamples and concatenates its outputs). Handed to
    ``EmpiricalInputNormalization`` so the two front-ends share one footprint (see its docstring).
    Returns None when smoothing applies no mask (full sky), so the input-norm skips the re-mask.
    """
    if not smoothing_kwargs:
        return None
    if "split_probes" in smoothing_kwargs:
        parts = []
        for spec in smoothing_kwargs["split_probes"]:
            mask = spec["smoothing_kwargs"].get("mask")
            if mask is None:
                return None
            mask = np.asarray(mask)
            parent_output_idx = spec.get("parent_output_idx")
            if parent_output_idx is not None:
                # upsample the probe's coarse footprint to the output nside (each parent's value
                # repeated to its children), the identical upsampling PerProbeSmoothing applies
                mask = mask[parent_output_idx]
            parts.append(mask)
        return np.concatenate(parts, axis=-1)
    mask = smoothing_kwargs.get("mask")
    return None if mask is None else np.asarray(mask)


def _make_transformer_body(tokenizer, transformer, jit_compile_body):
    """Build the tokenizer -> transformer forward, optionally as one XLA-compiled subgraph.

    The ``HealpySmoothing`` front-end is deliberately left *outside* this body: it relies on
    ``tf.sparse.sparse_dense_matmul``, which XLA does not support (and which keeps the whole
    training step off the XLA path). Everything from the tokenizer onward — ``tf.gather``,
    dense projections, ``MultiHeadAttention``, ``LayerNormalization``, reshapes/patch-merges —
    is XLA-clean. On the small-dim/short-sequence transformer used here the runtime is dominated
    by the launch count and intermediate DRAM traffic of those many tiny kernels, so jit-compiling
    this region fuses them into far fewer kernels without changing the numerics.

    ``jit_compile_body=False`` returns a plain Python callable, identical to inlining the two calls,
    so the default behaviour is unchanged. The compiled region sits inside the (non-jit) outer
    train step, which sidesteps the known "XLA + MirroredStrategy freezes" issue that only affects
    jit-compiling the entire ``strategy.run`` step.
    """

    def body(x, training):
        x = tokenizer(x, training=training)
        return transformer(x, training=training)

    if jit_compile_body:
        LOGGER.warning("Compiling the tokenizer->transformer body with jit_compile=True (XLA)")
        return tf.function(body, jit_compile=True)
    return body


class HealpixNestedTokenizer(tf.keras.layers.Layer):
    """Reorder footprint maps into superpixel-major, NEST-contiguous nested blocks.

    The training pipeline yields HEALPix maps ``(B, P, C)`` over the (partial-sky)
    footprint pixels ``smooth_indices`` at resolution ``nside`` in NEST ordering. The
    transformer needs the pixels grouped into ``N`` top-level tokens (one per occupied
    ``token_nside`` superpixel), each holding the full set of ``4^L`` fine children in
    NEST order, where ``L = order(nside) - order(token_nside)``.

    This non-trainable layer builds a static gather index that performs that reordering
    in one ``tf.gather``. Footprint pixels that are absent from an occupied superpixel
    (only possible when ``token_nside`` is coarser than the msfm footprint padding
    ``n_side_down``) are filled with zeros via an appended zero pixel row.
    """

    def __init__(self, smooth_indices, nside, token_nside, in_channels, **kwargs):
        super().__init__(**kwargs)

        if nside <= token_nside:
            raise ValueError(f"nside ({nside}) must be greater than token_nside ({token_nside})")

        smooth_indices = np.asarray(smooth_indices).astype(np.int64)
        n_levels = int(hp.nside2order(nside) - hp.nside2order(token_nside))
        block = 4 ** n_levels
        n_pix_in = int(smooth_indices.shape[0])

        # NEST parent superpixel at token_nside, and the child offset within it.
        superpix = smooth_indices >> (2 * n_levels)
        child = smooth_indices & (block - 1)

        occupied = np.unique(superpix)  # sorted occupied top-level tokens
        n_tokens = int(occupied.shape[0])
        rank = np.searchsorted(occupied, superpix)  # 0..n_tokens-1 per pixel

        # Each footprint pixel maps to a unique (token, child) slot.
        target = rank * block + child

        # gather_idx[slot] = index of the footprint pixel that fills it, or n_pix_in
        # (the appended zero row) for empty slots.
        gather_idx = np.full(n_tokens * block, n_pix_in, dtype=np.int64)
        gather_idx[target] = np.arange(n_pix_in, dtype=np.int64)
        n_pad = int(np.sum(gather_idx == n_pix_in))

        self.in_channels = in_channels
        self.nside = nside
        self.token_nside = token_nside
        self.num_nested_levels = n_levels
        self.num_top_level_tokens = n_tokens
        self.num_pixels = int(n_tokens * block)
        self._n_pix_in = n_pix_in
        self.gather_idx = tf.constant(gather_idx, dtype=tf.int32)

        LOGGER.warning(
            f"HealpixNestedTokenizer: nside={nside}, token_nside={token_nside}, "
            f"levels={n_levels}, pixels/token={block}, tokens(N)={n_tokens}, "
            f"num_pixels={self.num_pixels}, zero-padded slots={n_pad} "
            f"({'padding-free' if n_pad == 0 else 'PARTIAL superpixels — token_nside < footprint padding'})"
        )

    def call(self, x, training=None):
        # (B, P, C) -> (B, P+1, C) with a trailing zero pixel used for empty slots
        zero_row = tf.zeros_like(x[:, :1, :])
        x = tf.concat([x, zero_row], axis=1)
        # gather into (B, N * 4^L, C)
        return tf.gather(x, self.gather_idx, axis=1)


class HealpixTransformerNetwork(tf.keras.Model):
    """Maps-only nested transformer: smoothing -> tokenizer -> transformer.

    The Gaussian smoothing is the identical ``HealpySmoothing`` front-end used by the
    DeepSphere networks, so the maps seen by the transformer are preprocessed the same
    way. Returns the ``(B, num_outputs)`` summary directly.
    """

    def __init__(
        self,
        smoothing_kwargs,
        smooth_indices,
        nside,
        token_nside,
        in_channels,
        num_outputs,
        transformer_kwargs,
        jit_compile_body=False,
        head_dropout_rate=None,
        input_norm=False,
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
        self.transformer = HealpixNestedHierarchicalLocalWindowTransformer(
            num_pixels=self.tokenizer.num_pixels,
            nside=nside,
            nside_down=token_nside,
            in_channels=in_channels,
            num_outputs=num_outputs,
            head_dropout_rate=head_dropout_rate,
            **transformer_kwargs,
        )
        # smoothing (sparse, XLA-incompatible) stays eager; the tokenizer->transformer body
        # is optionally fused with XLA — see _make_transformer_body.
        self._body = _make_transformer_body(self.tokenizer, self.transformer, jit_compile_body)
        # compute dtype of the body (bf16 under a mixed policy, else float32); the fp32-smoothed
        # maps are cast to it before the body so the transformer runs in the mixed-precision dtype.
        self._body_compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype

    def call(self, maps, training=False):
        x = maps
        if self.smoothing is not None:
            x = self.smoothing(x, training=training)
        if self.input_norm is not None:
            x = self.input_norm(x)
        x = tf.cast(x, self._body_compute_dtype)
        return self._body(x, training=training)


class TransformerMapsPlusCLSNetwork(tf.keras.Model):
    """Maps + Cls nested transformer, mirroring ``MapsPlusCLSNetwork``.

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
        self.transformer = HealpixNestedHierarchicalLocalWindowTransformer(
            num_pixels=self.tokenizer.num_pixels,
            nside=nside,
            nside_down=token_nside,
            in_channels=in_channels,
            num_outputs=map_feature_dim,
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

        # Independent LayerNorm per branch before fusion (as in MapsPlusCLSNetwork).
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
