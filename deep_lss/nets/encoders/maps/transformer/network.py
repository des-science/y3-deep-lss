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
     ``layers/maps/input_normalization.py``) — standardizes the smoothed maps per channel with a
     scalar mean/std measured from training data at the start of a fresh run and frozen into the
     checkpoint, then re-applies the footprint mask so masked pixels stay exactly zero. Channels
     of very different physical scale (lensing shear vs clustering counts) thus enter the
     transformer balanced.

  3. ``HealpixTransformerNetwork`` (maps only) — a pre-built ``tf.keras.Model`` that BaseModel
     uses directly (passed with ``n_side=None``), so no HealpyGCNN graph is constructed. The
     maps + Cls composite ``TransformerMapsPlusCLSNetwork`` lives in
     ``deep_lss.nets.composite.transformer_maps_plus_cls`` and reuses the helpers and tokenizer
     defined here.
"""

import numpy as np
import healpy as hp
import tensorflow as tf

from msfm.utils import logger

from .healpix_transformer import HealpixNestedHierarchicalLocalWindowTransformer
from deep_lss.nets.encoders.maps.multires import MultiResEncoderMixin
from deep_lss.nets.layers.maps.smoothing import HealpySmoothing, fp32_policy_scope
from deep_lss.nets.layers.maps.input_normalization import EmpiricalInputNormalization

LOGGER = logger.get_logger(__file__)


def _build_fp32_smoothing(smoothing_kwargs, spmm_backend="csr"):
    """Construct the single-resolution ``HealpySmoothing`` front-end in float32.

    The ``{"split_probes": [...]}`` (mixed per-probe smooth_nside) case is handled by
    ``HealpixMultiResMapEncoder``, which builds its own ``PerProbeSmoothing``; this helper only
    ever sees a single-kernel spec on the single-resolution path. ``spmm_backend`` selects the
    sparse-matmul kernel for the smoothing (see deepsphere.utils.make_spmm_operator); the smoothing
    is carved out of the XLA body and runs eager/graph, where "csr" (cuSPARSE) is a numerically
    equivalent, faster drop-in for the default "coo" path.
    """
    if not smoothing_kwargs:
        return None
    if "split_probes" in smoothing_kwargs:
        raise ValueError(
            "_build_fp32_smoothing received a split_probes spec — the multi-resolution "
            "encoder (HealpixMultiResMapEncoder) owns per-probe smoothing."
        )
    with fp32_policy_scope():
        return HealpySmoothing(**smoothing_kwargs, spmm_backend=spmm_backend)


def _input_norm_footprint(smoothing_kwargs):
    """Per-channel survey footprint ``(n_pix, n_channels)`` for the single-resolution front-end.

    Returns the single ``HealpySmoothing`` ``mask`` handed to ``EmpiricalInputNormalization`` so
    the two front-ends share one footprint (see its docstring), or None when no mask is applied
    (full sky). The ``split_probes`` (multi-resolution) case builds per-group masks inside
    ``HealpixMultiResMapEncoder`` instead.
    """
    if not smoothing_kwargs:
        return None
    if "split_probes" in smoothing_kwargs:
        raise ValueError(
            "_input_norm_footprint received a split_probes spec — the multi-resolution "
            "encoder (HealpixMultiResMapEncoder) builds per-group input-norm masks."
        )
    mask = smoothing_kwargs.get("mask")
    return None if mask is None else np.asarray(mask)


def _masked_attention_token_valid(masked_attention, smoothing_kwargs, tokenizer):
    """Resolve the ``masked_attention`` option into per-slot token validity, or None.

    ``masked_attention`` is either a bool — ``True`` reuses the footprint the smoothing /
    input-norm front-ends already apply (``_input_norm_footprint``) — or an explicit
    ``(n_pix,)`` / ``(n_pix, n_channels)`` array over the footprint pixels, mirroring how
    those layers take their mask as a constructor array rather than deducing it from the
    data. Fractional (apodized/downsampled) values are binarized with ``> 0`` (the
    ``EmpiricalInputNormalization`` convention) and a pixel counts as valid if ANY channel
    observes it: channels that do not observe it stay zeroed by the input-norm mask, while
    the pixel still carries the other channels' information into attention.
    """
    if masked_attention is None or masked_attention is False:
        return None
    if masked_attention is True:
        footprint = _input_norm_footprint(smoothing_kwargs)
        if footprint is None:
            raise ValueError(
                "masked_attention: true requires the smoothing front-end's footprint "
                "mask (smoothing_kwargs['mask']), but none is configured — pass the "
                "mask array explicitly instead."
            )
    else:
        footprint = np.asarray(masked_attention)
    valid = footprint > 0
    if valid.ndim == 2:
        valid = valid.any(axis=-1)
    token_valid = tokenizer.valid_slots(valid)
    LOGGER.warning(
        f"Masked attention enabled: {int(token_valid.sum())}/{len(token_valid)} token "
        f"slots valid ({int(valid.sum())}/{len(valid)} footprint pixels)"
    )
    return token_valid


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
        self._gather_idx_np = gather_idx
        self.gather_idx = tf.constant(gather_idx, dtype=tf.int32)

        LOGGER.warning(
            f"HealpixNestedTokenizer: nside={nside}, token_nside={token_nside}, "
            f"levels={n_levels}, pixels/token={block}, tokens(N)={n_tokens}, "
            f"num_pixels={self.num_pixels}, zero-padded slots={n_pad} "
            f"({'padding-free' if n_pad == 0 else 'PARTIAL superpixels — token_nside < footprint padding'})"
        )

    def valid_slots(self, pixel_valid):
        """Map per-footprint-pixel validity ``(P,)`` to per-slot validity ``(num_pixels,)``.

        Applies the same reordering as ``call``; the appended zero-pad row (empty slots)
        is always invalid. Used to hand the transformer its static ``token_valid`` mask.
        """
        pixel_valid = np.asarray(pixel_valid).astype(bool).reshape(-1)
        if len(pixel_valid) != self._n_pix_in:
            raise ValueError(
                f"pixel_valid has {len(pixel_valid)} entries, expected {self._n_pix_in} "
                f"footprint pixels."
            )
        return np.concatenate([pixel_valid, [False]])[self._gather_idx_np]

    def call(self, x, training=None):
        # (B, P, C) -> (B, P+1, C) with a trailing zero pixel used for empty slots
        zero_row = tf.zeros_like(x[:, :1, :])
        x = tf.concat([x, zero_row], axis=1)
        # gather into (B, N * 4^L, C)
        return tf.gather(x, self.gather_idx, axis=1)


class HealpixMapEncoder(tf.keras.Model):
    """Common interface for the transformer map branch (smooth -> normalize -> tokenize -> transform).

    Both concrete encoders — ``HealpixSingleResMapEncoder`` (one kernel, all probes at one nside)
    and ``HealpixMultiResMapEncoder`` (per-probe nsides with injection) — implement the same four
    methods so the rest of the pipeline is resolution-agnostic:

      - ``call(maps, training)``          -> ``(B, num_outputs)`` map feature / summary.
      - ``smooth_groups(maps, training)`` -> list of per-resolution-group smoothed maps (fp32).
      - ``masks`` (property)              -> matching list of per-group footprint masks.
      - ``load_input_norm_stats(stats)``  -> load a list of ``(mean, inv_std)`` into the group layers.

    The last three drive the empirical input-norm measurement in ``run_training`` through a single
    code path (``compute_input_norm_stats``), regardless of resolution.
    """


class HealpixSingleResMapEncoder(HealpixMapEncoder):
    """Single-resolution map branch: smoothing -> input-norm -> tokenizer -> transformer.

    Used when smoothing resolves to a single kernel (all active probes share one nside). The
    Gaussian smoothing is the identical ``HealpySmoothing`` front-end used by the DeepSphere
    networks, so the maps seen by the transformer are preprocessed the same way. Returns
    ``(B, num_outputs)`` — the summary for maps-only, or the ``map_feature_dim`` feature for the
    maps+cls composite, matching the transformer's ``num_outputs``.
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
        masked_attention=False,
        spmm_backend="csr",
    ):
        super().__init__()

        # sparse smoothing stays in float32 (no fast bf16 cuSPARSE kernel) — see _build_fp32_smoothing
        self.smoothing = _build_fp32_smoothing(smoothing_kwargs, spmm_backend=spmm_backend)
        if self.smoothing is None:
            LOGGER.warning("No smoothing layer is included in the transformer network")

        # standardize the smoothed fp32 maps with per-channel statistics (fresh runs measure them
        # via compute_input_norm_stats; resumes/evaluation restore the two checkpointed mean/inv_std
        # variables) — adds variables, so toggling changes the checkpoint lineage. The footprint
        # mask is the same config geometry handed to the smoothing front-end (see the layer).
        self._input_norm_mask = _input_norm_footprint(smoothing_kwargs)
        self.input_norm = (
            EmpiricalInputNormalization(in_channels, self._input_norm_mask) if input_norm else None
        )

        self.tokenizer = HealpixNestedTokenizer(smooth_indices, nside, token_nside, in_channels)
        # masked_attention: exclude masked pixels from the transformer's attention/merges/pooling
        # instead of feeding them through as zeros (see _masked_attention_token_valid). Mask
        # constants only — no variables, so toggling keeps the checkpoint lineage.
        self.transformer = HealpixNestedHierarchicalLocalWindowTransformer(
            num_pixels=self.tokenizer.num_pixels,
            nside=nside,
            nside_down=token_nside,
            in_channels=in_channels,
            num_outputs=num_outputs,
            head_dropout_rate=head_dropout_rate,
            token_valid=_masked_attention_token_valid(masked_attention, smoothing_kwargs, self.tokenizer),
            **transformer_kwargs,
        )
        # smoothing (sparse, XLA-incompatible) stays eager; the tokenizer->transformer body is
        # optionally fused with XLA — see _make_transformer_body.
        self._body = _make_transformer_body(self.tokenizer, self.transformer, jit_compile_body)
        # compute dtype of the body (bf16 under a mixed policy, else float32); the fp32-smoothed
        # maps are cast to it before the body so the transformer runs in the mixed-precision dtype.
        self._body_compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype

    def smooth_groups(self, maps, training=False):
        """Smoothed maps (fp32) as a one-element list, used to measure the input-norm statistics."""
        return [maps if self.smoothing is None else self.smoothing(maps, training=training)]

    @property
    def masks(self):
        """Footprint mask as a one-element list (aligned with ``smooth_groups``)."""
        return [self._input_norm_mask]

    def load_input_norm_stats(self, stats):
        """Load the single ``(mean, inv_std)`` group into the input-norm layer."""
        (mean, inv_std), = stats
        self.input_norm.load_stats(mean, inv_std)

    def call(self, maps, training=False):
        x = maps
        if self.smoothing is not None:
            x = self.smoothing(x, training=training)
        if self.input_norm is not None:
            x = self.input_norm(x)
        x = tf.cast(x, self._body_compute_dtype)
        return self._body(x, training=training)


class HealpixMultiResMapEncoder(MultiResEncoderMixin, HealpixMapEncoder):
    """Multi-resolution map branch: per-probe smoothing at native nsides -> per-group input-norm
    -> per-group tokenizers -> one nested transformer that takes the finest probe as its main
    input and injects each coarser probe at the hierarchy level already running at that nside.
    The smoothing/grouping/input-norm plumbing is the shared ``MultiResEncoderMixin`` (also used
    by the GCNN ``ResNetMultiResEncoder``).

    Used in place of the inline smoothing/input-norm/tokenizer/body path when smoothing resolves
    to a ``{"split_probes": [...]}`` spec (mixed per-probe ``smooth_nside``, i.e. combined
    probes). The coarser probe (clustering @256) is never upsampled — it enters the hierarchy at
    its own scale, so the network is one level deeper for the finer probe (lensing @512).

    Returns ``(B, num_outputs)`` (the summary for maps-only, or the ``map_feature_dim`` feature
    for the maps+cls composite, matching the transformer's ``num_outputs``).
    """

    def __init__(
        self,
        smoothing_kwargs,
        nside,
        token_nside,
        num_outputs,
        transformer_kwargs,
        jit_compile_body=False,
        head_dropout_rate=None,
        input_norm=False,
        spmm_backend="csr",
    ):
        super().__init__()

        # per-probe fp32 smoothing + grouping by nside (finest first) — shared mixin plumbing;
        # the finest nside == the output nside is the transformer's main input, coarser nsides
        # are injections.
        groups = self._init_smoothing_and_groups(smoothing_kwargs, spmm_backend=spmm_backend)
        self._fine_group_idx = 0
        if groups[0]["nside"] != nside:
            raise ValueError(
                f"finest probe nside {groups[0]['nside']} != output nside {nside}; the main "
                "transformer input must be at the output nside."
            )
        if len(groups) < 2:
            raise ValueError("HealpixMultiResMapEncoder needs at least two resolution groups.")

        # one tokenizer per group; all must tile the same N top-level tokens
        self.tokenizers = [
            HealpixNestedTokenizer(g["indices"], g["nside"], token_nside, g["n_channels"]) for g in groups
        ]
        n_tokens = self.tokenizers[0].num_top_level_tokens
        for tok, g in zip(self.tokenizers, groups):
            if tok.num_top_level_tokens != n_tokens:
                raise ValueError(
                    f"group nside {g['nside']} has {tok.num_top_level_tokens} top-level tokens, "
                    f"expected {n_tokens} (footprints must share the same token_nside superpixels)."
                )

        fine_channels = groups[0]["n_channels"]
        injection_specs = [{"nside": g["nside"], "in_channels": g["n_channels"]} for g in groups[1:]]

        self.transformer = HealpixNestedHierarchicalLocalWindowTransformer(
            num_pixels=self.tokenizers[0].num_pixels,
            nside=nside,
            nside_down=token_nside,
            in_channels=fine_channels,
            num_outputs=num_outputs,
            head_dropout_rate=head_dropout_rate,
            injections=injection_specs,
            **transformer_kwargs,
        )

        # per-group input normalization (shared mixin plumbing, own mask per group)
        self._init_group_input_norm(input_norm)

        self._body_compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
        self._body = self._make_body(jit_compile_body)

        LOGGER.warning(
            f"HealpixMultiResMapEncoder: fine group nside={nside} ({fine_channels} ch), coarse "
            f"groups {[(g['nside'], g['n_channels']) for g in groups[1:]]}, N={n_tokens} "
            f"top-level tokens, num_outputs={num_outputs}, input_norm={input_norm}"
        )

    def _make_body(self, jit_compile_body):
        # tokenize each group and run the transformer with the coarse groups as injections; the
        # fp32 smoothing/input-norm stay outside (kept eager, like the single-resolution body).
        def body(group_tensors, training):
            tokens = [tok(t) for tok, t in zip(self.tokenizers, group_tensors)]
            injections = {
                self._groups[gi]["nside"]: tokens[gi]
                for gi in range(len(self._groups))
                if gi != self._fine_group_idx
            }
            return self.transformer(
                tokens[self._fine_group_idx], injections=injections, training=training
            )

        if jit_compile_body:
            LOGGER.warning("Compiling the multi-res tokenizer->transformer body with jit_compile=True (XLA)")
            return tf.function(body, jit_compile=True)
        return body

    def call(self, maps, training=False):
        group_tensors = self.smooth_groups(maps, training=training)
        if self.input_norms is not None:
            group_tensors = [norm(t) for norm, t in zip(self.input_norms, group_tensors)]
        group_tensors = [tf.cast(t, self._body_compute_dtype) for t in group_tensors]
        return self._body(group_tensors, training=training)


def build_map_encoder(
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
    masked_attention=False,
    spmm_backend="csr",
):
    """Build the transformer map branch, dispatching on the smoothing spec.

    A ``{"split_probes": [...]}`` spec (mixed per-probe ``smooth_nside``, i.e. combined probes)
    selects ``HealpixMultiResMapEncoder`` — per-probe smoothing with the coarser probe(s) injected
    into the hierarchy rather than upsampled; ``masked_attention`` is unsupported there (injection
    and ``token_valid`` are mutually exclusive) and raises. Any other spec (all active probes at a
    single nside) selects ``HealpixSingleResMapEncoder``.

    Both encoders return ``(B, num_outputs)`` and share the ``smooth_groups`` / ``masks`` /
    ``load_input_norm_stats`` input-norm interface (see ``HealpixMapEncoder``).
    """
    if "split_probes" in (smoothing_kwargs or {}):
        if masked_attention:
            raise ValueError(
                "masked_attention is not supported with multi-resolution (split_probes) smoothing."
            )
        return HealpixMultiResMapEncoder(
            smoothing_kwargs=smoothing_kwargs,
            nside=nside,
            token_nside=token_nside,
            num_outputs=num_outputs,
            transformer_kwargs=transformer_kwargs,
            jit_compile_body=jit_compile_body,
            head_dropout_rate=head_dropout_rate,
            input_norm=input_norm,
            spmm_backend=spmm_backend,
        )
    return HealpixSingleResMapEncoder(
        smoothing_kwargs=smoothing_kwargs,
        smooth_indices=smooth_indices,
        nside=nside,
        token_nside=token_nside,
        in_channels=in_channels,
        num_outputs=num_outputs,
        transformer_kwargs=transformer_kwargs,
        jit_compile_body=jit_compile_body,
        head_dropout_rate=head_dropout_rate,
        input_norm=input_norm,
        masked_attention=masked_attention,
        spmm_backend=spmm_backend,
    )


class HealpixTransformerNetwork(tf.keras.Model):
    """Maps-only nested transformer: a thin wrapper over a single ``map_encoder``.

    Delegates the whole map branch (smoothing -> input-norm -> tokenizer -> transformer, single- or
    multi-resolution) to the encoder built by ``build_map_encoder`` and returns its
    ``(B, num_outputs)`` summary directly. The empirical input-norm statistics are measured through
    ``self.map_encoder`` in ``run_training``.
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
        masked_attention=False,
        spmm_backend="csr",
    ):
        super().__init__()
        self.map_encoder = build_map_encoder(
            smoothing_kwargs=smoothing_kwargs,
            smooth_indices=smooth_indices,
            nside=nside,
            token_nside=token_nside,
            in_channels=in_channels,
            num_outputs=num_outputs,
            transformer_kwargs=transformer_kwargs,
            jit_compile_body=jit_compile_body,
            head_dropout_rate=head_dropout_rate,
            input_norm=input_norm,
            masked_attention=masked_attention,
            spmm_backend=spmm_backend,
        )

    def call(self, maps, training=False):
        return self.map_encoder(maps, training=training)
