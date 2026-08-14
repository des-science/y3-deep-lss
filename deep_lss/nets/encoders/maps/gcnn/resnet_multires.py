# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created July 2026
Author: Arne Thomsen

Multi-resolution DeepSphere/ResNet map branch — the GCNN analogue of the transformer's
``HealpixMultiResMapEncoder``. Used when ``network.smooth_nside`` requests mixed per-probe nsides
(combined probes, e.g. clustering at 256 with lensing at the native 512), which
``configuration.get_smoothing_kwargs`` turns into a ``{"split_probes": [...]}`` spec.

Structure (mirroring the transformer encoder as closely as the Sequential GCNN allows):

  1. ``PerProbeSmoothing`` — each probe smoothed with its own kernel at its own nside; the coarse
     probe is downsampled in-network and NEVER upsampled back.
  2. per-resolution-group ``EmpiricalInputNormalization`` (optional, own footprint mask per group).
  3. ``gcnn_pre`` — a stock ``HealpyGCNN`` running the fine probe (lensing @512) through the
     leading ResNet layers until the pooling stack reaches the coarse nside.
  4. injection fusion — the transformer idiom (``nested_transformer`` ``injection_proj`` /
     ``injection_fuse``): a linear Dense embeds the smoothed coarse channels to the fine stream's
     channel width, they are concatenated, and a second linear Dense fuses back to that width.
  5. ``gcnn_post`` — a second stock ``HealpyGCNN`` at the coarse nside running the remaining
     layers (and, for maps-only networks, the regression head, which HealpyGCNN passes through).

``HealpyGCNN`` is a ``keras.Sequential`` with the sphere graphs baked in at construction, so the
mid-stack injection is realized by splitting the ResNet layer list into two GCNN segments rather
than by modifying deepsphere. Pixel-row alignment at the seam is guaranteed by construction — both
``HealpyGCNN._transform_indices`` (ud_grade parent set) and ``configuration.get_smooth_nside_indices``
(unique NEST parents) produce the ascending parent set of the common footprint — and asserted at
build time.
"""

import numpy as np
import tensorflow as tf

from deepsphere import HealpyGCNN
from deepsphere import healpy_layers

from msfm.utils import logger

from deep_lss.nets.encoders.maps.gcnn.resnet import _CONV_HELPERS
from deep_lss.nets.encoders.maps.multires import MultiResEncoderMixin
from deep_lss.nets.layers.maps.readout import assemble_scale_taps, forward_with_pool_taps
from deep_lss.utils.configuration import get_smooth_nside_indices

LOGGER = logger.get_logger(__file__)

# layers that change the nside during a forward pass (the set HealpyGCNN tracks indices for)
_NSIDE_DOWN_LAYERS = (healpy_layers.HealpyPool, healpy_layers.HealpyPseudoConv, healpy_layers.Healpy_ViT)
_NSIDE_UP_LAYERS = (healpy_layers.HealpyPseudoConv_Transpose,)


def split_layers_at_nside(layers, nside_in, initial_Fin, injection_nside):
    """Split a ResNet layer list after the first layer at which the pooled nside == injection_nside.

    Walks the layer list with the same nside and Fin bookkeeping as ``HealpyGCNN``'s build loop, so
    the returned ``Fin_at_split`` is exactly the feature count the second GCNN segment must be built
    with (``initial_Fin`` drives the sparse-matmul split counts there).

    Args:
        layers (list): DeepSphere/keras layer list (e.g. ``ResNetLayers.get_conv_layers()``).
        nside_in (int): nside of the input maps (the fine probe's nside).
        initial_Fin (int): number of input channels of the fine probe.
        injection_nside (int): nside at which the coarse probe is injected.

    Returns:
        tuple: ``(pre_layers, post_layers, Fin_at_split)``.
    """
    if nside_in <= injection_nside:
        raise ValueError(f"injection nside {injection_nside} must be below the input nside {nside_in}")

    current_nside = nside_in
    current_Fin = initial_Fin
    for i, layer in enumerate(layers):
        if isinstance(layer, _NSIDE_DOWN_LAYERS):
            current_nside //= 2**layer.p
        elif isinstance(layer, _NSIDE_UP_LAYERS):
            current_nside *= 2**layer.p
        # mirror HealpyGCNN's Fin bookkeeping: only layers exposing a (non-None) Fout update it
        fout = getattr(layer, "Fout", None)
        if fout is not None:
            current_Fin = fout
        if current_nside == injection_nside:
            return layers[: i + 1], layers[i + 1 :], current_Fin
        if current_nside < injection_nside:
            break
    raise ValueError(
        f"injection nside {injection_nside} is never reached exactly by the pooling stack starting "
        f"at nside {nside_in} (check pool_layers/conv_layers against smooth_nside)"
    )


class ResNetMultiResEncoder(MultiResEncoderMixin, tf.keras.Model):
    """Multi-resolution DeepSphere/ResNet map branch (see the module docstring).

    Exposes the same interface as the transformer map encoders (``HealpixMapEncoder``), with the
    smoothing/grouping/input-norm plumbing provided by the shared ``MultiResEncoderMixin``:

      - ``call(maps, training)``          -> ``(B, n_pix_out, n_ch)`` conv features when built from
        ``ResNetLayers.get_conv_layers()`` (maps+cls composite), or ``(B, out_features)`` when built
        from ``ResNetLayers.get_layers()`` (maps-only, head included).
      - ``smooth_groups(maps, training)`` -> list of per-resolution-group smoothed maps (fp32).
      - ``masks`` (property)              -> matching list of per-group footprint masks.
      - ``load_input_norm_stats(stats)``  -> load a list of ``(mean, inv_std)`` into the group layers.

    Args:
        smoothing_kwargs (dict): the ``{"split_probes": [...]}`` spec from
            ``configuration.get_smoothing_kwargs`` (per-probe kwargs already carry the effective
            ``max_batch_size`` and the nside-scaled ``white_noise_sigma``).
        layers (list): ResNet layer list built WITHOUT smoothing/input-norm
            (``ResNetLayers(smoothing_kwargs=None, smoothing_external=True, ...)``).
        nside (int): network input nside (the fine probe's nside, from ``resolve_smooth_nside``).
        n_neighbors (int): HealpyGCNN graph neighbors (``network.n_neighbors``).
        max_batch_size (int): effective batch size for the GCNN sparse-matmul splits.
        input_norm (bool, optional): per-group ``EmpiricalInputNormalization`` after the smoothing,
            same placement and checkpoint-lineage caveats as the transformer encoders.
        spmm_backend (str, optional): sparse-matmul backend for the graph convolutions in both GCNN
            segments AND for the per-probe ``PerProbeSmoothing`` kernels ("coo"/"csr"/"gather"; see
            deepsphere.utils.make_spmm_operator). Defaults to "csr". ``PerProbeSmoothing`` lives in
            deep_lss.nets.layers.maps.smoothing and re-exports deepsphere's ``HealpySmoothing``,
            which the transformer path shares (also at the "csr" app default).
        fusion (str, optional): how the injected coarse (clustering) channels combine with the pooled
            fine (lensing) stream at the seam. "concat" (default) is ``injection_fuse(concat([x, inj]))``;
            "bilinear" additionally feeds the elementwise product ``x*inj`` into the concat, handing the
            trunk the local cross statistic (galaxy-galaxy lensing is a product of the two aligned fields)
            directly. Separate checkpoint lineage from "concat" only via ``injection_fuse``'s input width,
            which is inferred lazily — but treat it as its own lineage anyway (retrain).
        injection_conv_layers (int, optional): number of channel-preserving graph convolutions (+ LayerNorm)
            to run at the injection nside on the FUSED both-probe stream, right after the fusion Dense and
            before the rest of ``gcnn_post`` (i.e. prepended to ``post_layers``). Motivation: in the default
            schedule the pooling stack precedes every real graph conv, so the first genuine convolution runs
            only at the coarsest nside (e.g. 64) — no equivariant conv ever touches the finest, most
            non-Gaussian scales, and the two probes are combined only by a pointwise Dense at the seam. These
            convs extract non-Gaussian features at the injection nside (256) BEFORE the field is pooled down,
            AND — because they act on the fused lensing+clustering stream — do content-dependent CROSS-PROBE
            mixing at high resolution (the combined-probe gap bench_v4 diagnosed). Orientation-preserving
            (real graph conv, keeps the NEST child-order gauge), unlike a symmetrized pool. A conv at nside
            256 is compute-heavy, so size ``n_steps`` to the measured throughput. Defaults to 0 (off).
        injection_conv_kwargs (dict, optional): build kwargs for the ``injection_conv_layers`` — ``poly_degree``
            (Chebyshev/graph-conv order K, default 5) and ``conv_type`` ("cheby"/"mono"/"bernstein", default
            "cheby"). Ignored when ``injection_conv_layers == 0``. Defaults to None.
        fusion_width (int, optional): output width of ``injection_fuse``, i.e. the width of the fused
            both-probe stream entering ``gcnn_post``. ``None`` (default) keeps the historical behaviour,
            ``split_Fin`` — the fine stream's width at the seam, which equals ``base_channels`` in the
            standard schedule. Widening it costs one wider Dense plus wider ``injection_conv_layers``
            (channel-preserving at this width, so they scale with it). Separate checkpoint lineage.

            CAUTION — this was added to test a "the seam is a pinch" hypothesis that did NOT hold up.
            The coarse stream carries only 4 channels, and ``injection_proj`` is pointwise, so its
            output spans a <=4-dimensional subspace: the concat entering ``injection_fuse`` has
            effective per-pixel dimension <= ``split_Fin`` + 4, not ``2 * split_Fin``. Fusing that to
            ``split_Fin`` therefore drops at most four directions, and ``post_layers[0]`` is a
            ``HealpyPseudoConv`` emitting only 128 channels anyway. Any width beyond roughly
            ``split_Fin`` + 4 adds no cross-probe information. Kept because it is correct and inert
            unset; if used, prefer a modest value and pair it with ``fuse_act``. Defaults to None.
        fuse_act (str, optional): activation applied to ``injection_fuse``'s output (followed by a
            LayerNorm), e.g. ``"relu"``. ``None`` (default) keeps the historical behaviour.

            WHY THIS EXISTS: without it the seam is entirely LINEAR. ``injection_fuse`` is a Dense with
            no activation and ``post_layers[0]`` is a ``HealpyPseudoConv`` whose relu is applied after
            its own linear op, so fuse and the following pseudo-conv compose into ONE linear map and
            concat fusion contributes no multiplicative cross-probe interaction at all — consistent with
            ``fusion="bilinear"`` (which adds ``x*inj`` explicitly) having measured a wash. This knob is
            deliberately the ``(activation, LayerNorm)`` pair that ``injection_conv_layers`` attaches to
            its graph conv, MINUS the conv, so the two differ by exactly the spatial mixing. It is not
            parameter-matched to a graph conv: it isolates the nonlinearity, not the capacity.
            Separate checkpoint lineage (adds the LayerNorm's weights). Defaults to None.
    """

    def __init__(
        self,
        smoothing_kwargs,
        layers,
        nside,
        n_neighbors,
        max_batch_size,
        input_norm=False,
        spmm_backend="csr",
        fusion="concat",
        injection_conv_layers=0,
        injection_conv_kwargs=None,
        fusion_width=None,
        fuse_act=None,
    ):
        super().__init__()
        if fusion not in ("concat", "bilinear"):
            raise ValueError(f"fusion must be 'concat' or 'bilinear', got {fusion!r}")
        if fusion_width is not None and (not isinstance(fusion_width, int) or fusion_width <= 0):
            raise ValueError(f"fusion_width must be a positive int or None, got {fusion_width!r}")
        if fuse_act is not None:
            # resolve eagerly: an unknown name would otherwise fail deep inside the first call
            tf.keras.activations.get(fuse_act)
        if not isinstance(injection_conv_layers, int) or injection_conv_layers < 0:
            raise ValueError(f"injection_conv_layers must be a non-negative int, got {injection_conv_layers!r}")
        self.fusion = fusion

        # per-probe fp32 smoothing + grouping by nside (finest first) — shared mixin plumbing;
        # the fine group is the GCNN input, the coarse group is injected where the pooling stack
        # reaches its nside
        groups = self._init_smoothing_and_groups(smoothing_kwargs, spmm_backend=spmm_backend)
        if len(groups) != 2:
            raise NotImplementedError(
                f"ResNetMultiResEncoder supports exactly one fine + one coarse resolution group "
                f"(lensing@native + clustering@smooth_nside), got {len(groups)} groups: "
                f"{[(g['nside'], g['n_channels']) for g in groups]}. Generalize the layer split "
                "into one GCNN segment per group if ever needed."
            )
        fine, coarse = groups
        if fine["nside"] != nside:
            raise ValueError(
                f"finest probe nside {fine['nside']} != network input nside {nside}; the fine "
                "group must be at the nside the network and pipeline run at."
            )

        pre_layers, post_layers, split_Fin = split_layers_at_nside(layers, nside, fine["n_channels"], coarse["nside"])

        # row alignment at the seam: gcnn_pre pools its pixel axis with HealpyGCNN._transform_indices
        # (ascending ud_grade parent set of the fine footprint), which must equal the coarse spec's
        # footprint (ascending unique NEST parents of the same common footprint) — assert, not trust
        expected_coarse, _ = get_smooth_nside_indices(np.asarray(fine["indices"]), nside, coarse["nside"])
        if not np.array_equal(expected_coarse, np.asarray(coarse["indices"])):
            raise ValueError(
                "coarse-probe footprint is not the parent set of the fine footprint — the pooled "
                "fine stream and the injected coarse channels would be row-misaligned"
            )

        self.gcnn_pre = HealpyGCNN(
            nside=nside,
            indices=fine["indices"],
            layers=pre_layers,
            n_neighbors=n_neighbors,
            max_batch_size=max_batch_size,
            initial_Fin=fine["n_channels"],
            spmm_backend=spmm_backend,
        )
        # injection fusion, transformer idiom (nested_transformer injection_proj/injection_fuse):
        # linear Dense embed of the coarse channels, concat with the fine stream, linear Dense fuse
        # back to the fine stream's width — applied before all layers at the coarse nside and below
        # the pinch: every bit of cross-probe information passes through injection_fuse's output width.
        # It defaults to the fine stream's width at the seam (split_Fin = base_channels), which is 8x
        # narrower than the trunk it feeds; fusion_width widens it. gcnn_post is then built at the wider
        # Fin, and post_layers[0] is a strided HealpyPseudoConv whose Dense infers its input width
        # lazily, so nothing downstream needs changing. Separate checkpoint lineage.
        fused_Fin = fusion_width or split_Fin
        self.injection_proj = tf.keras.layers.Dense(split_Fin, name="injection_proj")
        self.injection_fuse = tf.keras.layers.Dense(fused_Fin, name="injection_fuse")

        # Optional NONLINEARITY at the seam. Without it the fusion is one composite LINEAR map:
        # injection_fuse is a Dense with no activation and post_layers[0] is a HealpyPseudoConv whose
        # relu is applied after its own linear op, so concat fusion contributes no multiplicative
        # cross-probe interaction at all. This is deliberately the (activation + LayerNorm) pair that
        # `injection_conv_layers` attaches to its graph conv, MINUS the conv itself, so the two arms
        # differ by exactly the spatial mixing and a positive injection_conv result can be attributed.
        # It is NOT parameter-matched to a graph conv (which also adds K*Fin*Fout weights) -- it isolates
        # the nonlinearity, not the capacity. Applied in call(), not prepended to post_layers, because a
        # bare activation is not a layer HealpyGCNN's build loop knows how to walk.
        self.fuse_act = tf.keras.layers.Activation(fuse_act, name="fuse_act") if fuse_act is not None else None
        self.fuse_act_norm = (
            tf.keras.layers.LayerNormalization(axis=-1, name="fuse_act_norm") if fuse_act is not None else None
        )

        # optional channel-preserving graph conv(s) at the injection nside on the FUSED both-probe stream,
        # prepended to gcnn_post so they run right after fusion, at the coarse nside (256), before any further
        # pooling. HealpyGCNN builds the sphere graph at coarse["nside"] for these Chebyshev helpers; the
        # LayerNorms fall through its passthrough branch. Channel-preserving at split_Fin (the fused width).
        if injection_conv_layers > 0:
            icw = injection_conv_kwargs or {}
            conv_helper = _CONV_HELPERS[icw.get("conv_type", "cheby")]
            K = icw.get("poly_degree", 5)
            injection_convs = []
            for _ in range(injection_conv_layers):
                injection_convs.append(conv_helper(K=K, Fout=fused_Fin, activation=tf.nn.relu))
                injection_convs.append(tf.keras.layers.LayerNormalization(axis=-1))
            post_layers = injection_convs + post_layers

        self.gcnn_post = HealpyGCNN(
            nside=coarse["nside"],
            indices=coarse["indices"],
            layers=post_layers,
            n_neighbors=n_neighbors,
            max_batch_size=max_batch_size,
            initial_Fin=fused_Fin,
            spmm_backend=spmm_backend,
        )

        # per-group input normalization (shared mixin plumbing, own mask per group)
        self._init_group_input_norm(input_norm)

        LOGGER.warning(
            f"ResNetMultiResEncoder: fine group nside={nside} ({fine['n_channels']} ch, "
            f"{len(pre_layers)} layers), coarse group nside={coarse['nside']} "
            f"({coarse['n_channels']} ch) injected at {split_Fin} channels, fused to {fused_Fin} "
            f"({len(post_layers)} layers after the fusion), fusion={fusion}, "
            f"fuse_act={fuse_act or 'None (LINEAR seam)'}, "
            f"injection_conv_layers={injection_conv_layers}, input_norm={input_norm}"
        )

    def call(self, maps, training=False, return_taps=False):
        """Forward pass.

        Args:
            maps (tf.Tensor): input maps for every probe group.
            training (bool): Keras training flag.
            return_taps (bool): also return the multi-scale readout's tap tensors, one per
                resolution — the FUSED SEAM first, then the output of each downsampling stage in
                ``gcnn_post``, then the trunk output. The seam is the finest tap because it is the
                first point at which both probes are present; anything finer exists only in the
                single-probe ``gcnn_pre`` stream and is not a cross-probe scale. The last
                downsampling tap is dropped by ``assemble_scale_taps`` because the residual body
                follows it at the same nside and width. Consumed by
                ``ResNetMapsPlusCLSNetwork(map_pool_multiscale=True)``.

        Returns:
            tf.Tensor: conv features ``(B, n_pix_reduced, n_ch)``, or ``(features, taps)`` when
            ``return_taps`` is set.
        """
        group_tensors = self.smooth_groups(maps, training=training)
        if self.input_norms is not None:
            group_tensors = [norm(t) for norm, t in zip(self.input_norms, group_tensors)]
        fine_t, coarse_t = group_tensors

        x = self.gcnn_pre(fine_t, training=training)  # (B, P_coarse, split_Fin)
        inj = self.injection_proj(coarse_t)  # (B, P_coarse, split_Fin)
        # fuse the pooled lensing stream (x) with the injected clustering channels (inj), both aligned
        # at the coarse nside. "bilinear" also feeds x*inj so the trunk gets the local cross statistic
        # (galaxy-galaxy lensing ~ product of the two fields at the same position) directly; injection_fuse
        # infers its input width lazily, so the wider concat needs no construction change.
        parts = [x, inj, x * inj] if self.fusion == "bilinear" else [x, inj]
        x = self.injection_fuse(tf.concat(parts, axis=-1))  # (B, P_coarse, fused_Fin)
        if self.fuse_act is not None:
            # the seam's only nonlinearity when set; without it fuse and the next pseudo-conv compose
            # into a single linear map (see __init__)
            x = self.fuse_act_norm(self.fuse_act(x), training=training)
        if not return_taps:
            return self.gcnn_post(x, training=training)

        seam = x
        out, pool_taps = forward_with_pool_taps(self.gcnn_post, x, training=training)
        return out, [seam] + assemble_scale_taps(pool_taps, out)
