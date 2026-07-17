from __future__ import annotations

import healpy as hp
import numpy as np
import tensorflow as tf
from msfm.utils import logger

from .nested_transformer import NestedHierarchicalLocalWindowTransformer

LOGGER = logger.get_logger(__file__)


def window_mean_squared_distances(
    nside,
    num_nested_levels,
    window_levels,
    ref_windows=64,
    seed=0,
):
    """Per-stage tables of normalized squared geodesic distances between window tokens.

    Returns a list (length ``num_nested_levels``) of (S, S) float32 arrays with
    ``S = 4 ** min(window_levels, num_nested_levels - level)``, the number of tokens in
    a local attention window at that stage (each patch merge halves the stage nside).
    Entry (i, j) is ``(d_ij / d_max) ** 2`` where ``d_ij`` is the geodesic distance
    between window tokens i and j, averaged over ``ref_windows`` representative nested
    windows to smooth pole / base-pixel-boundary distortion, and ``d_max`` is the
    stage's maximum averaged distance.

    The token order (nested children ``a*S .. (a+1)*S - 1``) matches the row-major
    ``tf.reshape(x, [-1, S, D])`` flattening in NestedLocalWindowBlock, so the tables
    line up with the attention logits without reindexing.
    """
    rng = np.random.default_rng(seed)
    tables = []
    for level in range(num_nested_levels):
        nside_stage = nside >> level
        levels_used = min(window_levels, num_nested_levels - level)
        sequence_length = 4 ** levels_used

        npix = hp.nside2npix(nside_stage)
        n_windows = npix // sequence_length
        anchors = rng.choice(
            n_windows, size=min(ref_windows, n_windows), replace=False
        )

        acc = np.zeros((sequence_length, sequence_length), dtype=np.float64)
        for anchor in anchors:
            ipix = np.arange(anchor * sequence_length, (anchor + 1) * sequence_length)
            vec = np.stack(hp.pix2vec(nside_stage, ipix, nest=True), axis=-1)  # (S, 3)
            cos_dist = np.clip(vec @ vec.T, -1.0, 1.0)
            acc += np.arccos(cos_dist)
        dist = acc / len(anchors)

        tables.append((dist / dist.max()).astype(np.float32) ** 2)
    return tables


def window_binned_distances(
    nside,
    num_nested_levels,
    window_levels,
    num_bins=16,
    ref_windows=64,
    seed=0,
):
    """Per-stage distance-bin tables for the binned relative attention bias.

    Returns a list (length ``num_nested_levels``) of ``(bin_idx, bin_centers)`` tuples:
    ``bin_idx`` is an (S, S) int32 table assigning each window-token pair to a geodesic
    distance bin, and ``bin_centers`` is the mean normalized squared distance per bin
    (used for the RBF init of GeodesicBinnedBiasAttention — same normalized-d^2 units as
    the tables of ``window_mean_squared_distances``, which this reuses).

    Bin 0 is the diagonal (d = 0); the off-diagonal entries are quantile-binned into at
    most ``num_bins - 1`` bins. Small late-stage tables (S = 16, 4) have fewer distinct
    values than bins, in which case each distinct value gets its own bin. Empty bins are
    dropped and the indices renumbered contiguously, so ``bin_idx.max() ==
    len(bin_centers) - 1`` always holds.
    """
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2 (diagonal bin + at least one off-diagonal bin)")

    tables = window_mean_squared_distances(
        nside=nside,
        num_nested_levels=num_nested_levels,
        window_levels=window_levels,
        ref_windows=ref_windows,
        seed=seed,
    )
    out = []
    for d2 in tables:
        S = d2.shape[0]
        off_mask = ~np.eye(S, dtype=bool)
        off = d2[off_mask]
        uniq = np.unique(off)
        n_target = min(num_bins - 1, len(uniq))
        if len(uniq) == n_target:
            # one bin per distinct value
            off_bins = np.searchsorted(uniq, off)
        else:
            edges = np.unique(np.quantile(off, np.linspace(0.0, 1.0, n_target + 1)))
            off_bins = np.digitize(off, edges[1:-1], right=False)

        bin_idx = np.zeros((S, S), dtype=np.int64)
        bin_idx[off_mask] = 1 + off_bins
        # drop any empty bins defensively and renumber contiguously
        used, bin_idx = np.unique(bin_idx, return_inverse=True)
        bin_idx = bin_idx.reshape(S, S)

        bin_centers = np.array(
            [d2[bin_idx == k].mean() for k in range(len(used))], dtype=np.float32
        )
        bin_centers[0] = 0.0
        out.append((bin_idx.astype(np.int32), bin_centers))
    return out


class HealpixNestedHierarchicalLocalWindowTransformer(
    NestedHierarchicalLocalWindowTransformer
):
    def __init__(
        self,
        num_pixels,
        nside,
        nside_down,
        in_channels,
        pos_encoding=None,
        pos_encoding_kwargs=None,
        bias_ref_windows=64,
        injections=None,
        **kwargs,
    ):
        if nside <= nside_down:
            raise ValueError("nside must be greater than nside_down")

        num_nested_levels = int(hp.nside2order(nside) - hp.nside2order(nside_down))

        # pos_encoding: positional encoding for the local window attention.
        #   None              — plain, position-free local attention.
        #   "geodesic"        — distance-kernel bias in every local window block (see
        #                       GeodesicKernelAttention). The tables depend only on nside
        #                       and the window layout, so they are precomputed here and
        #                       passed down as local_dist_sq. With a patchified stem
        #                       (stem_levels) the hierarchy starts that many levels
        #                       coarser, so the tables describe the body geometry.
        #   "geodesic_binned" — distance-binned learnable relative bias (see
        #                       GeodesicBinnedBiasAttention), passed down as
        #                       local_dist_bins.
        # pos_encoding_kwargs: options of the chosen encoding —
        #   coeff_init (both):           RBF init of the bias, b = coeff_init * d^2.
        #                                Defaults: 0.0 for "geodesic" (legacy behavior;
        #                                known not to bootstrap — bench_t7 symmetric),
        #                                -1.0 for "geodesic_binned" (engaged at step 0).
        #   num_bins ("geodesic_binned"): max distance bins per stage (default 16).
        stem_levels = kwargs.get("stem_levels", 0)
        pe_kwargs = dict(pos_encoding_kwargs or {})
        if pos_encoding is None and pe_kwargs:
            raise ValueError("pos_encoding_kwargs given but pos_encoding is None.")
        if pos_encoding == "geodesic":
            kwargs["local_dist_sq"] = window_mean_squared_distances(
                nside=nside >> stem_levels,
                num_nested_levels=num_nested_levels - stem_levels,
                window_levels=kwargs.get("window_levels", 3),
                ref_windows=bias_ref_windows,
            )
            kwargs["pos_coeff_init"] = float(pe_kwargs.pop("coeff_init", 0.0))
        elif pos_encoding == "geodesic_binned":
            kwargs["local_dist_bins"] = window_binned_distances(
                nside=nside >> stem_levels,
                num_nested_levels=num_nested_levels - stem_levels,
                window_levels=kwargs.get("window_levels", 3),
                num_bins=int(pe_kwargs.pop("num_bins", 16)),
                ref_windows=bias_ref_windows,
            )
            kwargs["pos_coeff_init"] = float(pe_kwargs.pop("coeff_init", -1.0))
        elif pos_encoding is not None:
            raise ValueError(
                f"pos_encoding must be None, 'geodesic' or 'geodesic_binned', "
                f"got {pos_encoding!r}."
            )
        if pe_kwargs:
            raise ValueError(
                f"Unknown pos_encoding_kwargs for pos_encoding={pos_encoding!r}: "
                f"{sorted(pe_kwargs)}"
            )

        # Number of fine nside pixels inside each nside_down top-level token.
        num_pixels_per_top_level_token = hp.nside2npix(nside) // hp.nside2npix(
            nside_down
        )
        if num_pixels % num_pixels_per_top_level_token != 0:
            raise ValueError(
                f"Cannot split {num_pixels} pixels into "
                f"{num_pixels_per_top_level_token} top-level tokens"
            )

        # token_valid (masked attention) flows through to the base class; here we can
        # additionally pin its length to the known pixel count.
        token_valid = kwargs.get("token_valid")
        if token_valid is not None and len(token_valid) != num_pixels:
            raise ValueError(
                f"token_valid has {len(token_valid)} entries, expected num_pixels = "
                f"{num_pixels}."
            )

        num_top_level_tokens = num_pixels // num_pixels_per_top_level_token
        nested_shape = (4,) * num_nested_levels
        full_nested_shape = (in_channels, num_top_level_tokens, *nested_shape)

        # injections: secondary inputs whose maps live at a coarser nside than the main input
        # (e.g. clustering @256 with a lensing @512 main input). Each {"nside", "in_channels"}
        # joins the hierarchy at the body level already running at that nside. Translate the
        # nside into the core's body-loop level and precompute the (C_inj, N, 4, ..., 4) nested
        # shape used to reshape each coarse flat input in call().
        injection_nested = {}
        core_injections = []
        for spec in injections or []:
            nside_inj = int(spec["nside"])
            if not (nside_down < nside_inj < nside):
                raise ValueError(
                    f"injection nside {nside_inj} must satisfy nside_down ({nside_down}) "
                    f"< nside_inj < nside ({nside})."
                )
            inject_nested_level = int(hp.nside2order(nside) - hp.nside2order(nside_inj))
            body_level = inject_nested_level - stem_levels
            if body_level < 1:
                raise ValueError(
                    f"injection nside {nside_inj} falls at or inside the patchified stem "
                    f"(stem_levels={stem_levels}); it must join a coarser body level."
                )
            coarse_nested_shape = (
                int(spec["in_channels"]),
                num_top_level_tokens,
                *((4,) * (num_nested_levels - inject_nested_level)),
            )
            # keyed by injection nside (what the encoder passes); value carries the core
            # body-loop level and the (C_inj, N, 4, ..., 4) shape used to reshape the flat input
            injection_nested[nside_inj] = (body_level, coarse_nested_shape)
            core_injections.append({"level": body_level, "in_channels": int(spec["in_channels"])})

        body_nsides = [nside >> level for level in range(stem_levels, num_nested_levels)]
        LOGGER.warning(
            f"HealpixNestedHierarchicalLocalWindowTransformer: nside={nside} "
            f"(npix={hp.nside2npix(nside)}) -> token nside={nside_down} "
            f"(npix={hp.nside2npix(nside_down)}), {num_nested_levels} nested levels, "
            f"footprint: {num_pixels} pixels -> {num_top_level_tokens} top-level "
            f"tokens ({num_pixels_per_top_level_token} pixels/token), local attention "
            f"stages at nsides {body_nsides}"
        )
        if pos_encoding == "geodesic":
            LOGGER.warning(
                f"Geodesic distance-kernel tables: {num_nested_levels - stem_levels} "
                f"stages starting at nside={nside >> stem_levels}"
                + (f" (shifted by stem_levels={stem_levels})" if stem_levels > 0 else "")
                + f", averaged over {bias_ref_windows} reference windows, "
                + f"kernel_coeff init {kwargs['pos_coeff_init']}"
            )
        elif pos_encoding == "geodesic_binned":
            LOGGER.warning(
                f"Geodesic binned-bias tables: {num_nested_levels - stem_levels} "
                f"stages starting at nside={nside >> stem_levels}"
                + (f" (shifted by stem_levels={stem_levels})" if stem_levels > 0 else "")
                + f", averaged over {bias_ref_windows} reference windows, per-stage "
                + f"bins {[len(centers) for _, centers in kwargs['local_dist_bins']]}, "
                + f"RBF init coeff {kwargs['pos_coeff_init']}"
            )

        super().__init__(
            num_nested_levels=num_nested_levels,
            in_channels=in_channels,
            injections=core_injections,
            **kwargs,
        )

        self.nside = nside
        self.nside_down = nside_down
        self.num_pixels = num_pixels
        self.nested_shape = full_nested_shape
        self.pos_encoding = pos_encoding
        self.pos_encoding_kwargs = pos_encoding_kwargs
        self.bias_ref_windows = bias_ref_windows
        # {injection nside: (body-loop level, (C_inj, N, 4, ..., 4) nested shape)}, for call()
        self._injection_nested = injection_nested

    def _flat_to_nested(self, batch: tf.Tensor, nested_shape) -> tf.Tensor:
        """Reshape a flat ``(B, P, C)`` batch to nested ``(B, C, N, 4, ..., 4)``.

        ``nested_shape`` is the target ``(C, N, 4, ..., 4)`` (channels first, then the N
        top-level tokens and the size-4 nested axes); the expected flat pixel count is
        ``P = N * 4^L`` and channel count ``C = nested_shape[0]``. Used for both the main
        input (``self.nested_shape``) and each coarser injection.
        """
        num_pixels = int(np.prod(nested_shape[1:]))
        in_channels = int(nested_shape[0])

        rank = batch.shape.rank
        if rank is not None and rank != 3:
            raise ValueError(f"Expected batch with shape (B, P, C), got {batch.shape}.")

        assertions = []
        if rank is None:
            assertions.append(
                tf.debugging.assert_equal(
                    tf.rank(batch),
                    3,
                    message="Expected batch with shape (B, P, C).",
                )
            )

        pixel_dim = batch.shape[1]
        if pixel_dim is not None and pixel_dim != num_pixels:
            raise ValueError(f"Expected {num_pixels} pixels, got {pixel_dim}.")
        if pixel_dim is None:
            assertions.append(
                tf.debugging.assert_equal(
                    tf.shape(batch)[1],
                    tf.cast(num_pixels, tf.shape(batch).dtype),
                    message=f"Expected {num_pixels} pixels.",
                )
            )

        channel_dim = batch.shape[2]
        if channel_dim is not None and channel_dim != in_channels:
            raise ValueError(f"Expected {in_channels} channels, got {channel_dim}.")
        if channel_dim is None:
            assertions.append(
                tf.debugging.assert_equal(
                    tf.shape(batch)[2],
                    tf.cast(in_channels, tf.shape(batch).dtype),
                    message=f"Expected {in_channels} channels.",
                )
            )

        if assertions:
            with tf.control_dependencies(assertions):
                batch = tf.identity(batch)

        # (B, P, C) -> (B, C, P)
        batch = tf.transpose(batch, perm=[0, 2, 1])

        # (B, C, P) -> (B, C, N, 4, 4, ..., 4)
        target_shape = tf.concat(
            [
                tf.reshape(tf.shape(batch)[0], [1]),
                tf.constant(nested_shape, dtype=tf.shape(batch).dtype),
            ],
            axis=0,
        )
        nested = tf.reshape(batch, target_shape)
        nested.set_shape([batch.shape[0], *nested_shape])
        return nested

    def batch_flat_to_nested(self, batch: tf.Tensor) -> tf.Tensor:
        """Convert pipeline batch shaped ``(B, P, C)`` to nested transformer input."""
        return self._flat_to_nested(batch, self.nested_shape)

    def call(self, x, training=None, injections=None):
        """``injections``: optional ``{injection nside: (B, P_c, C_c)}`` flat coarse inputs."""
        nested_injections = None
        if injections:
            nested_injections = {}
            for nside_inj, flat in injections.items():
                body_level, nested_shape = self._injection_nested[nside_inj]
                nested_injections[body_level] = self._flat_to_nested(flat, nested_shape)
        return super().call(
            self.batch_flat_to_nested(x), training=training, injections=nested_injections
        )
