from __future__ import annotations

import healpy as hp
import numpy as np
import tensorflow as tf
from msfm.utils import logger

from .nested_transfomer import NestedHierarchicalLocalWindowTransformer

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
        bias_ref_windows=64,
        **kwargs,
    ):
        if nside <= nside_down:
            raise ValueError("nside must be greater than nside_down")

        num_nested_levels = int(hp.nside2order(nside) - hp.nside2order(nside_down))

        # pos_encoding: positional encoding for the local window attention.
        #   None       — plain, position-free local attention.
        #   "geodesic" — distance-kernel bias in every local window block (see
        #                GeodesicKernelAttention). The tables depend only on nside and
        #                the window layout, so they are precomputed here and passed
        #                down as local_dist_sq. With a patchified stem (stem_levels)
        #                the hierarchy starts that many levels coarser, so the tables
        #                describe the body geometry.
        stem_levels = kwargs.get("stem_levels", 0)
        if pos_encoding == "geodesic":
            kwargs["local_dist_sq"] = window_mean_squared_distances(
                nside=nside >> stem_levels,
                num_nested_levels=num_nested_levels - stem_levels,
                window_levels=kwargs.get("window_levels", 3),
                ref_windows=bias_ref_windows,
            )
        elif pos_encoding is not None:
            raise ValueError(
                f"pos_encoding must be None or 'geodesic', got {pos_encoding!r}."
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
                + f", averaged over {bias_ref_windows} reference windows"
            )

        super().__init__(
            num_nested_levels=num_nested_levels,
            in_channels=in_channels,
            **kwargs,
        )

        self.nside = nside
        self.nside_down = nside_down
        self.num_pixels = num_pixels
        self.nested_shape = full_nested_shape
        self.pos_encoding = pos_encoding
        self.bias_ref_windows = bias_ref_windows

    def batch_flat_to_nested(self, batch: tf.Tensor) -> tf.Tensor:
        """Convert pipeline batch shaped ``(B, P, C)`` to nested transformer input."""
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
        if pixel_dim is not None and pixel_dim != self.num_pixels:
            raise ValueError(
                f"Expected {self.num_pixels} pixels, got {pixel_dim}."
            )
        if pixel_dim is None:
            assertions.append(
                tf.debugging.assert_equal(
                    tf.shape(batch)[1],
                    tf.cast(self.num_pixels, tf.shape(batch).dtype),
                    message=f"Expected {self.num_pixels} pixels.",
                )
            )

        channel_dim = batch.shape[2]
        if channel_dim is not None and channel_dim != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {channel_dim}."
            )
        if channel_dim is None:
            assertions.append(
                tf.debugging.assert_equal(
                    tf.shape(batch)[2],
                    tf.cast(self.in_channels, tf.shape(batch).dtype),
                    message=f"Expected {self.in_channels} channels.",
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
                tf.constant(self.nested_shape, dtype=tf.shape(batch).dtype),
            ],
            axis=0,
        )
        nested = tf.reshape(batch, target_shape)
        nested.set_shape([batch.shape[0], *self.nested_shape])
        return nested

    def call(self, x, training=None):
        return super().call(self.batch_flat_to_nested(x), training=training)
