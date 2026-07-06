from __future__ import annotations

import healpy as hp
import tensorflow as tf

from .nested_transfomer import NestedHierarchicalLocalWindowTransformer


class HealpixNestedHierarchicalLocalWindowTransformer(
    NestedHierarchicalLocalWindowTransformer
):
    def __init__(self, num_pixels, nside, nside_down, in_channels, **kwargs):
        if nside <= nside_down:
            raise ValueError("nside must be greater than nside_down")

        num_nested_levels = int(hp.nside2order(nside) - hp.nside2order(nside_down))

        # Number of fine nside pixels inside each nside_down top-level token.
        num_pixels_per_top_level_token = hp.nside2npix(nside) // hp.nside2npix(
            nside_down
        )
        if num_pixels % num_pixels_per_top_level_token != 0:
            raise ValueError(
                f"Cannot split {num_pixels} pixels into "
                f"{num_pixels_per_top_level_token} top-level tokens"
            )

        num_top_level_tokens = num_pixels // num_pixels_per_top_level_token
        nested_shape = (4,) * num_nested_levels
        full_nested_shape = (in_channels, num_top_level_tokens, *nested_shape)

        super().__init__(
            num_nested_levels=num_nested_levels,
            in_channels=in_channels,
            **kwargs,
        )

        self.nside = nside
        self.nside_down = nside_down
        self.num_pixels = num_pixels
        self.nested_shape = full_nested_shape

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
