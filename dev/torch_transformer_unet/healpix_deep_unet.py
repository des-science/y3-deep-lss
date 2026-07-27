import logging

import healpy as hp
import torch

from .nested_deep_unet import DeepNestedHierarchicalUNet

LOGGER = logging.getLogger(__name__)


class HealpixDeepNestedUNet(DeepNestedHierarchicalUNet):
    """HEALPix wrapper around :class:`DeepNestedHierarchicalUNet` for dense regression.

    Translates a flat ``(B, P, C)`` pipeline batch of nested-ordered HEALPix pixels into the
    ``(B, C, N, 4, ..., 4)`` nested tensor the core consumes, runs the U-Net, and translates
    the dense ``(B, C, N, 4, ..., 4)`` prediction back to a flat ``(B, P, C)`` map at the same
    nside. The output is the input map plus a learned residual correction — same shape in and
    out — which is what the PM -> N-body correction task needs.

    The fine map lives at ``nside``; each ``nside_down`` coarse pixel becomes one top-level
    token holding ``(nside / nside_down) ** 2`` fine pixels, arranged as
    ``num_nested_levels = order(nside) - order(nside_down)`` size-4 nested axes (NESTED /
    child-major HEALPix ordering). ``P = num_pixels`` need not be the full sphere — any
    footprint that is a whole number of top-level tokens works.

    All other keyword arguments (``base_embed_dim``, ``growth``, ``stem_levels``,
    ``local_blocks_per_level``, ``global_blocks``, ``layerscale_init``, ...) pass straight
    through to the core. The number of output channels equals ``in_channels`` (residual
    correction), so there is no separate output-size argument.
    """

    def __init__(self, num_pixels, nside, nside_down, in_channels, **kwargs):
        if nside <= nside_down:
            raise ValueError(f"nside ({nside}) must be greater than nside_down ({nside_down}).")

        num_nested_levels = int(hp.nside2order(nside) - hp.nside2order(nside_down))

        # Number of fine nside pixels inside each nside_down top-level token.
        num_pixels_per_top_level_token = hp.nside2npix(nside) // hp.nside2npix(nside_down)
        if num_pixels % num_pixels_per_top_level_token != 0:
            raise ValueError(
                f"Cannot split {num_pixels} pixels into top-level tokens of "
                f"{num_pixels_per_top_level_token} pixels each."
            )
        num_top_level_tokens = num_pixels // num_pixels_per_top_level_token

        super().__init__(
            num_nested_levels=num_nested_levels,
            in_channels=in_channels,
            **kwargs,
        )

        self.nside = nside
        self.nside_down = nside_down
        self.num_pixels = num_pixels
        self.num_top_level_tokens = num_top_level_tokens
        # (C, N, 4, ..., 4) nested shape used to reshape each flat batch in forward.
        self.nested_shape = (in_channels, num_top_level_tokens, *((4,) * num_nested_levels))

        LOGGER.info(
            "HealpixDeepNestedUNet: nside=%d (npix=%d) -> token nside=%d (npix=%d), %d nested "
            "levels; footprint %d pixels -> %d top-level tokens (%d pixels/token); "
            "stem_levels=%d, body_levels=%d, channel_dims=%s; dense residual output.",
            nside, hp.nside2npix(nside), nside_down, hp.nside2npix(nside_down),
            num_nested_levels, num_pixels, num_top_level_tokens,
            num_pixels_per_top_level_token, self.stem_levels, self.body_levels,
            list(self.channel_dims),
        )

    def batch_flat_to_nested(self, batch: torch.Tensor) -> torch.Tensor:
        """Convert a pipeline batch shaped ``(B, P, C)`` to nested ``(B, C, N, 4, ..., 4)``."""
        if batch.ndim != 3:
            raise ValueError(f"Expected batch shaped (B, P, C), got shape {tuple(batch.shape)}.")
        if batch.shape[1] != self.num_pixels:
            raise ValueError(f"Expected {self.num_pixels} pixels, got {batch.shape[1]}.")
        if batch.shape[2] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {batch.shape[2]}.")
        # (B, P, C) -> (B, C, P) -> (B, C, N, 4, ..., 4)
        return batch.movedim(2, 1).contiguous().reshape(batch.shape[0], *self.nested_shape)

    def batch_nested_to_flat(self, batch: torch.Tensor) -> torch.Tensor:
        """Convert a nested ``(B, C, N, 4, ..., 4)`` prediction back to flat ``(B, P, C)``."""
        expected_ndim = 3 + self.num_nested_levels
        if batch.ndim != expected_ndim:
            raise ValueError(
                f"Expected nested batch with {expected_ndim} dims (B, C, N, 4, ..., 4), "
                f"got shape {tuple(batch.shape)}."
            )
        # (B, C, N, 4, ..., 4) -> (B, C, P) -> (B, P, C); inverse of batch_flat_to_nested.
        flat = batch.reshape(batch.shape[0], self.in_channels, self.num_pixels)
        return flat.movedim(1, 2).contiguous()

    def forward(self, x):
        return self.batch_nested_to_flat(super().forward(self.batch_flat_to_nested(x)))
