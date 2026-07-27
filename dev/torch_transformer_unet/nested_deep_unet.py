"""Nested hierarchical local-window U-Net for dense same-nside HEALPix regression.

The network is an encoder / decoder ("U-Net") over the nested HEALPix hierarchy. The
encoder is the local-window transformer stack: local nested-window attention followed by
:class:`NestedPatchMerge4` down-steps that fold each finest size-4 axis into the channel
dimension, coarsening the map to the top-level tokens. A short global-attention bottleneck
mixes those tokens. The decoder mirrors the encoder: each merge is inverted by a
:class:`NestedPatchExpand4` up-step that splits a size-4 axis back out, the matching
encoder feature is brought in through a skip connection (:class:`NestedSkipFuse`), and more
local-window blocks refine the upsampled stream. The output is projected back to the input
channels at the original nside and added to the input as a **residual correction**.

The intended use is learning a small-scale correction that maps cheap particle-mesh (PM)
simulations onto expensive N-body simulations: the input map is returned corrected, same
shape in and out.

Stabilizers / options carried on every transformer block: pre-normalized residual branches
(default), DropPath / stochastic depth, per-branch LayerScale (off by default), and a
patchified stem (``stem_levels``) that folds the finest nested levels into the input
projection so the hierarchy starts coarser. Any footprint that is a whole number of
top-level tokens works — the map need not cover the full sphere.
"""

import torch
import torch.nn as nn

from .nested_transfomer import MLP, NestedPatchMerge4, make_channel_dims


class DropPath(nn.Module):
    """Drop residual paths per sample during training.

    The implementation follows stochastic depth: entire residual branches are
    randomly zeroed per batch item and surviving branches are rescaled by the
    keep probability so expected activations are preserved.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        if drop_prob < 0.0 or drop_prob >= 1.0:
            raise ValueError("drop_prob must satisfy 0 <= drop_prob < 1")
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class LayerScale(nn.Module):
    """Per-channel learnable scale on a residual branch (Touvron et al. 2021,
    "Going deeper with image transformers", arXiv:2103.17239).

    Initialised to a small value so each attention / MLP branch starts as a
    near-identity perturbation of the residual stream, which keeps deep stacks stable
    at initialisation. ``gamma`` then learns the useful per-channel branch magnitude.
    Pass ``init_value=None`` at the block level to disable it entirely.
    """

    def __init__(self, dim: int, init_value: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class DeepTransformerBlock(nn.Module):
    """Transformer block with configurable pre-norm and stochastic depth."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        drop_path: float = 0.0,
        pre_norm: bool = True,
        residual_dropout: float = 0.0,
        layerscale_init: float | None = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
        self.residual_dropout = nn.Dropout(residual_dropout) if residual_dropout > 0.0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # LayerScale on each residual branch (disabled when layerscale_init is None,
        # which reproduces the plain-residual behaviour exactly). Applied to the branch
        # output before residual_dropout / drop_path and the residual add.
        self.ls1 = LayerScale(dim, layerscale_init) if layerscale_init is not None else nn.Identity()
        self.ls2 = LayerScale(dim, layerscale_init) if layerscale_init is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            attn_input = self.norm1(x)
            attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
            x = x + self.drop_path(self.residual_dropout(self.ls1(attn_out)))
            x = x + self.drop_path(self.residual_dropout(self.ls2(self.mlp(self.norm2(x)))))
            return x

        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop_path(self.residual_dropout(self.ls1(attn_out))))
        x = self.norm2(x + self.drop_path(self.residual_dropout(self.ls2(self.mlp(x)))))
        return x


class DeepNestedLocalWindowBlock(nn.Module):
    """Local nested-window attention using :class:`DeepTransformerBlock`.

    Attends within each local nested window (the finest ``window_levels`` size-4 axes,
    a ``4 ** window_levels``-token sequence) and leaves the tensor shape unchanged, so it
    is used identically on the encoder down-path and the decoder up-path.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_levels: int = 3,
        mlp_ratio: int = 4,
        drop_path: float = 0.0,
        pre_norm: bool = True,
        residual_dropout: float = 0.0,
        layerscale_init: float | None = None,
    ):
        super().__init__()
        if window_levels < 1:
            raise ValueError("window_levels must be >= 1")
        self.window_levels = window_levels
        self.block = DeepTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            pre_norm=pre_norm,
            residual_dropout=residual_dropout,
            layerscale_init=layerscale_init,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_nested_levels = x.ndim - 3
        if num_nested_levels <= 0:
            raise ValueError("DeepNestedLocalWindowBlock needs at least one nested resolution dimension.")

        levels_used = min(self.window_levels, num_nested_levels)
        original_shape = x.shape
        dim = x.shape[-1]
        window_shape = x.shape[-levels_used - 1 : -1]

        for size in window_shape:
            if size != 4:
                raise ValueError("Every nested resolution dimension must have size 4.")

        sequence_length = 1
        for size in window_shape:
            sequence_length *= size

        x = x.contiguous().reshape(-1, sequence_length, dim)
        x = self.block(x)
        return x.reshape(original_shape)


class NestedPatchExpand4(nn.Module):
    """Inverse of :class:`NestedPatchMerge4`: split one nested resolution axis back out.

    Input:
        x: (B, N, 4, ..., 4, in_dim)

    Output:
        x: (B, N, 4, ..., 4, 4, out_dim)

    The decoder up-step. A single Linear maps ``in_dim -> 4 * out_dim`` and the result is
    reshaped into a new size-4 axis at position ``-2`` — exactly the axis
    :class:`NestedPatchMerge4` folded into the channel dimension on the way down (child-major
    / NESTED ordering), so merge and expand are consistent inverses. ``out_dim`` follows the
    encoder channel schedule in reverse.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.expansion = nn.Linear(in_dim, 4 * out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 3:
            raise ValueError("NestedPatchExpand4 needs at least the (B, N, in_dim) axes.")
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Expected last dimension {self.in_dim}, got {x.shape[-1]}.")

        # (..., in_dim) -> (..., 4 * out_dim) -> (..., 4, out_dim), the new child axis at -2.
        x = self.expansion(x)
        x = x.reshape(*x.shape[:-1], 4, self.out_dim)
        return self.norm(x)


class NestedSkipFuse(nn.Module):
    """Fuse a decoder up-step with the matching encoder skip feature.

    Both tensors share the shape ``(B, N, 4, ..., 4, dim)`` at the same resolution level.
    They are concatenated on the channel axis and a LayerNorm + Linear fuse the ``2 * dim``
    concat back to ``dim`` — the same concat-then-project idiom the patch merges use, so the
    decoder recovers small-scale detail that the encoder saw before it was merged away.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(2 * dim)
        self.fuse = nn.Linear(2 * dim, dim)

    def forward(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        if x_up.shape != x_skip.shape:
            raise ValueError(
                f"Skip fusion expects matching shapes, got up {tuple(x_up.shape)} vs "
                f"skip {tuple(x_skip.shape)}."
            )
        return self.fuse(self.norm(torch.cat([x_up, x_skip], dim=-1)))


class DeepNestedHierarchicalUNet(nn.Module):
    """Nested hierarchical local-window U-Net for dense same-nside regression.

    Input and output:
        x: (B, C, N, 4, 4, ..., 4)   ->   y: (B, C, N, 4, 4, ..., 4)   (same shape)

    where:
        B = batch size
        C = in_channels
        N = number of top-level tokens (any partial footprint; need not be the full sphere)
        M = num_nested_levels size-4 axes

    Structure:
        input projection (optionally a patchified stem over the finest stem_levels)
        -> encoder: [local nested attention -> patch merge] x body_levels   (save skips)
        -> bottleneck: global attention over the N top-level tokens
        -> decoder: [patch expand -> fuse encoder skip -> local nested attention] x body_levels
        -> output projection back to C channels at the input nside
        -> add to the input (residual correction)

    Hyperparameters — what to tune
    ------------------------------
    ``in_channels`` and ``num_nested_levels`` are fixed by the data (the HEALPix wrapper
    derives ``num_nested_levels`` from ``nside`` / ``nside_down``); everything below is a
    genuine tuning knob. Listed roughly in the order worth reaching for:

    Capacity (reach for these first):
      * ``base_embed_dim`` — channel width at the finest level; the primary capacity dial.
      * ``growth`` — how width scales toward the coarse bottleneck: ``"constant"`` keeps it
        flat, ``"double"``/``"full"`` widen it (more params where there are fewer tokens, the
        classic U-Net trade), ``"128"`` adds a fixed 128 per level. Every resulting width must
        stay divisible by ``num_heads``.

    Depth:
      * ``local_blocks_per_level`` — local-window transformer blocks per level, applied on
        *both* the encoder and decoder side.
      * ``global_blocks`` — global-attention blocks in the bottleneck (>= 1).

    Compute / resolution:
      * ``stem_levels`` — patchify the finest levels into the in/out projection instead of
        attending over them (see below); a compute lever when the finest levels are costly.
        The prediction is always emitted at the input nside.
      * ``window_levels`` — how many nested levels each local-window attention spans.

    Regularization / stability (leave off/at defaults unless it overfits or diverges):
      * ``drop_path_rate`` / ``drop_path_schedule`` — stochastic depth; ``None`` (default)
        disables it, which suits the shallow networks this is aimed at.
      * ``residual_dropout`` — dropout on residual branches.
      * ``layerscale_init`` — LayerScale; ``None`` (default) off, set e.g. ``1e-4`` to stabilize
        deeper stacks.
      * ``pre_norm``, ``mlp_ratio``, ``num_heads`` — architectural knobs; defaults are fine
        until you are squeezing the last few percent.

    A patchified stem (``stem_levels``) folds the finest nested levels into the input
    projection so the hierarchy runs over ``body_levels = num_nested_levels - stem_levels``
    levels; the output projection unfolds them symmetrically so the prediction lands back at
    the input nside.
    """

    def __init__(
        self,
        in_channels,
        num_nested_levels,
        base_embed_dim=64,
        growth="double",
        num_heads=4,
        window_levels=3,
        stem_levels=0,
        local_blocks_per_level=1,
        global_blocks=2,
        mlp_ratio=4,
        drop_path_rate=None,
        drop_path_schedule="linear",
        pre_norm=True,
        residual_dropout=0.0,
        layerscale_init=None,
    ):
        super().__init__()
        if num_nested_levels < 1:
            raise ValueError("num_nested_levels must be >= 1 (the U-Net needs a hierarchy to merge).")
        if local_blocks_per_level < 0:
            raise ValueError("local_blocks_per_level must be >= 0")
        if global_blocks < 1:
            raise ValueError("global_blocks must be >= 1")
        # drop_path_rate=None disables stochastic depth (DropPath is then a no-op); this is the
        # default because these correction networks are typically shallow.
        if drop_path_rate is not None and (drop_path_rate < 0.0 or drop_path_rate >= 1.0):
            raise ValueError("drop_path_rate must be None or satisfy 0 <= drop_path_rate < 1")
        if drop_path_schedule not in {"linear", "constant"}:
            raise ValueError("drop_path_schedule must be 'linear' or 'constant'")

        # stem_levels: patchified stem — the finest ``stem_levels`` nested levels are folded
        # into the input projection (one linear embed of 4**stem_levels fine tokens per patch,
        # child-major) and the transformer hierarchy starts that many levels coarser. The
        # output projection performs the inverse unfold, so the dense prediction is emitted at
        # the input nside regardless of stem_levels. stem_levels=0 keeps the per-finest-pixel
        # stem (and per-pixel output).
        if stem_levels < 0:
            raise ValueError("stem_levels must be >= 0")
        if stem_levels >= num_nested_levels:
            raise ValueError(
                f"stem_levels={stem_levels} must leave at least one nested level "
                f"(num_nested_levels={num_nested_levels})."
            )
        body_levels = num_nested_levels - stem_levels

        self.in_channels = in_channels
        self.num_nested_levels = num_nested_levels
        self.stem_levels = stem_levels
        self.body_levels = body_levels
        self.base_embed_dim = base_embed_dim
        self.growth = growth
        self.num_heads = num_heads
        self.window_levels = window_levels
        self.drop_path_rate = drop_path_rate
        self.drop_path_schedule = drop_path_schedule
        self.pre_norm = pre_norm
        self.residual_dropout = residual_dropout
        self.layerscale_init = layerscale_init

        # Channel dimensions of the hierarchy after the stem. Length is body_levels + 1;
        # channel_dims[0] is the finest body level, channel_dims[-1] the bottleneck.
        self.channel_dims = make_channel_dims(base_embed_dim, body_levels, growth)
        for dim in self.channel_dims:
            if dim % num_heads != 0:
                raise ValueError(f"Channel dimension {dim} must be divisible by num_heads={num_heads}.")

        # DropPath schedule spans the full encoder -> bottleneck -> decoder depth, consumed in
        # forward order so a "linear" schedule rises monotonically from input to output.
        # drop_path_rate=None means stochastic depth is off, i.e. a flat rate of 0.0.
        max_drop_rate = 0.0 if drop_path_rate is None else drop_path_rate
        total_blocks = 2 * body_levels * local_blocks_per_level + global_blocks
        if drop_path_schedule == "linear" and total_blocks > 1:
            drop_rates = torch.linspace(0.0, max_drop_rate, total_blocks).tolist()
        else:
            drop_rates = [max_drop_rate] * total_blocks
        drop_iter = iter(drop_rates)

        def make_local_stage(dim: int) -> nn.ModuleList:
            return nn.ModuleList(
                [
                    DeepNestedLocalWindowBlock(
                        dim=dim,
                        num_heads=num_heads,
                        window_levels=window_levels,
                        mlp_ratio=mlp_ratio,
                        drop_path=next(drop_iter),
                        pre_norm=pre_norm,
                        residual_dropout=residual_dropout,
                        layerscale_init=layerscale_init,
                    )
                    for _ in range(local_blocks_per_level)
                ]
            )

        # Entry features: the C map channels of one fine token, or (with a patchified stem)
        # the flattened 4**stem_levels * C values of one patch.
        stem_features = (4 ** stem_levels) * in_channels
        self.input_proj = nn.Linear(stem_features, self.channel_dims[0])

        # Encoder down-path: local attention at each level, then a merge that folds the last
        # nested axis into channels (channel_dims[level] -> channel_dims[level + 1]).
        self.enc_stages = nn.ModuleList([make_local_stage(self.channel_dims[level]) for level in range(body_levels)])
        self.patch_merges = nn.ModuleList(
            [NestedPatchMerge4(self.channel_dims[level], self.channel_dims[level + 1]) for level in range(body_levels)]
        )

        # Bottleneck: global attention over the N top-level tokens (0 nested axes remain).
        final_dim = self.channel_dims[-1]
        self.global_blocks = nn.ModuleList(
            [
                DeepTransformerBlock(
                    dim=final_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=next(drop_iter),
                    pre_norm=pre_norm,
                    residual_dropout=residual_dropout,
                    layerscale_init=layerscale_init,
                )
                for _ in range(global_blocks)
            ]
        )
        self.norm = nn.LayerNorm(final_dim)

        # Decoder up-path (mirror of the encoder). For each level, expand splits a size-4 axis
        # back out (channel_dims[level + 1] -> channel_dims[level]), the encoder skip at that
        # level is fused in, and local attention refines the result. Indexed by level; the
        # decoder loop walks them in reverse (bottleneck -> finest).
        self.patch_expands = nn.ModuleList(
            [NestedPatchExpand4(self.channel_dims[level + 1], self.channel_dims[level]) for level in range(body_levels)]
        )
        self.skip_fuses = nn.ModuleList([NestedSkipFuse(self.channel_dims[level]) for level in range(body_levels)])
        # Built in reverse (coarse -> fine) so the DropPath schedule keeps rising through the
        # decoder; stored fine-first to index by level like the other decoder module lists.
        dec_stages = [None] * body_levels
        for level in reversed(range(body_levels)):
            dec_stages[level] = make_local_stage(self.channel_dims[level])
        self.dec_stages = nn.ModuleList(dec_stages)

        # Output projection: inverse of the stem embed. Maps channel_dims[0] back to one fine
        # token's C channels, or (with a patchified stem) the 4**stem_levels * C values that
        # forward() unfolds into the finest stem_levels nested axes.
        self.output_proj = nn.Linear(self.channel_dims[0], stem_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, N, 4, ..., 4) -> (B, C, N, 4, ..., 4), returning input + learned correction."""
        expected_ndim = 3 + self.num_nested_levels
        if x.ndim != expected_ndim:
            raise ValueError(
                f"Expected input with {expected_ndim} dims: (B, C, N, 4, ..., 4), got shape {tuple(x.shape)}."
            )

        B, channels, N = x.shape[:3]
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {channels}.")
        for size in x.shape[3:]:
            if size != 4:
                raise ValueError("Every nested resolution dimension must have size 4.")

        # Keep the channel-first input for the final residual add.
        x_in = x

        # (B, C, N, 4, ..., 4) -> (B, N, 4, ..., 4, C)
        x = x.movedim(1, -1).contiguous()

        # Patchified stem: fold the finest stem_levels nested axes into the feature dimension.
        # They are the trailing axes (channel-minor, child-major), so this is a pure reshape,
        # and the hierarchy then runs over the remaining body_levels axes.
        if self.stem_levels > 0:
            stem_features = (4 ** self.stem_levels) * self.in_channels
            x = x.reshape(B, N, *([4] * self.body_levels), stem_features)

        x = self.input_proj(x)

        # Encoder: refine at each level, save the pre-merge feature as a skip, then merge down.
        skips = []
        for level in range(self.body_levels):
            for block in self.enc_stages[level]:
                x = block(x)
            skips.append(x)
            x = self.patch_merges[level](x)

        # Bottleneck: global attention over the N top-level tokens.
        for block in self.global_blocks:
            x = block(x)
        x = self.norm(x)

        # Decoder: expand back up, fuse the matching encoder skip, refine.
        for level in reversed(range(self.body_levels)):
            x = self.patch_expands[level](x)
            x = self.skip_fuses[level](x, skips[level])
            for block in self.dec_stages[level]:
                x = block(x)

        # Project back to the input channels and unfold the patchified stem so the prediction
        # is dense at the input nside.
        x = self.output_proj(x)
        if self.stem_levels > 0:
            x = x.reshape(B, N, *([4] * self.num_nested_levels), self.in_channels)

        # (B, N, 4, ..., 4, C) -> (B, C, N, 4, ..., 4); add as a residual correction.
        correction = x.movedim(-1, 1).contiguous()
        return x_in + correction
