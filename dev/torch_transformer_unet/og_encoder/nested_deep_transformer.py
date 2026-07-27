"""Deeper nested hierarchical local-window transformer variants.

This module intentionally mirrors :mod:`nested_transfomer` while adding several
configuration-friendly stabilizers and readout options:

* pre-normalized transformer blocks, enabled by default;
* DropPath / stochastic depth on residual branches;
* LayerScale on each residual branch;
* a patchified stem (``stem_levels``) that folds the finest nested levels into the
  input projection so the hierarchy starts coarser; and
* an optional multi-scale readout that feeds every stage's pooled feature to the head.
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
        # which reproduces the pre-LayerScale behaviour exactly). Applied to the branch
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
    """Local nested-window attention using :class:`DeepTransformerBlock`."""

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


class DeepNestedHierarchicalLocalWindowTransformer(nn.Module):
    """Nested hierarchical local-window transformer with depth controls.

    The tensor interface matches ``NestedHierarchicalLocalWindowTransformer``.
    Depth is controlled by ``local_blocks_per_level`` and ``global_blocks``;
    stochastic depth by ``drop_path_rate`` / ``drop_path_schedule``; residual-branch
    scaling by ``layerscale_init`` (off by default). A patchified stem
    (``stem_levels``) folds the finest nested levels into the input projection so the
    hierarchy runs over ``body_levels = num_nested_levels - stem_levels`` levels, and
    ``multiscale_readout`` concatenates every stage's pooled feature into the head input.
    """

    def __init__(
        self,
        in_channels,
        num_outputs,
        num_nested_levels,
        base_embed_dim=128,
        growth="constant",
        num_heads=4,
        window_levels=3,
        stem_levels=0,
        local_blocks_per_level=2,
        global_blocks=2,
        mlp_ratio=4,
        drop_path_rate=0.1,
        drop_path_schedule="linear",
        pre_norm=True,
        residual_dropout=0.0,
        layerscale_init=None,
        multiscale_readout=False,
    ):
        super().__init__()
        if num_nested_levels < 0:
            raise ValueError("num_nested_levels must be >= 0")
        if local_blocks_per_level < 0:
            raise ValueError("local_blocks_per_level must be >= 0")
        if global_blocks < 1:
            raise ValueError("global_blocks must be >= 1")
        if drop_path_rate < 0.0 or drop_path_rate >= 1.0:
            raise ValueError("drop_path_rate must satisfy 0 <= drop_path_rate < 1")
        if drop_path_schedule not in {"linear", "constant"}:
            raise ValueError("drop_path_schedule must be 'linear' or 'constant'")

        # stem_levels: patchified stem — the finest ``stem_levels`` nested levels are folded
        # into the input projection (one linear embed of 4**stem_levels fine tokens per patch,
        # child-major) and the transformer hierarchy starts that many levels coarser. The
        # embed is information-lossless when channel_dims[0] >= 4**stem_levels * in_channels;
        # what is given up is only per-token nonlinear processing below the patch scale.
        # stem_levels=0 reproduces the per-finest-pixel stem exactly.
        if stem_levels < 0:
            raise ValueError("stem_levels must be >= 0")
        if stem_levels > 0 and stem_levels >= num_nested_levels:
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
        self.multiscale_readout = multiscale_readout

        # Channel dimensions of the hierarchy after the stem. Length is body_levels + 1.
        self.channel_dims = make_channel_dims(base_embed_dim, body_levels, growth)
        for dim in self.channel_dims:
            if dim % num_heads != 0:
                raise ValueError(f"Channel dimension {dim} must be divisible by num_heads={num_heads}.")

        total_blocks = body_levels * local_blocks_per_level + global_blocks
        if drop_path_schedule == "linear" and total_blocks > 1:
            drop_rates = torch.linspace(0.0, drop_path_rate, total_blocks).tolist()
        else:
            drop_rates = [drop_path_rate] * total_blocks
        drop_iter = iter(drop_rates)

        # Entry features: the C map channels of one fine token, or (with a patchified stem)
        # the flattened 4**stem_levels * C values of one patch.
        stem_features = (4 ** stem_levels) * in_channels
        self.input_proj = nn.Linear(stem_features, self.channel_dims[0])
        self.local_stages = nn.ModuleList()
        for level in range(body_levels):
            dim = self.channel_dims[level]
            self.local_stages.append(
                nn.ModuleList(
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
            )

        self.patch_merges = nn.ModuleList(
            [NestedPatchMerge4(self.channel_dims[level], self.channel_dims[level + 1]) for level in range(body_levels)]
        )

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

        # multiscale_readout: give the head direct access to every scale — the mean of each
        # stage's tokens right before its merge, one LayerNorm per stage to homogenise the
        # feature scales, concatenated with the pooled top-level feature. Without it, small-
        # scale information must survive every merge to reach the head at all.
        self.readout_norms = (
            nn.ModuleList([nn.LayerNorm(self.channel_dims[level]) for level in range(body_levels)])
            if multiscale_readout
            else None
        )
        head_input_dim = final_dim + (sum(self.channel_dims[:-1]) if multiscale_readout else 0)
        self.head = nn.Linear(head_input_dim, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_ndim = 3 + self.num_nested_levels
        if x.ndim != expected_ndim:
            raise ValueError(f"Expected input with {expected_ndim} dims: (B, C, N, 4, ..., 4), got shape {tuple(x.shape)}.")

        B, channels, N = x.shape[:3]
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {channels}.")
        for size in x.shape[3:]:
            if size != 4:
                raise ValueError("Every nested resolution dimension must have size 4.")

        # (B, C, N, 4, ..., 4) -> (B, N, 4, ..., 4, C)
        x = x.movedim(1, -1).contiguous()

        # Patchified stem: fold the finest stem_levels nested axes into the feature
        # dimension. They are the trailing axes (channel-minor, child-major), so this is a
        # pure reshape, and the hierarchy then runs over the remaining body_levels axes.
        if self.stem_levels > 0:
            stem_features = (4 ** self.stem_levels) * self.in_channels
            x = x.reshape(B, N, *([4] * self.body_levels), stem_features)

        x = self.input_proj(x)

        readout_features = []
        for level in range(self.body_levels):
            for block in self.local_stages[level]:
                x = block(x)
            if self.readout_norms is not None:
                # multi-scale tap: mean over every token / nested axis (keep batch and
                # channel), one LayerNorm per stage.
                token_axes = tuple(range(1, x.ndim - 1))
                readout_features.append(self.readout_norms[level](x.mean(dim=token_axes)))
            x = self.patch_merges[level](x)

        for block in self.global_blocks:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)

        if self.readout_norms is not None:
            x = torch.cat(readout_features + [x], dim=-1)

        return self.head(x)
