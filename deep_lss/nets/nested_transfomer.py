from __future__ import annotations

import tensorflow as tf


def make_channel_dims(base_embed_dim, num_nested_levels, growth):
    """
    Returns the channel dimension at each resolution level.

    There are M nested levels, so there are M merges.

    Example with M = 4 and base_embed_dim = 64:

        constant:
            [64, 64, 64, 64, 64]

        double:
            [64, 128, 256, 512, 1024]

        full:
            [64, 256, 1024, 4096, 16384]
    """
    if growth == "constant":
        factor, increase = 1, 0
    elif growth == "double":
        factor, increase = 2, 0
    elif growth == "full":
        factor, increase = 4, 0
    elif growth == "128":
        factor, increase = 1, 128
    else:
        raise ValueError(
            "growth must be one of: 'constant', 'double', 'full', '128'"
        )

    dims = [base_embed_dim]

    for _ in range(num_nested_levels):
        dims.append(dims[-1] * factor + increase)

    return dims


def _require_static_rank(x: tf.Tensor, layer_name: str) -> int:
    rank = x.shape.rank
    if rank is None:
        raise ValueError(f"{layer_name} requires inputs with a statically known rank.")
    return rank


def _maybe_assert_dim(
    x: tf.Tensor,
    axis: int,
    expected: int,
    description: str,
    rank: int | None = None,
):
    """Return a TensorFlow assertion op when a dimension is dynamic."""
    static_rank = x.shape.rank
    if static_rank is None:
        if rank is None:
            raise ValueError(f"{description} requires inputs with a statically known rank.")
        resolved_axis = axis if axis >= 0 else rank + axis
        actual_static = None
    else:
        resolved_axis = axis if axis >= 0 else static_rank + axis
        actual_static = x.shape[resolved_axis]

    if actual_static is not None:
        if actual_static != expected:
            raise ValueError(
                f"Expected {description} to be {expected}, got {actual_static}."
            )
        return None

    shape = tf.shape(x)
    return tf.debugging.assert_equal(
        shape[resolved_axis],
        tf.cast(expected, shape.dtype),
        message=f"Expected {description} to be {expected}.",
    )


def _apply_assertions(x: tf.Tensor, assertions) -> tf.Tensor:
    assertions = [assertion for assertion in assertions if assertion is not None]
    if not assertions:
        return x

    with tf.control_dependencies(assertions):
        return tf.identity(x)


class MLP(tf.keras.layers.Layer):
    def __init__(self, dim, mlp_ratio=4, **kwargs):
        super().__init__(**kwargs)

        hidden_dim = dim * mlp_ratio
        if int(hidden_dim) != hidden_dim:
            raise ValueError("dim * mlp_ratio must be an integer.")
        hidden_dim = int(hidden_dim)

        self.fc1 = tf.keras.layers.Dense(hidden_dim)
        self.activation = tf.keras.layers.Activation(tf.keras.activations.gelu)
        self.fc2 = tf.keras.layers.Dense(dim)

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class TransformerBlock(tf.keras.layers.Layer):
    """
    Standard transformer block over a sequence.

    Input:
        x: (B_like, S, D)

    where:
        B_like = any batch-like dimension
        S      = sequence length
        D      = feature/channel dimension
    """

    def __init__(self, dim, num_heads, mlp_ratio=4, **kwargs):
        super().__init__(**kwargs)

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}."
            )

        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
        )
        self.norm2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.mlp = MLP(dim, mlp_ratio)

    def call(self, x, training=None):
        shortcut = x

        x_norm = self.norm1(x)
        attn_out = self.attn(
            query=x_norm,
            value=x_norm,
            key=x_norm,
            training=training,
        )

        x = shortcut + attn_out
        x = x + self.mlp(self.norm2(x), training=training)

        return x


class NestedLocalWindowBlock(tf.keras.layers.Layer):
    """
    Local attention over the last few nested resolution dimensions.

    Input:
        x: (B, N, 4, 4, ..., 4, D)

    Example:

        x: (B, N, 4, 4, 4, 4, D)

    If window_levels = 3, attention is applied over:

        4 x 4 x 4 = 64 tokens

    Internally:

        (B, N, 4, 4, 4, 4, D)
            ->
        (B * N * 4, 64, D)
            -> attention
        (B, N, 4, 4, 4, 4, D)

    This does not reshape the data into a 2D image.
    It only flattens local nested windows into sequences.
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_levels=3,
        mlp_ratio=4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if window_levels < 1:
            raise ValueError("window_levels must be >= 1")

        self.dim = dim
        self.window_levels = window_levels
        self.block = TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

    def call(self, x, training=None):
        """
        x: (B, N, 4, 4, ..., 4, D)
        """
        rank = _require_static_rank(x, self.__class__.__name__)
        num_nested_levels = rank - 3

        if num_nested_levels <= 0:
            raise ValueError(
                "NestedLocalWindowBlock needs at least one nested resolution dimension."
            )

        assertions = [_maybe_assert_dim(x, -1, self.dim, "feature dimension")]

        levels_used = min(self.window_levels, num_nested_levels)
        sequence_length = 4 ** levels_used

        # Validate the local window dimensions: the last levels_used nested
        # dimensions immediately before the channel dimension must all be 4.
        first_window_axis = rank - levels_used - 1
        for axis in range(first_window_axis, rank - 1):
            assertions.append(
                _maybe_assert_dim(x, axis, 4, "nested resolution dimension")
            )

        x = _apply_assertions(x, assertions)

        original_static_shape = x.shape.as_list()
        for axis in range(first_window_axis, rank - 1):
            original_static_shape[axis] = 4
        original_static_shape[-1] = self.dim
        x.set_shape(original_static_shape)

        original_shape = tf.shape(x)
        feature_dim = tf.shape(x)[-1]

        # Flatten local nested window into a sequence:
        #   (..., 4, 4, 4, D) -> (..., 64, D)
        x = tf.reshape(x, tf.stack([-1, sequence_length, feature_dim]))
        x.set_shape([None, sequence_length, self.dim])

        x = self.block(x, training=training)

        # Restore nested tensor shape.
        x = tf.reshape(x, original_shape)
        x.set_shape(original_static_shape)

        return x


class NestedPatchMerge4(tf.keras.layers.Layer):
    """
    Merge the last nested resolution dimension.

    Input:
        x: (B, N, 4, 4, ..., 4, in_dim)

    Output:
        x: (B, N, 4, 4, ..., out_dim)

    The final nested dimension has size 4.

    For each parent token:

        4 child tokens x in_dim features = 4 * in_dim features

    Then:

        4 * in_dim -> out_dim

    The value of out_dim depends on the channel growth strategy.
    """

    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.reduction = tf.keras.layers.Dense(out_dim)

    def call(self, x, training=None):
        """
        x: (B, N, 4, 4, ..., 4, in_dim)
        """
        rank = _require_static_rank(x, self.__class__.__name__)
        if rank < 4:
            raise ValueError(
                "NestedPatchMerge4 needs at least one nested resolution dimension."
            )

        assertions = [
            _maybe_assert_dim(x, -2, 4, "last nested dimension"),
            _maybe_assert_dim(x, -1, self.in_dim, "feature dimension"),
        ]
        x = _apply_assertions(x, assertions)

        static_shape = x.shape.as_list()
        static_shape[-2] = 4
        static_shape[-1] = self.in_dim
        x.set_shape(static_shape)

        # Everything except the final nested dimension and channel dimension.
        # Example:
        #   x:            (B, N, 4, 4, 4, D)
        #   prefix_shape: (B, N, 4, 4)
        prefix_shape = tf.shape(x)[:-2]
        prefix_static_shape = x.shape[:-2]

        # Concatenate the 4 children into the channel dimension:
        #   (B, N, 4, 4, 4, D) -> (B, N, 4, 4, 4D)
        new_last_dim = tf.constant([4 * self.in_dim], dtype=prefix_shape.dtype)
        x = tf.reshape(x, tf.concat([prefix_shape, new_last_dim], axis=0))
        x.set_shape(prefix_static_shape.concatenate([4 * self.in_dim]))

        x = self.norm(x)
        x = self.reduction(x)

        return x


class NestedHierarchicalLocalWindowTransformer(tf.keras.Model):
    """
    Hierarchical Local Window Transformer for nested tensors.

    Input:
        x: (B, C, N, 4, 4, ..., 4)

    where:
        B = batch size
        C = input channels
        N = number of top-level/basic patches
        M = num_nested_levels
        each nested resolution dimension has size 4

    Internal representation:
        x: (B, N, 4, 4, ..., 4, D)

    Processing:
        input projection
        -> local nested attention
        -> patch merge
        -> local nested attention
        -> patch merge
        -> ...
        -> final tensor of shape (B, N, D_final)
        -> global attention over N tokens
        -> pooling over N
        -> prediction head

    The final global attention operates over N tokens, so internally it has
    an N x N attention matrix per head.
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
        local_blocks_per_level=1,
        global_blocks=1,
        mlp_ratio=4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if num_nested_levels < 0:
            raise ValueError("num_nested_levels must be >= 0")

        if local_blocks_per_level < 0:
            raise ValueError("local_blocks_per_level must be >= 0")

        if global_blocks < 1:
            raise ValueError("global_blocks must be >= 1")

        self.in_channels = in_channels
        self.num_nested_levels = num_nested_levels
        self.base_embed_dim = base_embed_dim
        self.growth = growth
        self.num_heads = num_heads
        self.window_levels = window_levels
        self.local_blocks_per_level = local_blocks_per_level
        self.num_global_blocks = global_blocks
        self.mlp_ratio = mlp_ratio

        # Channel dimensions at each resolution level.
        # Length is num_nested_levels + 1.
        self.channel_dims = make_channel_dims(
            base_embed_dim=base_embed_dim,
            num_nested_levels=num_nested_levels,
            growth=growth,
        )

        for dim in self.channel_dims:
            if dim % num_heads != 0:
                raise ValueError(
                    f"Channel dimension {dim} must be divisible by num_heads={num_heads}."
                )

        # Project input channels C -> base_embed_dim.
        # Applied independently to every fine nested token.
        self.input_proj = tf.keras.layers.Dense(self.channel_dims[0])

        # One local stage per nested resolution level.
        # Stage i operates before merge i.
        # Its channel dimension is channel_dims[i].
        local_stages = []
        for level in range(num_nested_levels):
            dim = self.channel_dims[level]
            local_stages.append(
                [
                    NestedLocalWindowBlock(
                        dim=dim,
                        num_heads=num_heads,
                        window_levels=window_levels,
                        mlp_ratio=mlp_ratio,
                        name=f"local_stage_{level}_block_{block_index}",
                    )
                    for block_index in range(local_blocks_per_level)
                ]
            )
        self.local_stages = local_stages

        # One patch merge per nested level.
        # Merge i maps channel_dims[i] -> channel_dims[i + 1].
        self.patch_merges = [
            NestedPatchMerge4(
                in_dim=self.channel_dims[level],
                out_dim=self.channel_dims[level + 1],
                name=f"patch_merge_{level}",
            )
            for level in range(num_nested_levels)
        ]

        # Final global attention over the N basic-patch tokens.
        final_dim = self.channel_dims[-1]
        self.global_blocks = [
            TransformerBlock(
                dim=final_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                name=f"global_block_{block_index}",
            )
            for block_index in range(global_blocks)
        ]

        self.norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.head = tf.keras.layers.Dense(num_outputs)

    def call(self, x, training=None):
        """
        Input:
            x: (B, C, N, 4, 4, ..., 4)

        Output:
            y: (B, num_outputs)
        """
        expected_ndim = 3 + self.num_nested_levels
        rank = x.shape.rank

        if rank is not None and rank != expected_ndim:
            raise ValueError(
                f"Expected input with {expected_ndim} dims: "
                f"(B, C, N, 4, ..., 4), got shape {tuple(x.shape)}."
            )

        assertions = []
        if rank is None:
            assertions.append(
                tf.debugging.assert_equal(
                    tf.rank(x),
                    expected_ndim,
                    message=(
                        f"Expected input with {expected_ndim} dims: "
                        "(B, C, N, 4, ..., 4)."
                    ),
                )
            )

        assertions.append(
            _maybe_assert_dim(
                x, 1, self.in_channels, "input channel dimension", rank=expected_ndim
            )
        )
        for axis in range(3, expected_ndim):
            assertions.append(
                _maybe_assert_dim(
                    x, axis, 4, "nested resolution dimension", rank=expected_ndim
                )
            )
        x = _apply_assertions(x, assertions)

        input_static_shape = x.shape.as_list() if x.shape.rank is not None else None

        # Move channels to the end:
        #   (B, C, N, 4, 4, ..., 4) -> (B, N, 4, 4, ..., 4, C)
        perm = [0] + list(range(2, expected_ndim)) + [1]
        x = tf.transpose(x, perm=perm)
        batch_dim = input_static_shape[0] if input_static_shape is not None else None
        token_dim = input_static_shape[2] if input_static_shape is not None else None
        x.set_shape(
            [batch_dim, token_dim, *([4] * self.num_nested_levels), self.in_channels]
        )

        # Project C -> base_embed_dim:
        #   (B, N, 4, 4, ..., 4, C) -> (B, N, 4, 4, ..., 4, D0)
        x = self.input_proj(x)

        # Hierarchical local processing.
        for level in range(self.num_nested_levels):
            for block in self.local_stages[level]:
                x = block(x, training=training)

            x = self.patch_merges[level](x, training=training)

        # After all merges:
        #   x: (B, N, final_dim)
        # Apply final global attention over N tokens.
        for block in self.global_blocks:
            x = block(x, training=training)

        x = self.norm(x)

        # Pool over the N basic patches:
        #   (B, N, final_dim) -> (B, final_dim)
        x = tf.reduce_mean(x, axis=1)

        # Classification or regression head:
        #   (B, final_dim) -> (B, num_outputs)
        x = self.head(x)

        return x
