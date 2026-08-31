from __future__ import annotations

import numpy as np
import tensorflow as tf
from deepsphere.gnn_layers import DropPath
from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


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
        raise ValueError("growth must be one of: 'constant', 'double', 'full', '128'")

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
            raise ValueError(f"Expected {description} to be {expected}, got {actual_static}.")
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


def _fp32_layer_norm(**kwargs):
    """LayerNormalization whose mean/variance are computed in float32 even under a
    mixed-bfloat16 global policy.

    The bf16 variance + rsqrt reduction is the main NaN source in the deeper stacks
    (extra local blocks / nested levels): the 7-bit mantissa loses too much precision in
    the normalization statistics. Forcing the layer to a pure float32 policy fixes that
    while the fp32 output is auto-cast back to the compute dtype by the following layer,
    so the residual stream and activation memory stay bf16. Mirrors the fp32 smoothing
    carve-out in ``transformer_networks.py``.
    """
    return tf.keras.layers.LayerNormalization(dtype="float32", **kwargs)


class Fp32SoftmaxMultiHeadAttention(tf.keras.layers.MultiHeadAttention):
    """MultiHeadAttention that evaluates the attention softmax in float32.

    The Q/K/V projections and the value aggregation stay in the (bf16) compute dtype, so
    attention activation memory is unchanged; only the softmax normalization is upcast to
    float32 and cast back afterwards. This is cheap insurance against a bf16 softmax
    blowing up in deep stacks (the fp32 LayerNorm is the primary stabilizer).

    Overriding ``_masked_softmax`` keeps this valid across Keras 2/3. A mask is applied
    the same way the stock ``keras.layers.Softmax`` does — expanded at the heads axis and
    added as ``(1 - mask) * -1e9`` — but on the fp32 scores, so masked and unmasked runs
    share the same fp32 normalization (and an all-True mask is bit-identical to no mask).
    The fp32 path covers the single-axis attention used here (softmax over the last, key,
    axis); multi-axis attention falls back to the stock implementation.
    """

    def _masked_softmax(self, attention_scores, attention_mask=None):
        if len(self._attention_axes) != 1:
            return super()._masked_softmax(attention_scores, attention_mask)
        compute_dtype = attention_scores.dtype
        scores = tf.cast(attention_scores, tf.float32)
        if attention_mask is not None:
            # expand at the heads axis, mirroring the Keras parent: (..., Tq, Tk) masks
            # broadcast over (B_like, H, Tq, Tk) scores
            mask_expansion_axis = -3
            for _ in range(scores.shape.rank - attention_mask.shape.rank):
                attention_mask = tf.expand_dims(attention_mask, axis=mask_expansion_axis)
            scores += (1.0 - tf.cast(attention_mask, tf.float32)) * -1e9
        scores = tf.nn.softmax(scores, axis=-1)
        return tf.cast(scores, compute_dtype)


class GeodesicKernelAttention(Fp32SoftmaxMultiHeadAttention):
    """Fp32SoftmaxMultiHeadAttention with a smooth geodesic-distance kernel bias.

    Adds ``b_ij(h) = a_h * dist_sq_ij`` to the attention logits, where ``dist_sq`` is a
    precomputed (S, S) table of squared geodesic distances between the S window tokens
    (normalized by the window's maximum distance) and ``a_h`` is a learnable per-head
    coefficient. This is the RBF kernel ``-d^2 / (2 sigma_h^2)`` with a learnable signed
    precision ``a_h``. Because the bias depends on pairwise distance only, it is symmetric
    (``b_ij = b_ji``) and invariant under every isometry of the window, so local attention
    regains real geometry without breaking the (approximate) rotation/reflection symmetry
    of the data. ``a_h`` is initialized at ``coeff_init``: 0 starts as exactly plain
    attention, but the bench_t7 symmetric run showed the coefficients then never engage
    (|a_h| <= 0.023 after 150k steps), so a non-zero init (e.g. -1, a real RBF at step 0)
    is needed to actually exercise the positional pathway.
    """

    def __init__(self, dist_sq, num_heads, coeff_init=0.0, **kwargs):
        super().__init__(num_heads=num_heads, **kwargs)
        self._kernel_num_heads = num_heads
        self.dist_sq = tf.constant(dist_sq, dtype=tf.float32)  # (S, S)
        self.coeff_init = float(coeff_init)

    def build(self, input_shape):
        self.kernel_coeff = self.add_weight(
            name="kernel_coeff",
            shape=(self._kernel_num_heads,),
            initializer=tf.keras.initializers.Constant(self.coeff_init),
            trainable=True,
            dtype="float32",
        )
        super().build(input_shape)

    def _masked_softmax(self, attention_scores, attention_mask=None):
        # attention_scores: (B_like, H, S, S)
        # Under a mixed-precision policy kernel_coeff is an AutoCastVariable that reads
        # in the compute dtype; force fp32 to match the fp32 dist_sq constant.
        coeff = tf.cast(self.kernel_coeff, tf.float32)
        bias = coeff[:, tf.newaxis, tf.newaxis] * self.dist_sq  # (H, S, S)
        attention_scores = attention_scores + tf.cast(bias[tf.newaxis], attention_scores.dtype)
        return super()._masked_softmax(attention_scores, attention_mask)


class GeodesicBinnedBiasAttention(Fp32SoftmaxMultiHeadAttention):
    """Fp32SoftmaxMultiHeadAttention with a learnable distance-binned relative bias.

    Adds ``b_ij(h) = bias_table[h, bin_idx_ij]`` to the attention logits, where
    ``bin_idx`` is a precomputed (S, S) table assigning each window-token pair to one of
    ``B`` geodesic-distance bins (bin 0 = the diagonal, d = 0). This is the Swin-style
    learnable relative position bias restricted to a function of pairwise distance only,
    so like GeodesicKernelAttention it is symmetric (``b_ij = b_ji``) and invariant under
    every isometry of the window — but with (num_heads, B) capacity instead of one scalar
    per head it can express non-monotone kernels and receives O(1) gradient signal per
    bin. The table is initialized as the RBF kernel ``coeff_init * bin_center_dsq`` so
    the positional pathway is engaged from step 0 (the zero-init scalar kernel was shown
    not to bootstrap).
    """

    def __init__(self, bin_idx, bin_centers, num_heads, coeff_init=-1.0, **kwargs):
        super().__init__(num_heads=num_heads, **kwargs)
        self._bias_num_heads = num_heads
        self.bin_idx = tf.constant(bin_idx, dtype=tf.int32)  # (S, S)
        self._bin_centers = np.asarray(bin_centers, dtype=np.float32)  # (B,)
        self.num_bins = len(self._bin_centers)
        self.coeff_init = float(coeff_init)

    def build(self, input_shape):
        init_table = self.coeff_init * np.tile(self._bin_centers[np.newaxis, :], (self._bias_num_heads, 1))  # (H, B)

        # callable initializer: Keras Constant only reliably supports scalars
        def _rbf_table_init(shape, dtype=None):
            return tf.constant(init_table, dtype=dtype or tf.float32)

        self.bias_table = self.add_weight(
            name="bias_table",
            shape=(self._bias_num_heads, self.num_bins),
            initializer=_rbf_table_init,
            trainable=True,
            dtype="float32",
        )
        super().build(input_shape)

    def _masked_softmax(self, attention_scores, attention_mask=None):
        # attention_scores: (B_like, H, S, S)
        # Under a mixed-precision policy bias_table is an AutoCastVariable that reads
        # in the compute dtype; force fp32 for the gather, mirroring kernel_coeff.
        table = tf.cast(self.bias_table, tf.float32)  # (H, B)
        bias = tf.gather(table, self.bin_idx, axis=1)  # (H, S, S)
        attention_scores = attention_scores + tf.cast(bias[tf.newaxis], attention_scores.dtype)
        return super()._masked_softmax(attention_scores, attention_mask)


class AttentionPool(tf.keras.layers.Layer):
    """Learned-query cross-attention pooling over a token sequence (the PMA of Lee et al.
    2019, "Set Transformer", arXiv:1810.00825).

    ``num_queries`` learned query vectors cross-attend to the input tokens and their
    outputs are concatenated: (B, N, D) -> (B, num_queries * D). With uniform attention
    this reduces to a mean pool (up to the value/output projections), so it strictly
    generalizes the mean pool — the readout can weight regions and form several distinct
    projections instead of one uniform average.

    key_mask: optional static (1, 1, N) boolean key-side mask (True = token may be
    attended to); masked tokens are excluded from the pool exactly.
    """

    def __init__(self, dim, num_heads, num_queries=1, fp32_softmax=True, key_mask=None, **kwargs):
        super().__init__(**kwargs)

        if num_queries < 1:
            raise ValueError("num_queries must be >= 1")

        self.dim = dim
        self.num_queries = num_queries
        self.key_mask = None if key_mask is None else tf.constant(key_mask, dtype=tf.bool)
        attn_cls = Fp32SoftmaxMultiHeadAttention if fp32_softmax else tf.keras.layers.MultiHeadAttention
        self.attn = attn_cls(
            num_heads=num_heads,
            key_dim=dim // num_heads,
        )

    def build(self, input_shape):
        self.query = self.add_weight(
            name="query",
            shape=(self.num_queries, self.dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x, training=None):
        # x: (B, N, D) -> (B, num_queries * D)
        batch = tf.shape(x)[0]
        query = tf.tile(tf.cast(self.query, x.dtype)[tf.newaxis], [batch, 1, 1])
        pooled = self.attn(
            query=query,
            value=x,
            key=x,
            attention_mask=self.key_mask,
            training=training,
        )
        return tf.reshape(pooled, [-1, self.num_queries * self.dim])


class LayerScale(tf.keras.layers.Layer):
    """Per-channel learnable scale on a residual branch (LayerScale, Touvron et al. 2021,
    "Going deeper with image transformers", arXiv:2103.17239).

    Initialized to a small value so each attention / MLP branch starts as a near-identity
    perturbation of the residual stream. This keeps deep stacks stable at initialization —
    the failure mode that made the ``deep`` (local_blocks 2) and ``coarse`` (6 levels)
    configs diverge to NaN within ~100 steps regardless of precision. ``gamma`` then learns
    the useful per-channel branch magnitude. The scale variable stays float32 (variable
    dtype) and is cast to the compute dtype for the multiply.
    """

    def __init__(self, dim, init_value=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.init_value = init_value

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=(self.dim,),
            initializer=tf.keras.initializers.Constant(self.init_value),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return x * tf.cast(self.gamma, x.dtype)


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

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4,
        layerscale_init=1e-4,
        fp32_softmax=True,
        attn_dist_sq=None,
        attn_dist_bins=None,
        attn_coeff_init=0.0,
        block_dropout_rate=None,
        drop_path_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.norm1 = _fp32_layer_norm(axis=-1, epsilon=1e-5)
        # fp32_softmax: upcast the attention softmax to float32 (see Fp32SoftmaxMultiHeadAttention).
        # When False, use the stock bf16 softmax — the fp32 LayerNorm remains the primary stabilizer.
        # attn_dist_sq: optional (S, S) normalized squared geodesic distances between the S
        # sequence tokens; enables the distance-kernel bias (GeodesicKernelAttention, which
        # always takes the fp32-softmax path).
        # attn_dist_bins: optional (bin_idx (S, S), bin_centers (B,)) tuple; enables the
        # distance-binned learnable bias (GeodesicBinnedBiasAttention) instead. Mutually
        # exclusive with attn_dist_sq. attn_coeff_init sets the RBF init of either bias.
        if attn_dist_sq is not None and attn_dist_bins is not None:
            raise ValueError(
                "attn_dist_sq and attn_dist_bins are mutually exclusive — pass one " "positional bias table, not both."
            )
        if attn_dist_bins is not None:
            bin_idx, bin_centers = attn_dist_bins
            self.attn = GeodesicBinnedBiasAttention(
                bin_idx=bin_idx,
                bin_centers=bin_centers,
                num_heads=num_heads,
                key_dim=dim // num_heads,
                coeff_init=attn_coeff_init,
            )
        elif attn_dist_sq is not None:
            self.attn = GeodesicKernelAttention(
                dist_sq=attn_dist_sq,
                num_heads=num_heads,
                key_dim=dim // num_heads,
                coeff_init=attn_coeff_init,
            )
        else:
            attn_cls = Fp32SoftmaxMultiHeadAttention if fp32_softmax else tf.keras.layers.MultiHeadAttention
            self.attn = attn_cls(
                num_heads=num_heads,
                key_dim=dim // num_heads,
            )
        self.norm2 = _fp32_layer_norm(axis=-1, epsilon=1e-5)
        self.mlp = MLP(dim, mlp_ratio)
        # LayerScale on each residual branch (disabled when layerscale_init is None, which
        # reproduces the pre-LayerScale behaviour exactly).
        self.ls1 = LayerScale(dim, layerscale_init) if layerscale_init is not None else None
        self.ls2 = LayerScale(dim, layerscale_init) if layerscale_init is not None else None
        # block_dropout_rate: residual-branch dropout (the ViT "drop rate") on the attention
        # and MLP branch outputs before each residual add. Variable-free, so toggling keeps
        # the checkpoint lineage.
        self.drop1 = tf.keras.layers.Dropout(block_dropout_rate) if block_dropout_rate is not None else None
        self.drop2 = tf.keras.layers.Dropout(block_dropout_rate) if block_dropout_rate is not None else None
        # Stochastic depth on the whole residual branch: the same DropPath the GCNN's ConvNeXt
        # block uses, at a rate held CONSTANT across depth (as resnet.py does) rather than timm's
        # linear ramp, so the knob reads the same on both architectures. Variable-free, so
        # toggling it preserves the checkpoint lineage.
        self.dp1 = DropPath(drop_path_rate) if drop_path_rate else None
        self.dp2 = DropPath(drop_path_rate) if drop_path_rate else None

    def call(self, x, training=None, attention_mask=None):
        # attention_mask: optional boolean mask broadcastable to (B_like, S, S); True
        # entries may be attended to (Keras MultiHeadAttention convention).
        shortcut = x

        x_norm = self.norm1(x)
        attn_out = self.attn(
            query=x_norm,
            value=x_norm,
            key=x_norm,
            attention_mask=attention_mask,
            training=training,
        )
        if self.ls1 is not None:
            attn_out = self.ls1(attn_out)
        if self.drop1 is not None:
            attn_out = self.drop1(attn_out, training=training)
        if self.dp1 is not None:
            attn_out = self.dp1(attn_out, training=training)

        x = shortcut + attn_out
        mlp_out = self.mlp(self.norm2(x), training=training)
        if self.ls2 is not None:
            mlp_out = self.ls2(mlp_out)
        if self.drop2 is not None:
            mlp_out = self.drop2(mlp_out, training=training)
        if self.dp2 is not None:
            mlp_out = self.dp2(mlp_out, training=training)
        x = x + mlp_out

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
        layerscale_init=1e-4,
        fp32_softmax=True,
        dist_sq=None,
        dist_bins=None,
        pos_coeff_init=0.0,
        window_mask=None,
        block_dropout_rate=None,
        drop_path_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if window_levels < 1:
            raise ValueError("window_levels must be >= 1")

        self.dim = dim
        self.window_levels = window_levels
        # dist_sq: optional (S, S) normalized squared geodesic distances between the S
        # window tokens, in the same row-major nested order as the flattening below.
        # dist_bins: optional (bin_idx (S, S), bin_centers (B,)) distance-bin tuple in the
        # same token order; selects the binned bias instead (see TransformerBlock).
        self._dist_sq_len = None if dist_sq is None else len(dist_sq)
        # window_mask: optional static (n_windows, S, S) boolean attention mask, one slice
        # per local window of a single sample in the same row-major window order as the
        # flattening below (True = token may be attended to). Tiled over the batch in call.
        if window_mask is None:
            self.window_mask = None
        else:
            window_mask = tf.constant(window_mask, dtype=tf.bool)
            if window_mask.shape.rank != 3 or window_mask.shape[1] != window_mask.shape[2]:
                raise ValueError(f"window_mask must have shape (n_windows, S, S), got {window_mask.shape}.")
            self.window_mask = window_mask
        self.block = TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            layerscale_init=layerscale_init,
            fp32_softmax=fp32_softmax,
            attn_dist_sq=dist_sq,
            attn_dist_bins=dist_bins,
            attn_coeff_init=pos_coeff_init,
            block_dropout_rate=block_dropout_rate,
            drop_path_rate=drop_path_rate,
        )

    def call(self, x, training=None):
        """
        x: (B, N, 4, 4, ..., 4, D)
        """
        rank = _require_static_rank(x, self.__class__.__name__)
        num_nested_levels = rank - 3

        if num_nested_levels <= 0:
            raise ValueError("NestedLocalWindowBlock needs at least one nested resolution dimension.")

        assertions = [_maybe_assert_dim(x, -1, self.dim, "feature dimension")]

        levels_used = min(self.window_levels, num_nested_levels)
        sequence_length = 4**levels_used

        if self._dist_sq_len is not None and self._dist_sq_len != sequence_length:
            raise ValueError(
                f"dist_sq table has {self._dist_sq_len} tokens but the local window " f"has {sequence_length}."
            )

        if self.window_mask is not None and self.window_mask.shape[1] != sequence_length:
            raise ValueError(
                f"window_mask has {self.window_mask.shape[1]} tokens per window but the "
                f"local window has {sequence_length}."
            )

        # Validate the local window dimensions: the last levels_used nested
        # dimensions immediately before the channel dimension must all be 4.
        first_window_axis = rank - levels_used - 1
        for axis in range(first_window_axis, rank - 1):
            assertions.append(_maybe_assert_dim(x, axis, 4, "nested resolution dimension"))

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

        attention_mask = None
        if self.window_mask is not None:
            # The flattened leading dim is B * n_windows with the n_windows axis fastest,
            # so the per-sample window masks tile across the batch. Static per-sample
            # geometry; the tile multiple becomes a compile-time constant under XLA.
            n_windows = self.window_mask.shape[0]
            batch = tf.shape(x)[0] // n_windows
            attention_mask = tf.tile(self.window_mask, [batch, 1, 1])

        x = self.block(x, training=training, attention_mask=attention_mask)

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

        self.norm = _fp32_layer_norm(axis=-1, epsilon=1e-5)
        self.reduction = tf.keras.layers.Dense(out_dim)

    def call(self, x, training=None):
        """
        x: (B, N, 4, 4, ..., 4, in_dim)
        """
        rank = _require_static_rank(x, self.__class__.__name__)
        if rank < 4:
            raise ValueError("NestedPatchMerge4 needs at least one nested resolution dimension.")

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


class NestedPatchMerge4DeepSets(tf.keras.layers.Layer):
    """
    Permutation-invariant merge of the last nested resolution dimension.

    Same interface as NestedPatchMerge4:

        (B, N, 4, 4, ..., 4, in_dim) -> (B, N, 4, 4, ..., out_dim)

    but symmetric: per-child Dense + gelu, mean over the 4 children, combining Dense.
    Invariant under all 24 permutations of the children (a superset of the local D4),
    unlike the concat merge, which hard-codes the arbitrary nested child order. The
    mean pool alone cannot distinguish edge from diagonal child pairs; this merge
    therefore requires a positional encoding in the preceding local attention
    (GeodesicKernelAttention), which the model constructor enforces.
    """

    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.norm = _fp32_layer_norm(axis=-1, epsilon=1e-5)
        self.child = tf.keras.layers.Dense(out_dim)
        self.activation = tf.keras.layers.Activation(tf.keras.activations.gelu)
        self.combine = tf.keras.layers.Dense(out_dim)

    def call(self, x, training=None):
        """
        x: (B, N, 4, 4, ..., 4, in_dim)
        """
        rank = _require_static_rank(x, self.__class__.__name__)
        if rank < 4:
            raise ValueError("NestedPatchMerge4DeepSets needs at least one nested resolution dimension.")

        assertions = [
            _maybe_assert_dim(x, -2, 4, "last nested dimension"),
            _maybe_assert_dim(x, -1, self.in_dim, "feature dimension"),
        ]
        x = _apply_assertions(x, assertions)

        static_shape = x.shape.as_list()
        static_shape[-2] = 4
        static_shape[-1] = self.in_dim
        x.set_shape(static_shape)

        x = self.norm(x)
        x = self.activation(self.child(x))  # (..., 4, out_dim)
        x = tf.reduce_mean(x, axis=-2)  # symmetric pool over the children
        return self.combine(x)


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
        -> pooling over N   (POOL -> NORM: no pre-pool LayerNorm; the norm follows in
                             TransformerSummaryNetwork, as on the GCNN twin)
        -> optional linear output projection (num_outputs; None leaves the pooled feature)

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
        stem_levels=0,
        local_blocks_per_level=1,
        global_blocks=1,
        mlp_ratio=4,
        layerscale_init=1e-4,
        fp32_softmax=True,
        head_dropout_rate=None,
        block_dropout_rate=None,
        drop_path_rate=0.0,
        merge_op="concat",
        pool="mean",
        pool_queries=1,
        multiscale_readout=False,
        local_dist_sq=None,
        local_dist_bins=None,
        pos_coeff_init=0.0,
        token_valid=None,
        injections=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if num_nested_levels < 0:
            raise ValueError("num_nested_levels must be >= 0")

        # stem_levels: patchified stem — the finest stem_levels nested levels are folded
        # into the input projection (one linear embed of 4^stem_levels fine tokens per
        # patch, child-major) and the transformer hierarchy starts that many levels
        # coarser. The embed is information-lossless when channel_dims[0] >=
        # 4^stem_levels * in_channels; what is given up is only per-token nonlinear
        # processing below the patch scale, which the smoothing front-end has already
        # band-limited away. stem_levels=0 reproduces the per-finest-pixel stem exactly.
        if stem_levels < 0:
            raise ValueError("stem_levels must be >= 0")
        if stem_levels > 0 and stem_levels >= num_nested_levels:
            raise ValueError(
                f"stem_levels={stem_levels} must leave at least one nested level "
                f"(num_nested_levels={num_nested_levels})."
            )
        body_levels = num_nested_levels - stem_levels

        if local_blocks_per_level < 0:
            raise ValueError("local_blocks_per_level must be >= 0")

        if global_blocks < 1:
            raise ValueError("global_blocks must be >= 1")

        # pool: final aggregation over the N top-level tokens. "mean" is the uniform
        # (or masked) mean pool; "attention" is the learned-query cross-attention pool
        # (AttentionPool, pool_queries queries — a strict generalization of the mean).
        if pool not in ("mean", "attention"):
            raise ValueError(f"pool must be 'mean' or 'attention', got {pool!r}.")
        if pool_queries < 1:
            raise ValueError("pool_queries must be >= 1")

        # merge_op: "concat" reproduces the original order-sensitive concat+Dense merge;
        # "deepsets" is the child-permutation-invariant merge (NestedPatchMerge4DeepSets).
        merge_classes = {
            "concat": NestedPatchMerge4,
            "deepsets": NestedPatchMerge4DeepSets,
        }
        if merge_op not in merge_classes:
            raise ValueError(f"merge_op must be one of {sorted(merge_classes)}, got {merge_op!r}.")
        if merge_op == "deepsets" and local_dist_sq is None and local_dist_bins is None:
            raise ValueError(
                "merge_op='deepsets' requires a positional encoding in the local "
                "window attention (local_dist_sq or local_dist_bins): the "
                "permutation-invariant merge discards the child order, so without "
                "positional information the network retains no relative geometry at all."
            )
        if local_dist_sq is not None and local_dist_bins is not None:
            raise ValueError(
                "local_dist_sq and local_dist_bins are mutually exclusive — pass one " "positional encoding, not both."
            )

        # local_dist_sq: optional per-stage (S, S) normalized squared geodesic distances
        # between window tokens; stage i needs S = 4 ** min(window_levels,
        # body_levels - i). With a patchified stem the tables describe the body geometry
        # (one per body stage, starting stem_levels above the finest level). Enables the
        # distance-kernel attention bias.
        if local_dist_sq is not None:
            if len(local_dist_sq) != body_levels:
                raise ValueError(
                    f"local_dist_sq must have one table per local stage " f"({body_levels}), got {len(local_dist_sq)}."
                )
            for level, table in enumerate(local_dist_sq):
                expected = 4 ** min(window_levels, body_levels - level)
                if len(table) != expected:
                    raise ValueError(f"local_dist_sq[{level}] has {len(table)} tokens, " f"expected {expected}.")

        # local_dist_bins: optional per-stage (bin_idx (S, S) int, bin_centers (B,))
        # tuples with the same per-stage S; selects the distance-binned learnable bias
        # (GeodesicBinnedBiasAttention) instead of the scalar distance kernel.
        # pos_coeff_init sets the (RBF) init of whichever bias is enabled.
        if local_dist_bins is not None:
            if len(local_dist_bins) != body_levels:
                raise ValueError(
                    f"local_dist_bins must have one (bin_idx, bin_centers) per local "
                    f"stage ({body_levels}), got {len(local_dist_bins)}."
                )
            for level, (bin_idx, bin_centers) in enumerate(local_dist_bins):
                expected = 4 ** min(window_levels, body_levels - level)
                bin_idx = np.asarray(bin_idx)
                if bin_idx.shape != (expected, expected):
                    raise ValueError(
                        f"local_dist_bins[{level}] bin_idx has shape {bin_idx.shape}, "
                        f"expected ({expected}, {expected})."
                    )
                if bin_idx.max() >= len(bin_centers):
                    raise ValueError(
                        f"local_dist_bins[{level}] bin_idx references bin "
                        f"{bin_idx.max()} but only {len(bin_centers)} bin_centers "
                        f"were given."
                    )

        # token_valid: optional (N * 4^num_nested_levels,) boolean validity of the finest
        # tokens in nested row-major order (True = observed pixel). Enables masked
        # attention: masked tokens are excluded from every attention softmax, zeroed
        # before every patch merge and excluded from the final pool, so the output is
        # independent of their values. Static config geometry (like the smoothing mask),
        # so the whole validity pyramid is precomputed as constants and no checkpoint
        # variables are added — toggling it keeps the checkpoint lineage.
        stage_window_masks = None
        pool_key_mask = None
        if token_valid is not None:
            token_valid = np.asarray(token_valid).astype(bool).reshape(-1)
            block_size = 4**num_nested_levels
            if len(token_valid) % block_size != 0 or len(token_valid) == 0:
                raise ValueError(
                    f"token_valid length {len(token_valid)} is not a nonzero multiple "
                    f"of 4^num_nested_levels = {block_size}."
                )
            num_top_tokens = len(token_valid) // block_size

            # validity pyramid: a parent token is valid if ANY of its 4 children is
            valid_levels = [token_valid]
            for _ in range(num_nested_levels):
                valid_levels.append(valid_levels[-1].reshape(-1, 4).any(axis=1))
            if not valid_levels[-1].any():
                raise ValueError("token_valid masks out every top-level token.")

            # pre-stem zeroing multiplier (1, N, 4, ..., 4, 1) over all num_nested_levels
            # axes: with a patchified stem, masked fine pixels must be zeroed before they
            # are linearly mixed into a patch, so the output stays independent of their
            # values. (Without a stem the level-0 multiplier below plays this role.)
            self._stem_valid = (
                tf.constant(
                    valid_levels[0].reshape(1, num_top_tokens, *([4] * num_nested_levels), 1),
                    dtype=tf.float32,
                )
                if stem_levels > 0
                else None
            )

            # per-stage key-side window attention masks (n_windows, S, S): a token may be
            # attended to iff it is valid. Invalid queries produce finite garbage (uniform
            # softmax) that is never read — it is zeroed before the next patch merge.
            # With a patchified stem, stage i operates at pyramid level stem_levels + i.
            stage_window_masks = []
            for level in range(body_levels):
                seq_len = 4 ** min(window_levels, body_levels - level)
                windows = valid_levels[stem_levels + level].reshape(-1, seq_len)
                stage_window_masks.append(np.broadcast_to(windows[:, None, :], (len(windows), seq_len, seq_len)))

            # per-stage zeroing multipliers (1, N, 4, ..., 4, 1), applied before merges
            self._stage_valid = [
                tf.constant(
                    valid_levels[stem_levels + level].reshape(1, num_top_tokens, *([4] * (body_levels - level)), 1),
                    dtype=tf.float32,
                )
                for level in range(body_levels)
            ]
            # per-stage valid-token counts for the masked multi-scale readout means
            self._stage_pool_count = [float(valid_levels[stem_levels + level].sum()) for level in range(body_levels)]
            # global-stage key-side mask (1, N, N) and masked mean-pool weights
            top_valid = valid_levels[-1]
            self._global_mask = tf.constant(
                np.broadcast_to(top_valid[None, None, :], (1, num_top_tokens, num_top_tokens))
            )
            self._pool_weights = tf.constant(top_valid[None, :, None], dtype=tf.float32)
            self._pool_count = float(top_valid.sum())
            # key-side mask (1, 1, N) for the attention pool's learned queries
            pool_key_mask = top_valid[None, None, :]
        else:
            self._stem_valid = None
            self._stage_valid = None
            self._stage_pool_count = None
            self._global_mask = None
            self._pool_weights = None
            self._pool_count = None

        self.in_channels = in_channels
        self.num_outputs = num_outputs
        self.num_nested_levels = num_nested_levels
        self.stem_levels = stem_levels
        self.body_levels = body_levels
        self.base_embed_dim = base_embed_dim
        self.growth = growth
        self.num_heads = num_heads
        self.window_levels = window_levels
        self.local_blocks_per_level = local_blocks_per_level
        self.num_global_blocks = global_blocks
        self.mlp_ratio = mlp_ratio
        self.layerscale_init = layerscale_init
        self.fp32_softmax = fp32_softmax
        self.head_dropout_rate = head_dropout_rate
        # block_dropout_rate: residual-branch dropout inside every local and global
        # TransformerBlock (see there). Variable-free — toggling keeps the checkpoint lineage.
        self.block_dropout_rate = block_dropout_rate
        self.merge_op = merge_op
        self.pool = pool
        self.pool_queries = pool_queries
        self.multiscale_readout = multiscale_readout
        self.local_dist_sq = local_dist_sq
        self.token_valid = token_valid

        # Channel dimensions at each resolution level of the hierarchy (after the stem).
        # Length is body_levels + 1.
        self.channel_dims = make_channel_dims(
            base_embed_dim=base_embed_dim,
            num_nested_levels=body_levels,
            growth=growth,
        )

        for dim in self.channel_dims:
            if dim % num_heads != 0:
                raise ValueError(f"Channel dimension {dim} must be divisible by num_heads={num_heads}.")

        # Project the input features -> base_embed_dim, applied independently to every
        # entry token of the hierarchy. Without a stem the input features are the C map
        # channels of one fine token; with a patchified stem they are the flattened
        # 4^stem_levels * C values of one patch.
        self.input_proj = tf.keras.layers.Dense(self.channel_dims[0])

        # One local stage per hierarchy level.
        # Stage i operates before merge i.
        # Its channel dimension is channel_dims[i].
        local_stages = []
        for level in range(body_levels):
            dim = self.channel_dims[level]
            local_stages.append(
                [
                    NestedLocalWindowBlock(
                        dim=dim,
                        num_heads=num_heads,
                        window_levels=window_levels,
                        mlp_ratio=mlp_ratio,
                        layerscale_init=layerscale_init,
                        fp32_softmax=fp32_softmax,
                        dist_sq=None if local_dist_sq is None else local_dist_sq[level],
                        dist_bins=(None if local_dist_bins is None else local_dist_bins[level]),
                        pos_coeff_init=pos_coeff_init,
                        window_mask=(None if stage_window_masks is None else stage_window_masks[level]),
                        block_dropout_rate=block_dropout_rate,
                        drop_path_rate=drop_path_rate,
                        name=f"local_stage_{level}_block_{block_index}",
                    )
                    for block_index in range(local_blocks_per_level)
                ]
            )
        self.local_stages = local_stages

        # One patch merge per hierarchy level.
        # Merge i maps channel_dims[i] -> channel_dims[i + 1].
        self.patch_merges = [
            merge_classes[merge_op](
                in_dim=self.channel_dims[level],
                out_dim=self.channel_dims[level + 1],
                name=f"patch_merge_{level}",
            )
            for level in range(body_levels)
        ]

        # injections: optional secondary inputs entering the hierarchy at a coarser body
        # level than the main (finest) input — e.g. a probe whose maps live at a lower nside
        # (clustering @256 vs lensing @512) joins the residual stream at the body level that
        # already runs at that nside, so the network is one level deeper for the fine probe.
        # Each entry is {"level": body-loop level L (1..body_levels-1), "in_channels": C_inj};
        # the tensor arrives already tiling the same N top-level tokens with body_levels-L
        # nested axes. At the start of level L its C_inj features are projected to
        # channel_dims[L], concatenated with the merged fine stream, and a Dense fuses the
        # concat back to channel_dims[L].
        #
        # NOTE — injection_proj is currently REDUNDANT, and deliberately kept. Nothing sits
        # between it and injection_fuse (see call()), so the pair collapses exactly:
        #   fuse(concat([x, W_p r + b_p])) = W_x x + (W_i W_p) r + (W_i b_p + b_f),
        # i.e. one Dense (channel_dims[L] + C_inj) -> channel_dims[L] with the same rank
        # ceiling min(channel_dims[L], C_inj). It costs channel_dims[L]^2 + channel_dims[L]
        # parameters (4160 at the combined prod geometry, L=1) and buys no expressiveness.
        # It is kept because it becomes load-bearing the moment the fusion is not a plain
        # concat: the GCNN twin's fusion="bilinear" feeds x*inj, which needs both streams at
        # the same width and is not even shape-compatible without the projection. Keeping the
        # idiom identical in both encoders is what makes that knob portable here. See
        # deep_lss.nets.encoders.maps.gcnn.resnet_multires (fusion / fuse_act), where the same
        # pair lives and both nonlinearity knobs measured a wash (bench_v4, bench_v8).
        #
        # This is NOT the patch-merge idiom, despite the resemblance: NestedPatchMerge4 is
        # LayerNorm -> concat -> ONE Dense. The injection seam has two Denses and no norm.
        #
        # Masked attention (token_valid) is not supported alongside injections.
        self.injections_spec = list(injections or [])
        if self.injections_spec and token_valid is not None:
            raise ValueError("injections are not supported together with masked attention (token_valid).")
        self.injection_proj = {}
        self.injection_fuse = {}
        self._injection_channels = {}
        for inj in self.injections_spec:
            level = int(inj["level"])
            if not (1 <= level <= body_levels - 1):
                raise ValueError(
                    f"injection level {level} out of range: expected 1..{body_levels - 1} "
                    f"(body_levels={body_levels})."
                )
            if str(level) in self.injection_proj:
                raise ValueError(f"duplicate injection at level {level}.")
            dim = self.channel_dims[level]
            self.injection_proj[str(level)] = tf.keras.layers.Dense(dim, name=f"injection_proj_{level}")
            self.injection_fuse[str(level)] = tf.keras.layers.Dense(dim, name=f"injection_fuse_{level}")
            self._injection_channels[level] = int(inj["in_channels"])

        # Final global attention over the N basic-patch tokens.
        final_dim = self.channel_dims[-1]
        self.global_blocks = [
            TransformerBlock(
                dim=final_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                layerscale_init=layerscale_init,
                fp32_softmax=fp32_softmax,
                block_dropout_rate=block_dropout_rate,
                drop_path_rate=drop_path_rate,
                name=f"global_block_{block_index}",
            )
            for block_index in range(global_blocks)
        ]

        # NO pre-pool LayerNorm: the readout is POOL -> NORM, matching the GCNN twin and
        # ConvNeXt's classifier rather than ViT's norm -> pool. Not equivalent -- norming first
        # equalizes each token's magnitude, and amplitude across sky positions is signal here.
        # The norm the head needs comes after the pool, from TransformerSummaryNetwork.
        # !! LINEAGE BREAK (2026-08-31): a variable was removed, so older checkpoints hard-error.

        # Learned-query attention pool over the N top-level tokens (see AttentionPool);
        # None reproduces the (masked) mean pool.
        self.attention_pool = (
            AttentionPool(
                dim=final_dim,
                num_heads=num_heads,
                num_queries=pool_queries,
                fp32_softmax=fp32_softmax,
                key_mask=pool_key_mask,
                name="attention_pool",
            )
            if pool == "attention"
            else None
        )
        # multiscale_readout: give the head direct access to every scale — the (masked)
        # mean of each stage's tokens right before its merge, one fp32 LayerNorm per
        # stage to homogenize the feature scales, concatenated with the pooled top-level
        # feature. Without it, small-scale information must survive every merge to reach
        # the head at all.
        self.readout_norms = (
            [_fp32_layer_norm(axis=-1, epsilon=1e-5, name=f"readout_norm_{level}") for level in range(body_levels)]
            if multiscale_readout
            else None
        )
        # Dropout on the pooled feature vector, right before the final linear layer — the same
        # position as the post-fusion head dropout in the maps+cls networks. Variable-free, so
        # toggling it keeps checkpoints compatible.
        self.head_dropout = tf.keras.layers.Dropout(head_dropout_rate) if head_dropout_rate is not None else None
        # num_outputs=None returns the pooled feature as-is. The summary networks follow this with
        # LayerNorm -> Dropout -> Dense(out_features), so a projection here would be a second linear
        # layer with no nonlinearity between the two -- it only earns its parameters when the width
        # itself matters, i.e. when a Cls branch is concatenated onto this feature.
        self.head = tf.keras.layers.Dense(num_outputs) if num_outputs is not None else None

        window_sizes = [4 ** min(window_levels, body_levels - level) for level in range(body_levels)]
        head_input_dim = self.channel_dims[-1] * (pool_queries if pool == "attention" else 1)
        if multiscale_readout:
            head_input_dim += sum(self.channel_dims[:-1])
        LOGGER.warning(
            f"NestedHierarchicalLocalWindowTransformer: {num_nested_levels} nested levels"
            + (
                f" = patchified stem (stem_levels={stem_levels}, embedding "
                f"{4 ** stem_levels} pixels x {in_channels} channels = "
                f"{(4 ** stem_levels) * in_channels} features per patch) + "
                f"{body_levels} body levels"
                if stem_levels > 0
                else ""
            )
            + f", channel_dims={list(self.channel_dims)} (base_embed_dim={base_embed_dim}, "
            f"growth={growth!r}), tokens per local window={window_sizes} "
            f"(window_levels={window_levels}), local_blocks_per_level="
            f"{local_blocks_per_level}, global_blocks={global_blocks}, "
            f"num_heads={num_heads}, mlp_ratio={mlp_ratio}, num_outputs={num_outputs}"
        )
        LOGGER.warning(
            f"NestedHierarchicalLocalWindowTransformer options: "
            f"pos_encoding="
            f"{'geodesic' if local_dist_sq is not None else 'geodesic_binned' if local_dist_bins is not None else None}"
            f" (coeff_init={pos_coeff_init}), "
            f"merge_op={merge_op!r}, pool={pool!r}"
            + (f" ({pool_queries} learned queries)" if pool == "attention" else "")
            + f", multiscale_readout={multiscale_readout}, "
            f"masked_attention={token_valid is not None}, "
            f"head input width={head_input_dim}, layerscale_init={layerscale_init}, "
            f"fp32_softmax={fp32_softmax}, block_dropout_rate={block_dropout_rate}, "
            f"drop_path_rate={drop_path_rate}, "
            f"head_dropout_rate={head_dropout_rate}"
        )
        if token_valid is not None:
            LOGGER.warning(
                "Masked attention: valid tokens per body stage="
                f"{[int(count) for count in self._stage_pool_count]}, "
                f"valid top-level tokens={int(self._pool_count)}"
            )

    def call(self, x, training=None, injections=None):
        """
        Input:
            x: (B, C, N, 4, 4, ..., 4)
            injections: optional {body-loop level L: (B, C_inj, N, 4, ..., 4)} with
                body_levels - L nested axes, tiling the same N top-level tokens as x.

        Output:
            y: (B, num_outputs), or (B, final_dim) when num_outputs is None
        """
        expected_ndim = 3 + self.num_nested_levels
        rank = x.shape.rank

        if rank is not None and rank != expected_ndim:
            raise ValueError(
                f"Expected input with {expected_ndim} dims: " f"(B, C, N, 4, ..., 4), got shape {tuple(x.shape)}."
            )

        assertions = []
        if rank is None:
            assertions.append(
                tf.debugging.assert_equal(
                    tf.rank(x),
                    expected_ndim,
                    message=(f"Expected input with {expected_ndim} dims: " "(B, C, N, 4, ..., 4)."),
                )
            )

        assertions.append(_maybe_assert_dim(x, 1, self.in_channels, "input channel dimension", rank=expected_ndim))
        for axis in range(3, expected_ndim):
            assertions.append(_maybe_assert_dim(x, axis, 4, "nested resolution dimension", rank=expected_ndim))
        x = _apply_assertions(x, assertions)

        input_static_shape = x.shape.as_list() if x.shape.rank is not None else None

        # Move channels to the end:
        #   (B, C, N, 4, 4, ..., 4) -> (B, N, 4, 4, ..., 4, C)
        perm = [0] + list(range(2, expected_ndim)) + [1]
        x = tf.transpose(x, perm=perm)
        batch_dim = input_static_shape[0] if input_static_shape is not None else None
        token_dim = input_static_shape[2] if input_static_shape is not None else None
        x.set_shape([batch_dim, token_dim, *([4] * self.num_nested_levels), self.in_channels])

        # Move each injection's channels to the end as well, mirroring the main input:
        #   (B, C_inj, N, 4, ..., 4) -> (B, N, 4, ..., 4, C_inj)
        injections = injections or {}
        if set(injections) != set(self._injection_channels):
            raise ValueError(
                f"injections keys {sorted(injections)} do not match the configured "
                f"injection levels {sorted(self._injection_channels)}."
            )
        injections_cl = {}
        for level, inj in injections.items():
            inj_ndim = 3 + (self.body_levels - level)
            inj_perm = [0] + list(range(2, inj_ndim)) + [1]
            inj = tf.transpose(inj, perm=inj_perm)
            inj.set_shape([batch_dim, token_dim, *([4] * (self.body_levels - level)), self._injection_channels[level]])
            injections_cl[level] = inj

        if self.stem_levels > 0:
            if self._stem_valid is not None:
                # zero masked fine pixels before they are linearly mixed into a patch
                x = x * tf.cast(self._stem_valid, x.dtype)
            # Patchified stem: fold the finest stem_levels nested axes into the feature
            # dimension (child-major, channel-minor — trailing axes, so a pure reshape):
            #   (B, N, 4 x M, C) -> (B, N, 4 x (M - s), 4^s * C)
            stem_features = (4**self.stem_levels) * self.in_channels
            prefix_shape = tf.shape(x)[: 2 + self.body_levels]
            x = tf.reshape(
                x,
                tf.concat(
                    [prefix_shape, tf.constant([stem_features], dtype=prefix_shape.dtype)],
                    axis=0,
                ),
            )
            x.set_shape([batch_dim, token_dim, *([4] * self.body_levels), stem_features])

        # Project the entry features -> base_embed_dim:
        #   (B, N, 4, 4, ..., 4, C or 4^s * C) -> (B, N, 4, 4, ..., 4, D0)
        x = self.input_proj(x)

        # Hierarchical local processing.
        readout_features = []
        for level in range(self.body_levels):
            # Inject a coarser secondary input that enters at this body level: project its
            # channels to channel_dims[level], concatenate with the merged fine stream, and
            # fuse back to channel_dims[level] before the local attention sees the level.
            # Nothing between proj and fuse, so the two are one linear map -- redundant here
            # and kept only so a non-concat fusion stays portable from the GCNN twin; see the
            # NOTE next to their construction in __init__. The fused output IS the residual
            # stream, and the following pre-norm block supplies the nonlinearity, so this seam
            # is not the "entirely LINEAR seam" that resnet_multires's fuse_act was added for.
            if level in injections_cl:
                inj = self.injection_proj[str(level)](injections_cl[level])
                x = self.injection_fuse[str(level)](tf.concat([x, inj], axis=-1))

            for block in self.local_stages[level]:
                x = block(x, training=training)

            if self._stage_valid is not None:
                # zero masked tokens so the merge only mixes valid children (their
                # in-window attention output is finite garbage — see NestedLocalWindowBlock)
                x = x * tf.cast(self._stage_valid[level], x.dtype)

            if self.readout_norms is not None:
                # multi-scale readout tap: (masked) mean of this stage's tokens. Masked
                # tokens are exactly zero here, so sum / valid-count is the masked mean.
                token_axes = list(range(1, x.shape.rank - 1))
                if self._stage_valid is not None:
                    stage_feature = tf.reduce_sum(x, axis=token_axes) / tf.cast(self._stage_pool_count[level], x.dtype)
                else:
                    stage_feature = tf.reduce_mean(x, axis=token_axes)
                readout_features.append(self.readout_norms[level](stage_feature))

            x = self.patch_merges[level](x, training=training)

        # After all merges:
        #   x: (B, N, final_dim)
        # Apply final global attention over N tokens.
        for block in self.global_blocks:
            x = block(x, training=training, attention_mask=self._global_mask)

        # fp32 BEFORE pooling, not optional: the pre-norm residual stream grows with depth, and
        # averaging N of those in bf16 loses precision. The removed LayerNorm supplied this cast
        # as a side effect, so every pooling branch below sees the dtype it saw before.
        x = tf.cast(x, tf.float32)

        # Pool over the N basic patches:
        #   (B, N, final_dim) -> (B, final_dim) or (B, pool_queries * final_dim)
        if self.attention_pool is not None:
            x = self.attention_pool(x, training=training)
        elif self._pool_weights is not None:
            # masked mean over the valid top-level tokens only
            x = tf.reduce_sum(x * tf.cast(self._pool_weights, x.dtype), axis=1)
            x = x / tf.cast(self._pool_count, x.dtype)
        else:
            x = tf.reduce_mean(x, axis=1)

        if self.readout_norms is not None:
            # concatenate the per-stage readout features (fp32 LayerNorm outputs) with
            # the pooled top-level feature; only the head input widens
            x = tf.concat([tf.cast(f, x.dtype) for f in readout_features] + [x], axis=-1)

        if self.head_dropout is not None:
            x = self.head_dropout(x, training=training)

        # Optional output projection:
        #   (B, final_dim) -> (B, num_outputs); identity when num_outputs is None
        if self.head is not None:
            x = self.head(x)

        return x
