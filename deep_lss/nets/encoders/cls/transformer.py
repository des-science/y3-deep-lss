# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Transformer summary encoder for binned power spectra (Cls).

Treats each auto/cross redshift-bin pair as a TOKEN (a length-``cls_n_bins`` power-spectrum curve)
and runs self-attention over the pair tokens. This targets the physics: the cosmological
information in Cls lives largely in how spectra relate ACROSS tomographic bins, so attention over
pairs with weight-sharing (one shared per-pair tokenizer) is a natural inductive bias, and it
generalizes across probe configs (lensing-only / 3x2pt / combined) with the same weights.

Attention is permutation-equivariant, so each token is given its STRUCTURED tomographic identity:
learned embeddings of the two global redshift-bin indices (z_i, z_j) — shared table over all
n_z bins — plus a probe-pair-type embedding (WLxWL / WLxGC / GCxGC). These let attention discover
redshift-locality (adjacent tomographic bins have correlated signal) without imposing a fixed grid.

Input contract matches MultiLayerPerceptron: a flat ``(B, n_cls)`` vector (bin-major / pair-minor,
``n_cls = cls_n_bins * n_pairs``) reshaped internally to ``(B, cls_n_bins, n_pairs)`` then
transposed to pair-tokens ``(B, n_pairs, cls_n_bins)``. The per-feature ``input_transform``
(AsinhScaleLayer) is applied on the flat vector first (structure-preserving); PCA whitening is not
supported and is rejected by the caller.

Author: Arne Thomsen
"""

import tensorflow as tf


class _LayerScale(tf.keras.layers.Layer):
    """Per-channel learnable scale (Touvron et al. 2021), small init to stabilize deep stacks."""

    def __init__(self, dim, init_value, **kwargs):
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

    def call(self, x):
        return x * self.gamma


class _EncoderBlock(tf.keras.layers.Layer):
    """Pre-norm transformer encoder block: MHSA + MLP, with optional LayerScale."""

    def __init__(self, d_model, num_heads, mlp_ratio, dropout_rate, activation, layerscale_init, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=max(d_model // num_heads, 1), dropout=dropout_rate
        )
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(int(d_model * mlp_ratio), activation=activation),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(d_model),
                tf.keras.layers.Dropout(dropout_rate),
            ]
        )
        self.ls1 = _LayerScale(d_model, layerscale_init) if layerscale_init is not None else None
        self.ls2 = _LayerScale(d_model, layerscale_init) if layerscale_init is not None else None

    def call(self, x, training=False):
        h = self.norm1(x)
        a = self.attn(h, h, training=training)
        a = self.ls1(a) if self.ls1 is not None else a
        x = x + a
        h = self.norm2(x)
        m = self.mlp(h, training=training)
        m = self.ls2(m) if self.ls2 is not None else m
        return x + m


class ClsTransformer(tf.keras.Model):
    def __init__(
        self,
        output_size,
        cls_n_bins,
        n_pairs,
        pair_zi,
        pair_zj,
        pair_ptype,
        n_z,
        input_transform=None,
        input_norm=False,
        d_model=64,
        num_heads=4,
        num_layers=3,
        mlp_ratio=4,
        dropout_rate=0.0,
        activation="gelu",
        stem="mlp",
        pool="mean",
        layerscale_init=None,
        head_units=None,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.cls_n_bins = cls_n_bins
        self.n_pairs = n_pairs
        self.d_model = d_model
        self.pool = pool
        act = tf.keras.activations.get(activation)

        # Optional per-pair-token input normalization, for parity with MultiLayerPerceptron and
        # ClsConv1D (which both normalize before their first learned layer). Applied after the
        # transpose to pair-tokens (B, n_pairs, cls_n_bins) with axis=-1, i.e. each pair's
        # length-cls_n_bins curve is normalized on its own before tokenization. Default OFF so
        # existing cls_transformer checkpoints (object-based) keep an unchanged variable lineage.
        self.input_norm = tf.keras.layers.LayerNormalization(axis=-1) if input_norm else None

        assert len(pair_zi) == len(pair_zj) == len(pair_ptype) == n_pairs, (
            f"pair identity arrays (len {len(pair_zi)}/{len(pair_zj)}/{len(pair_ptype)}) must match "
            f"n_pairs={n_pairs}; the identity ordering must align with the flat Cls pair axis."
        )
        # Fixed per-run identity (non-trainable buffers), aligned with the pair axis of the input.
        self.pair_zi = tf.constant(pair_zi, dtype=tf.int32)
        self.pair_zj = tf.constant(pair_zj, dtype=tf.int32)
        self.pair_ptype = tf.constant(pair_ptype, dtype=tf.int32)

        # Per-pair tokenizer: shared across pairs, maps a length-cls_n_bins curve to d_model.
        self.stem = stem
        if stem == "mlp":
            self.token_proj = tf.keras.layers.Dense(d_model, activation=act)
        elif stem == "conv":
            self.token_conv = tf.keras.layers.Conv1D(d_model, 3, padding="same", activation=act)
        else:
            raise ValueError(f"Unknown stem={stem!r} (expected 'mlp' or 'conv')")

        # Structured tomographic identity embeddings (added to each token).
        self.z_embed = tf.keras.layers.Embedding(n_z, d_model)  # shared for z_i and z_j
        self.ptype_embed = tf.keras.layers.Embedding(3, d_model)

        if pool == "cls":
            self.cls_token = self.add_weight(
                name="cls_token",
                shape=(1, 1, d_model),
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                trainable=True,
            )

        self.blocks = [
            _EncoderBlock(d_model, num_heads, mlp_ratio, dropout_rate, act, layerscale_init, name=f"encoder_{i}")
            for i in range(num_layers)
        ]
        self.final_norm = tf.keras.layers.LayerNormalization()
        self.head_dense = [tf.keras.layers.Dense(u, activation=act) for u in (head_units or [])]
        self.output_layer = tf.keras.layers.Dense(output_size, name="output")

    def build(self, input_shape):
        # The (B, n_cls) -> (B, cls_n_bins, n_pairs) reshape in call() requires the flat feature
        # dim to be exactly cls_n_bins * n_pairs. Fail here with a clear message instead of an
        # opaque reshape error if the class is constructed directly with a mismatched n_cls.
        n_cls = input_shape[-1]
        if n_cls is not None:
            expected = self.cls_n_bins * self.n_pairs
            assert n_cls == expected, (
                f"input feature dim {n_cls} != cls_n_bins*n_pairs = {self.cls_n_bins}*{self.n_pairs} "
                f"= {expected}; the (bins, pairs) reshape requires the hard_rebinned layout."
            )
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.input_transform(inputs) if self.input_transform is not None else inputs
        # flat (B, n_cls) -> (B, cls_n_bins, n_pairs) -> tokens (B, n_pairs, cls_n_bins)
        x = tf.reshape(x, (-1, self.cls_n_bins, self.n_pairs))
        x = tf.transpose(x, perm=[0, 2, 1])
        if self.input_norm is not None:
            x = self.input_norm(x)

        # Tokenize each pair-curve to d_model.
        if self.stem == "mlp":
            tokens = self.token_proj(x)  # (B, n_pairs, d_model)
        else:
            b = tf.shape(x)[0]
            xc = tf.reshape(x, (-1, self.cls_n_bins, 1))  # (B*n_pairs, cls_n_bins, 1)
            xc = self.token_conv(xc)  # (B*n_pairs, cls_n_bins, d_model)
            xc = tf.reduce_mean(xc, axis=1)  # pool over bins -> (B*n_pairs, d_model)
            tokens = tf.reshape(xc, (b, self.n_pairs, self.d_model))

        # Add structured identity: z_i + z_j (shared table) + probe-pair-type. Shape (n_pairs, d_model),
        # broadcast over the batch.
        identity = self.z_embed(self.pair_zi) + self.z_embed(self.pair_zj) + self.ptype_embed(self.pair_ptype)
        tokens = tokens + identity[tf.newaxis, :, :]

        if self.pool == "cls":
            b = tf.shape(tokens)[0]
            cls = tf.tile(self.cls_token, [b, 1, 1])  # (B, 1, d_model), no identity added
            tokens = tf.concat([cls, tokens], axis=1)

        for block in self.blocks:
            tokens = block(tokens, training=training)
        tokens = self.final_norm(tokens)

        if self.pool == "mean":
            pooled = tf.reduce_mean(tokens, axis=1)
        elif self.pool == "cls":
            pooled = tokens[:, 0, :]
        else:
            raise ValueError(f"Unknown pool={self.pool!r} (expected 'mean' or 'cls')")

        for dense in self.head_dense:
            pooled = dense(pooled)
        return self.output_layer(pooled)
