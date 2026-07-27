# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created July 2026
Author: Arne Thomsen

Global self-attention over the coarsest-nside pixel tokens of the DeepSphere conv body.

``HealpyGlobalAttention`` is a standard pre-LN transformer encoder block (optionally stacked
``n_layers`` deep) applied to the ``(B, N, C)`` feature sequence at the END of the graph-conv
stack, where N is the number of footprint pixels at the coarsest nside (~448 at nside 16) and C
is the deep channel width. It gives the otherwise purely-local Chebyshev/Bernstein graph-conv
body a single all-to-all mixing stage, in direct analogy to the global-attention stage of the
HEALPix nested transformer (``encoders/maps/transformer/healpix_transformer.py``,
``nested_transformer.py``), which the DeepSphere combined encoder loses to.

Design choices:

- **Channel-preserving drop-in.** Input and output are both ``(B, N, C)`` with C unchanged, so the
  block is simply appended to ``ResNetLayers``' conv-layer list: ``HealpyGCNN`` routes any
  unrecognized layer through its passthrough branch and only counts pooling layers toward the nside
  reduction, so this block never disturbs the graph / nside bookkeeping. It sits after the residual
  convs and before the head's flatten (maps-only) or the composite's flatten/pool (maps+cls).

- **Learnable per-token positional embedding.** The N footprint pixels are a FIXED sky layout
  (same pixels every sample), so a learned ``(N, C)`` table added once at the input is well-defined
  and gives the permutation-invariant attention full positional information — the ViT idiom, and a
  cleaner analogue here than the transformer's geodesic distance bias (whose windows tile a moving
  footprint). Zero-initialized so the block starts position-agnostic and learns position as needed.

- **LayerScale, small init.** Each residual branch is scaled by a learnable per-channel gamma
  initialized to ``layer_scale_init`` (default 1e-4), so the block starts as a near-identity map:
  the pretrained-from-scratch conv features pass through essentially unperturbed and attention is
  learned as a correction. This is the same stabilizer that fixed deep-stack NaNs in the transformer
  benchmark (see project_transformer_hparam_benchmark). Set ``layer_scale_init: null`` to disable.

The block is NOT rotation-equivariant (global attention relates arbitrary sky positions), by design
and only at the coarsest nside — the equivariant local graph convs still do all the fine-scale work.
"""

import tensorflow as tf

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


class HealpyGlobalAttention(tf.keras.layers.Layer):
    """Global multi-head self-attention encoder block(s) over ``(B, N, C)`` pixel tokens.

    Args:
        num_heads (int): number of attention heads. Defaults to 4.
        key_dim (int, optional): per-head key/query dimension. Defaults to ``C // num_heads``
            (inferred at build), i.e. the standard "split the channels across heads" sizing.
        mlp_ratio (float): hidden width of the per-block MLP as a multiple of C. Defaults to 2.0.
        n_layers (int): number of stacked encoder blocks. Defaults to 1.
        dropout_rate (float, optional): dropout inside attention and the MLP. Defaults to 0.0.
        positional_embedding (bool): add a learnable ``(N, C)`` positional embedding at the input.
            Defaults to True.
        layer_scale_init (float, optional): LayerScale init for both residual branches; ``None``
            disables LayerScale (branches added at unit weight). Defaults to 1e-4.
        activation (str): MLP activation. Defaults to "gelu".
    """

    def __init__(
        self,
        num_heads=4,
        key_dim=None,
        mlp_ratio=2.0,
        n_layers=1,
        dropout_rate=0.0,
        positional_embedding=True,
        layer_scale_init=1e-4,
        activation="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mlp_ratio = mlp_ratio
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.positional_embedding = positional_embedding
        self.layer_scale_init = layer_scale_init
        self.activation = activation
        # NOTE: deliberately no ``Fout`` attribute — HealpyGCNN / split_layers_at_nside read
        # ``layer.Fout`` to track the channel count; leaving it unset keeps the current width
        # (attention preserves C), which is exactly right for a channel-preserving block.

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"HealpyGlobalAttention expects (B, N, C) input, got {input_shape}")
        n_tokens, channels = input_shape[1], input_shape[2]
        if channels is None:
            raise ValueError("HealpyGlobalAttention needs a static channel dimension (C).")
        key_dim = self.key_dim if self.key_dim is not None else max(channels // self.num_heads, 1)
        hidden = int(round(self.mlp_ratio * channels))

        if self.positional_embedding:
            if n_tokens is None:
                raise ValueError(
                    "positional_embedding=True needs a static token count N; got a dynamic pixel "
                    "axis. Disable it or build the block with a known footprint length."
                )
            self.pos_emb = self.add_weight(
                name="pos_emb",
                shape=(1, n_tokens, channels),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.pos_emb = None

        self.blocks = []
        for i in range(self.n_layers):
            norm1 = tf.keras.layers.LayerNormalization(axis=-1, name=f"ln_attn_{i}")
            attn = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=key_dim,
                dropout=self.dropout_rate,
                name=f"mha_{i}",
            )
            norm2 = tf.keras.layers.LayerNormalization(axis=-1, name=f"ln_mlp_{i}")
            mlp = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(hidden, activation=self.activation, name=f"mlp_hidden_{i}"),
                    tf.keras.layers.Dropout(self.dropout_rate),
                    tf.keras.layers.Dense(channels, name=f"mlp_out_{i}"),
                ],
                name=f"mlp_{i}",
            )
            gamma_attn = gamma_mlp = None
            if self.layer_scale_init is not None:
                gamma_attn = self.add_weight(
                    name=f"ls_attn_{i}",
                    shape=(channels,),
                    initializer=tf.keras.initializers.Constant(self.layer_scale_init),
                    trainable=True,
                )
                gamma_mlp = self.add_weight(
                    name=f"ls_mlp_{i}",
                    shape=(channels,),
                    initializer=tf.keras.initializers.Constant(self.layer_scale_init),
                    trainable=True,
                )
            self.blocks.append((norm1, attn, norm2, mlp, gamma_attn, gamma_mlp))

        LOGGER.warning(
            f"HealpyGlobalAttention: {self.n_layers} block(s), N={n_tokens} tokens, C={channels}, "
            f"{self.num_heads} heads x key_dim {key_dim}, mlp_ratio {self.mlp_ratio}, "
            f"pos_emb={self.positional_embedding}, layer_scale_init={self.layer_scale_init}"
        )
        super().build(input_shape)

    def call(self, x, training=False):
        if self.pos_emb is not None:
            x = x + self.pos_emb
        for norm1, attn, norm2, mlp, gamma_attn, gamma_mlp in self.blocks:
            h = norm1(x)
            h = attn(h, h, training=training)
            x = x + (h if gamma_attn is None else gamma_attn * h)
            h = mlp(norm2(x), training=training)
            x = x + (h if gamma_mlp is None else gamma_mlp * h)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "mlp_ratio": self.mlp_ratio,
                "n_layers": self.n_layers,
                "dropout_rate": self.dropout_rate,
                "positional_embedding": self.positional_embedding,
                "layer_scale_init": self.layer_scale_init,
                "activation": self.activation,
            }
        )
        return config
