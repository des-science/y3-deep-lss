# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
1D convolutional summary encoder for binned power spectra (Cls).

Instead of flattening the rebinned Cls into a data vector (as MultiLayerPerceptron does), this
treats the auto/cross redshift-bin pairs as CHANNELS and the fixed ell-bin axis as the convolution
SEQUENCE. Because every pair is rebinned into the same ``cls_n_bins`` bins over its own
[l_min, l_max] with a shared l_min, bin index ~ normalized (sqrt-ell) scale across pairs, so a
shared 1D filter over the bin axis convolves over normalized scale (locality + weight-sharing).

Input contract matches MultiLayerPerceptron: a flat ``(B, n_cls)`` vector (``n_cls = cls_n_bins *
n_pairs``, bin-major / pair-minor) is reshaped internally to ``(B, cls_n_bins, n_pairs)``. The
optional per-feature ``input_transform`` (AsinhScaleLayer) is applied on the flat vector first (it
is per-feature, hence structure-preserving); PCA whitening is NOT supported (it rotates across all
features and destroys the (bin, pair) structure) and is rejected by the caller.

Author: Arne Thomsen
"""

import tensorflow as tf

from deep_lss.nets.encoders.maps.legacy.one_d_conv import OneDResidualBlock


class ClsConv1D(tf.keras.Model):
    def __init__(
        self,
        output_size,
        cls_n_bins,
        n_pairs,
        input_transform=None,
        base_channels=64,
        num_blocks=3,
        kernel_size=3,
        dropout_rate=0.0,
        activation="relu",
        pool="mean",
        head_units=None,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.cls_n_bins = cls_n_bins
        self.n_pairs = n_pairs
        self.pool = pool
        act = tf.keras.activations.get(activation)

        # LayerNorm over the channel (pair) axis before the stem — the pairs live on very different
        # amplitudes (auto vs cross, WL vs GC), analogous to the MLP's input LayerNorm.
        self.input_norm = tf.keras.layers.LayerNormalization(axis=-1)
        # Stem: project the n_pairs channels to base_channels so the residual blocks (which require
        # matching in/out channels for the skip) can stack.
        self.stem = tf.keras.layers.Conv1D(base_channels, kernel_size, padding="same", activation=act)
        self.blocks = [
            OneDResidualBlock(base_channels, kernel_size, activation=act, name=f"resblock_{i}")
            for i in range(num_blocks)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.flatten = tf.keras.layers.Flatten() if pool == "flatten" else None
        self.head_dense = [tf.keras.layers.Dense(u, activation=act) for u in (head_units or [])]
        self.output_layer = tf.keras.layers.Dense(output_size, name="output")

    def call(self, inputs, training=False):
        x = self.input_transform(inputs) if self.input_transform is not None else inputs
        # flat (B, n_cls) -> (B, cls_n_bins, n_pairs); bin-major / pair-minor matches preprocessing.
        x = tf.reshape(x, (-1, self.cls_n_bins, self.n_pairs))
        x = self.input_norm(x)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x, training=training)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        if self.pool == "mean":
            x = tf.reduce_mean(x, axis=1)
        elif self.pool == "max":
            x = tf.reduce_max(x, axis=1)
        elif self.pool == "flatten":
            x = self.flatten(x)
        else:
            raise ValueError(f"Unknown pool={self.pool!r} (expected 'mean', 'max' or 'flatten')")
        for dense in self.head_dense:
            x = dense(x)
        return self.output_layer(x)
