# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created August 2024
Author: Arne Thomsen

Generic multi-layer perceptron. Used as the power-spectrum (Cls) summary encoder in
``run_cls_training+evaluation.py`` and reused as the critic / theta-embedding block of the
mutual-information estimator in ``deep_lss.utils.mutual_info_loss``. The Cls-specific
preprocessing layers it accepts (``whitening``, ``input_transform``) live in
``deep_lss.nets.layers.cls.whitening`` and are passed in by the caller.
"""

import tensorflow as tf
from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


class MultiLayerPerceptron(tf.keras.Model):
    def __init__(
        self,
        output_size,
        num_hidden_units,
        num_layers,
        num_penultimate=None,
        dropout_rate=0.0,
        normalization="layer",
        activation="relu",
        whitening=None,
        residual=False,
        input_transform=None,
    ):
        super(MultiLayerPerceptron, self).__init__()

        self.input_transform = input_transform
        self.whitening = whitening
        self.residual = residual
        # Skip LayerNorm only when whitening already provides population-level unit variance
        # (whiten=True). With whiten=False the PCA only rotates; eigenvalue spread can be
        # huge, so LayerNorm is still needed to prevent activation explosion.
        skip_norm = whitening is not None and whitening.whiten
        if skip_norm:
            self.norm_layer = None
        elif normalization == "layer":
            self.norm_layer = tf.keras.layers.LayerNormalization()
        elif normalization == "batch":
            self.norm_layer = tf.keras.layers.BatchNormalization()
        else:
            raise ValueError(f"Unknown normalization type: {normalization}")

        # Hidden blocks as (dense, dropout-or-None) pairs so residual skips can be applied
        # cleanly between equal-width layers (the first layer changes width input -> hidden,
        # so it is never a residual block).
        self.hidden_blocks = []
        for _ in range(num_layers):
            dense = tf.keras.layers.Dense(num_hidden_units, activation=activation)
            dropout = tf.keras.layers.Dropout(dropout_rate) if dropout_rate > 0 else None
            self.hidden_blocks.append((dense, dropout))

        # Penultimate (width-changing) layer, never residual.
        if num_penultimate is not None:
            LOGGER.info("Including a penultimate layer in the MLP")
            self.penultimate_layer = tf.keras.layers.Dense(num_penultimate, name="penultimate")
        else:
            self.penultimate_layer = None

        self.output_layer = tf.keras.layers.Dense(output_size, name="output")

    def call(self, inputs, training=False):
        x = self.input_transform(inputs) if self.input_transform is not None else inputs
        x = self.whitening(x) if self.whitening is not None else x
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        for i, (dense, dropout) in enumerate(self.hidden_blocks):
            h = dense(x)
            if dropout is not None:
                h = dropout(h, training=training)
            # Residual skip only from the second hidden layer onward, where input and output
            # share num_hidden_units; the first layer maps input_dim -> num_hidden_units.
            x = x + h if (self.residual and i > 0) else h
        if self.penultimate_layer is not None:
            x = self.penultimate_layer(x)
        return self.output_layer(x)
