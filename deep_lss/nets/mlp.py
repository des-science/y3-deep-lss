# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created August 2024
Author: Arne Thomsen
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
    ):
        super(MultiLayerPerceptron, self).__init__()
        if normalization == "layer":
            self.norm_layer = tf.keras.layers.LayerNormalization()
        elif normalization == "batch":
            self.norm_layer = tf.keras.layers.BatchNormalization()
        else:
            raise ValueError(f"Unknown normalization type: {normalization}")

        self.hidden_layers = []
        for _ in range(num_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(num_hidden_units, activation=activation))
            if dropout_rate > 0:
                self.hidden_layers.append(tf.keras.layers.Dropout(dropout_rate))

        if num_penultimate is not None:
            LOGGER.info("Including a penultimate layer in the MLP")
            # self.hidden_layers.append(tf.keras.layers.Dense(num_penultimate, activation=activation))
            self.hidden_layers.append(tf.keras.layers.Dense(num_penultimate))

        self.output_layer = tf.keras.layers.Dense(output_size)

    def call(self, inputs):
        x = self.norm_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
