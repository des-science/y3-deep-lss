# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2024
Author: Arne Thomsen
"""

import tensorflow as tf

from deepsphere import healpy_layers
from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


class MeanLayer(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(MeanLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=self.axis)


def get_regression_head(
    out_features,
    head_type="dense",
    # dense
    second_to_last_features=None,
    activation="relu",
    dropout_rate=None,
    # convolutional
    poly_degree=5,
    norm_kwargs={},
):
    layers = []

    if head_type == "dense":
        LOGGER.info("Using a dense regression head")

        layers.append(tf.keras.layers.Flatten())
        layers.append(tf.keras.layers.LayerNormalization(axis=-1))

        if second_to_last_features is not None:
            LOGGER.warning("Including a second to last dense layer in the regression head")
            # the sliced Wasserstein penalty is easier with this extra dense layer
            layers.append(tf.keras.layers.Dense(16 * second_to_last_features, activation=activation))
            layers.append(tf.keras.layers.LayerNormalization(axis=-1))
            # no activation such that the z_features can be negative for a standard distribution
            layers.append(tf.keras.layers.Dense(second_to_last_features))

        if dropout_rate is not None:
            assert not second_to_last_features, "Dropout and second to last features should not be used together"
            LOGGER.warning(f"Using dropout with probability {dropout_rate} in the regression head")
            layers.append(tf.keras.layers.Dropout(dropout_rate))

        layers.append(tf.keras.layers.Dense(out_features))

    elif head_type == "conv":
        assert not second_to_last_features, "Second to last features not supported for convolutional head"
        assert dropout_rate is None, "Dropout not supported for convolutional head"

        LOGGER.info("Using a convolutional + averaging regression head")

        layers.append(tf.keras.layers.LayerNormalization(axis=-1, **norm_kwargs))
        layers.append(healpy_layers.HealpyChebyshev(K=poly_degree, Fout=out_features, activation=None))
        layers.append(MeanLayer(axis=-2, dtype=tf.float32))

    else:
        raise ValueError(f"Unknown regression head type: {head_type}")

    return layers
