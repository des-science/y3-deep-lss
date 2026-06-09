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
    dense_layers=None,
    activation="relu",
    dropout_rate=None,
    # convolutional
    poly_degree=5,
    norm_kwargs={},
):
    """Build the regression head layers.

    The dense head always starts with [Flatten, LayerNorm, …] so that
    ``head_layers_no_flatten = head[1:]`` is valid in ResNetLayers.

    Args:
        out_features: Output dimension.
        head_type: ``"dense"`` or ``"conv"``.
        dense_layers: List of hidden layer widths, e.g. ``[512, 128]``.  Each entry adds
            Dense(hᵢ, activation) → LayerNorm.
        activation: Activation for hidden dense layers.
        dropout_rate: Dropout rate applied after hidden layers (mutually exclusive with hidden layers).
        poly_degree: Chebyshev degree for the convolutional head.
        norm_kwargs: Extra kwargs for LayerNormalization.
    """
    layers = []

    if head_type == "dense":
        LOGGER.info("Using a dense regression head")

        layers.append(tf.keras.layers.Flatten())
        layers.append(tf.keras.layers.LayerNormalization(axis=-1))

        if dense_layers is not None:
            LOGGER.warning(f"Using dense_layers={dense_layers} in the regression head")
            for h in dense_layers:
                layers.append(tf.keras.layers.Dense(h, activation=activation))
                layers.append(tf.keras.layers.LayerNormalization(axis=-1))

        if dropout_rate is not None:
            assert not dense_layers, \
                "Dropout and hidden dense layers should not be used together"
            LOGGER.warning(f"Using dropout with probability {dropout_rate} in the regression head")
            layers.append(tf.keras.layers.Dropout(dropout_rate))

        layers.append(tf.keras.layers.Dense(out_features))

    elif head_type == "conv":
        assert dropout_rate is None, "Dropout not supported for convolutional head"

        LOGGER.info("Using a convolutional + averaging regression head")

        layers.append(tf.keras.layers.LayerNormalization(axis=-1, **norm_kwargs))
        layers.append(healpy_layers.HealpyChebyshev(K=poly_degree, Fout=out_features, activation=None))
        layers.append(MeanLayer(axis=-2, dtype=tf.float32))

    else:
        raise ValueError(f"Unknown regression head type: {head_type}")

    return layers


def get_cls_embedding_layers(hidden_layers, dropout_rate=None, activation="relu"):
    """Build an MLP to embed binned Cls before fusion with map features in MapsPlusCLSNetwork.

    Args:
        hidden_layers: List of int widths, e.g. ``[512, 512, 512, 512]``.
            ``None`` or empty list → returns ``[]`` (no embedding).
        dropout_rate: Optional float; a single Dropout appended after all hidden layers.
        activation: Activation for hidden Dense layers.

    Returns:
        List of Keras layers: interleaved Dense + LayerNorm, with optional trailing Dropout.
    """
    if not hidden_layers:
        return []
    layers = []
    for h in hidden_layers:
        layers.append(tf.keras.layers.Dense(h, activation=activation))
        layers.append(tf.keras.layers.LayerNormalization(axis=-1))
    if dropout_rate is not None:
        layers.append(tf.keras.layers.Dropout(dropout_rate))
    return layers
