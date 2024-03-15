# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2024
Author: Arne Thomsen
"""

import tensorflow as tf
from deepsphere import healpy_layers

from deep_lss.nets.regression_head import get_regression_head


class OneDResidualBlock(tf.keras.Model):
    """Simple residual block for 1D convolutions. The block consists of two 1D convolutions, each followed by a layer
    normalization and a ReLU activation."""

    def __init__(self, filters, kernel_size, norm_kwargs={}, norm_type="layer_norm", activation=tf.nn.relu, name=""):
        super(OneDResidualBlock, self).__init__(name=name)

        self.activation = activation

        self.conv1 = tf.keras.layers.Conv1D(filters, kernel_size, strides=1, padding="same", activation=activation)
        self.conv2 = tf.keras.layers.Conv1D(filters, kernel_size, strides=1, padding="same", activation=activation)

        if norm_type == "layer_norm":
            self.norm1 = tf.keras.layers.LayerNormalization(axis=-1, **norm_kwargs)
            self.norm2 = tf.keras.layers.LayerNormalization(axis=-1, **norm_kwargs)
        elif norm_type == "batch_norm":
            self.norm1 = tf.keras.layers.BatchNormalization(**norm_kwargs)
            self.norm2 = tf.keras.layers.BatchNormalization(**norm_kwargs)
        else:
            raise NotImplementedError

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.norm1(x, training=training)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = self.activation(x)

        x += input_tensor
        x = self.activation(x)

        return x


class OneDConvLayers:
    """Class used to build the layers of a simple resnet only making use of 1D convolutions, no graph convolutions.
    Note that the spherical structure of the data only enters through the HEALPix nested ordering and the layers are
    not equivarient to rotations."""

    def __init__(
        self,
        out_features,
        # width
        base_channels=8,
        # depth
        downsampling_layers=5,
        residual_layers=5,
        # convolutions
        kernel_size=9,
        # regression head
        second_to_last_features=128,
        dropout_rate=None,
        # misc
        activation=tf.nn.relu,
        smoothing_kwargs=None,
    ) -> None:
        """
        Initializes the OneDConvNet model, which only consists of 1D convolutions

        Args:
            out_features (int, optional): Number of output dimensions. Defaults to 6.
            base_channels (int, optional): Number of channels at the input to the network. In the downsampling
                layers, this gets doubled. Defaults to 8.
            second_to_last_features (int, optional): Number of channels in the second-to-last layer. Defaults to 128.
            downsampling_layers (int, optional): Number of downsampling layers. In every one of these, the HEALPix
                n_side is halved. Defaults to 5 (n_side=512 -> n_side=16)
            residual_layers (int, optional): Number of residual blocks. Defaults to 5.
            kernel_size (int, optional): Size of the one dimensional convolutional kernel. Defaults to 9.
            activation (callable, optional): Non-linear activation function to be used throughout. Defaults to
                tf.nn.relu.
            dropout_rate (float, optional): Dropout rate within the regression head. Defaults to 0.0.
            smoothing_kwargs (dict, optional): Keyword arguments to be passed to the smoothing layer. Defaults to None,
                then no smoothing is performed within the network.
        """

        self.layers = []

        if smoothing_kwargs is not None:
            self.layers.append(healpy_layers.HealpySmoothing(**smoothing_kwargs))

        # downsamplling and increasing channels
        n_channels = base_channels
        for i in range(downsampling_layers):
            self.layers.append(
                tf.keras.layers.Conv1D(
                    n_channels, kernel_size, padding="same", activation=activation, name=f"1d_conv_{i}"
                )
            )
            self.layers.append(tf.keras.layers.LayerNormalization(axis=-1, name=f"layer_norm_{i}"))

            n_channels *= 2
            self.layers.append(
                healpy_layers.HealpyPseudoConv(p=1, Fout=n_channels, activation=activation, name=f"pseudo_conv_{i}")
            )

        # residual blocks of 1d convolutions
        for i in range(residual_layers):
            self.layers.append(
                OneDResidualBlock(filters=n_channels, kernel_size=kernel_size, name=f"residual_block_{i}")
            )

        # regression head
        regression_head_layers = get_regression_head(
            out_features=out_features,
            head_type="dense",
            second_to_last_features=second_to_last_features,
            activation=activation,
            dropout_rate=dropout_rate,
        )
        self.layers.extend(regression_head_layers)

    def get_layers(self):
        return self.layers
