# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created November 2023
Author: Arne Thomsen
"""

import tensorflow as tf
from deepsphere import healpy_layers


class ResNetLayers:
    """Class used to build the layers of the ResNet network, which was used as the fiducial architecture in Janis'
    KiDS1000 analysis.
    """

    def __init__(
        self,
        # shapes
        output_shape=6,
        # width
        n_base_channels=32,
        n_second_to_last_channels=128,
        # depth
        n_downsampling=3,
        n_cheby=2,
        n_residuals=6,
        # misc
        poly_degree=5,
        norm_kwargs={},
        activation=tf.nn.relu,
        dropout_rate=0.0,
        smoothing_kwargs=None,
    ) -> None:
        """Note that the default parameters correspond to the fiducial architecture used in Janis' KiDS1000 analysis.

        Args:
            output_shape (int, optional): Output shape of the regression head. Defaults to 6.
            n_base_channels (int, optional): Number of channels after the first layer of the network. This number gets
                multiplied by a factor of two for every n_downsampling. Defaults to 32.
            n_downsampling (int, optional): Number of pseudoconvolutions to perform a downsampling of the neighboring
                Healpix pixels. Defaults to 3.
            n_cheby (int, optional): Number of Chebyshev convolutions to downsample with. Defaults to 2.
            n_residuals (int, optional): Number of residual layers. Defaults to 6.
            poly_degree (int, optional): Degree of the polynomials within the Chebyshev convolutions. Defaults to 5.
            norm_kwargs (dict, optional): Keyword arguments to be passed to the normalization layers. Defaults to {}.
            activation (callable, optional): Non-linear activation function to be used throughout. Defaults to
                tf.nn.relu.
            smoothing_kwargs (dict, optional): Keyword arguments to be passed to the smoothing layer. Defaults to None,
                then no smoothing is performed within the network.
        """
        self.layers = []

        if smoothing_kwargs is not None:
            self.layers.append(healpy_layers.HealpySmoothing(**smoothing_kwargs))

        # downsampling and increasing channels
        n_channels = n_base_channels
        for _ in range(n_downsampling):
            self.layers.append(healpy_layers.HealpyPseudoConv(p=1, Fout=n_channels, activation=activation))
            n_channels *= 2

        # downsampling and Chebyshev convolutions
        for _ in range(n_cheby):
            self.layers.append(healpy_layers.HealpyChebyshev(K=poly_degree, Fout=n_channels, activation=activation))
            self.layers.append(tf.keras.layers.LayerNormalization(axis=-1))
            self.layers.append(healpy_layers.HealpyPseudoConv(p=1, Fout=n_channels, activation=activation))

        # residual Chebyshev convolutions
        for _ in range(n_residuals):
            self.layers.append(
                healpy_layers.Healpy_ResidualLayer(
                    "CHEBY",
                    layer_kwargs={"K": poly_degree, "activation": activation, "use_bias": True},
                    # the delta loss is only compatible with layer and not batch normalization
                    use_bn=True,
                    bn_kwargs=norm_kwargs,
                    norm_type="layer_norm",
                )
            )

        # regression head
        self.layers.append(tf.keras.layers.Flatten())
        self.layers.append(tf.keras.layers.LayerNormalization(axis=-1))
        self.layers.append(tf.keras.layers.Dense(n_second_to_last_channels, activation=activation))
        self.layers.append(tf.keras.layers.Dropout(dropout_rate))
        self.layers.append(tf.keras.layers.Dense(output_shape))

    def get_layers(self):
        return self.layers
