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
        out_dim=6,
        # width
        base_channels=32,
        second_to_last_features=128,
        # depth
        downsampling_layers=3,
        cheby_layers=2,
        residual_layers=6,
        # misc
        poly_degree=5,
        norm_kwargs={},
        activation=tf.nn.relu,
        dropout_rate=0.0,
        smoothing_kwargs=None,
    ) -> None:
        """Class used to build the layers of the ResNet network, which was used as the fiducial architecture in Janis'
        KiDS1000 analysis.

        Args:
            out_dim (int, optional): Output shape of the regression head. This determines the size of the learned
                summary statistics. Defaults to 6.
            base_channels (int, optional): Number of channels after the first layer of the network. This number gets
                multiplied by a factor of two for every downsampling layer. Defaults to 32.
            downsampling_layers (int, optional): Number of pseudoconvolutions to perform a downsampling of the
                neighboring Healpix pixels. Note that these layers are fairly cheap and their number effectively
                determines how expensive the following (residual) graph convolutions are. Defaults to 3.
            cheby_layers (int, optional): Number of Chebyshev convolutions to downsample with. These layers play the
                same role as the pure downsampling layers, just include an additional (Chebyshev) graph convoution.
                Defaults to 2.
            residual_layers (int, optional): Number of residual layers. These are the main graph convolutions. Defaults
                to 6.
            poly_degree (int, optional): Degree of the polynomials within the Chebyshev convolutions. Defaults to 5.
            norm_kwargs (dict, optional): Keyword arguments to be passed to the normalization layers. Defaults to {}.
            activation (callable, optional): Non-linear activation function to be used throughout. Defaults to
                tf.nn.relu.
            dropout_rate (float, optional): Dropout rate within the regression head. Defaults to 0.0.
            smoothing_kwargs (dict, optional): Keyword arguments to be passed to the smoothing layer. Defaults to None,
                then no smoothing is performed within the network.
        """
        self.layers = []

        if smoothing_kwargs is not None:
            self.layers.append(healpy_layers.HealpySmoothing(**smoothing_kwargs))

        # downsampling and increasing channels
        n_channels = base_channels
        for _ in range(downsampling_layers):
            self.layers.append(healpy_layers.HealpyPseudoConv(p=1, Fout=n_channels, activation=activation))
            n_channels *= 2

        # downsampling and Chebyshev convolutions
        for _ in range(cheby_layers):
            self.layers.append(healpy_layers.HealpyChebyshev(K=poly_degree, Fout=n_channels, activation=activation))
            self.layers.append(tf.keras.layers.LayerNormalization(axis=-1, **norm_kwargs))
            self.layers.append(healpy_layers.HealpyPseudoConv(p=1, Fout=n_channels, activation=activation))

        # residual Chebyshev convolutions
        for _ in range(residual_layers):
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
        if second_to_last_features is not None:
            self.layers.append(tf.keras.layers.Dense(second_to_last_features, activation=activation))
        self.layers.append(tf.keras.layers.Dropout(dropout_rate))
        self.layers.append(tf.keras.layers.Dense(out_dim))

    def get_layers(self):
        return self.layers
