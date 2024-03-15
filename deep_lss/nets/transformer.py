# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2024
Author: Arne Thomsen
"""

import tensorflow as tf
from deepsphere import healpy_layers

from deep_lss.nets.regression_head import get_regression_head


class ViTLayers:
    """Class used to build the layers of a simple Vision Transformer network. Note that the spherical structure of the
    data only enters through the HEALPix nested ordering and the layers are not equivarient to rotations. There is no
    use of the DeepSphere graph within this."""

    def __init__(
        self,
        out_features=6,
        # transformer
        hidden_dim=128,
        healpix_patch_fac=4,
        attention_heads=4,
        transformer_layers=4,
        pos_encoding=True,
        layer_norm=True,
        # misc
        second_to_last_features=None,
        activation=tf.nn.relu,
        dropout_rate=0.0,
        smoothing_kwargs=None,
    ) -> None:
        """
        Initialize the ViT layers, very similar to https://arxiv.org/abs/2010.11929.

        Args:
            out_features (int, optional): Output shape of the regression head. This determines the size of the learned
                summary statistics. Defaults to 6.
            hidden_dim (int, optional): The dimension of the key, value, and query space within the standard multi-head
                transformer layers. Defaults to 128.
            healpix_patch_fac (int, optional): The downsampling factor for the Healpix data. The n_side is halved for
                every downsampling, controlling the size of the patches. Larger values mean more and smaller patches,
                which is generally less computationally expensive since the cost of the attention layers is quadratic
                in the patch size. Defaults to 4, then n_side=512 -> n_side=32, such that each ViT patch is made up of
                all of the n_side=512 subpixels within a n_side=32 superpixel.
            attention_heads (int, optional): The number of attention heads. Defaults to 4.
            transformer_layers (int, optional): The number of standard multi-head transformer blocks. Defaults to 4.
            pos_encoding (bool, optional): Whether to add positional encoding. Defaults to True.
            layer_norm (bool, optional): Whether to use layer normalization. Defaults to True.
            activation (callable, optional): Non-linear activation function to be used throughout. Defaults to
                tf.nn.relu.
            dropout_rate (float, optional): Dropout rate within the regression head. Defaults to 0.0.
            smoothing_kwargs (dict, optional): Keyword arguments to be passed to the smoothing layer. Defaults to None,
                then no smoothing is performed within the network.
        """

        self.layers = []

        if smoothing_kwargs is not None:
            self.layers.append(healpy_layers.HealpySmoothing(**smoothing_kwargs))

        # an 1d convolutional embedding followed by n_layers of off-the-shelf multihead attention
        self.layers.append(
            healpy_layers.Healpy_ViT(
                p=healpix_patch_fac,
                key_dim=hidden_dim,
                num_heads=attention_heads,
                positional_encoding=pos_encoding,
                n_layers=transformer_layers,
                layer_norm=layer_norm,
                activation=activation,
            )
        )

        # regression head
        self.layers.append(tf.keras.layers.Flatten())
        self.layers.append(tf.keras.layers.LayerNormalization(axis=-1))
        # TODO this creates too many trainable parameters
        if second_to_last_features is not None:
            raise NotImplementedError
        # self.layers.append(tf.keras.layers.Dense(n_second_to_last_features, activation=activation))
        self.layers.append(tf.keras.layers.Dropout(dropout_rate))
        self.layers.append(tf.keras.layers.Dense(out_features))

    def get_layers(self):
        return self.layers


class GTLayers:
    """Class used to build the layers of a graph Transformer network. The graph attention layers follow HEALPix
    downsampling through pseudo convolutions, similar to the ResNet network."""

    def __init__(
        self,
        out_features,
        # downsampling
        base_channels=32,
        downsampling_layers=4,
        # transformer
        hidden_dim=128,
        attention_heads=4,
        transformer_layers=4,
        pos_encoding=True,
        layer_norm=True,
        # regression head
        second_to_last_features=None,
        dropout_rate=None,
        # misc
        activation=tf.nn.relu,
        smoothing_kwargs=None,
    ) -> None:
        """
        Initialize the graph Transformer layers.

        Args:
            out_features (int, optional): Output shape of the regression head. This determines the size of the learned
                summary statistics. Defaults to 6.
            base_channels (int, optional): Number of channels after the first layer of the network. This number gets
                multiplied by a factor of two for every downsampling layer. Defaults to 32.
            downsampling_layers (int, optional): Number of pseudoconvolutions to perform a downsampling of the
                neighboring Healpix pixels. Note that these layers are fairly cheap and their number effectively
                determines how expensive the following graph attention layers are. Defaults to 3.
            hidden_dim (int, optional): The dimension of the key, value, and query space within the graph
                transformer layers. Defaults to 128.
            attention_heads (int, optional): The number of attention heads in the transformer. Defaults to 4.
            transformer_layers (int, optional): The number of transformer layers. Defaults to 4.
            pos_encoding (bool, optional): Whether to add positional encoding. Defaults to True.
            layer_norm (bool, optional): Whether to use layer normalization. Defaults to True.
            second_to_last_features (int, optional): The number of features in the second-to-last layer. Defaults to
                None.
            dropout_rate (float, optional): Dropout rate within the regression head. Defaults to None, then it's not
                included.
            activation (callable, optional): Non-linear activation function to be used throughout. Defaults to
                tf.nn.relu.
            smoothing_kwargs (dict, optional): Keyword arguments to be passed to the smoothing layer. Defaults to None,
                then no smoothing is performed within the network.
        """

        self.layers = []

        if smoothing_kwargs is not None:
            self.layers.append(healpy_layers.HealpySmoothing(**smoothing_kwargs))

        # downsampling
        n_channels = base_channels
        for _ in range(downsampling_layers):
            self.layers.append(healpy_layers.HealpyPseudoConv(p=1, Fout=n_channels, activation=activation))
            n_channels *= 2

        self.layers.append(
            healpy_layers.Healpy_Transformer(
                key_dim=hidden_dim,
                num_heads=attention_heads,
                positional_encoding=pos_encoding,
                n_layers=transformer_layers,
                layer_norm=layer_norm,
            )
        )

        # regression head
        assert second_to_last_features is None, "Not implemented as this creates too many trainable parameters"
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
