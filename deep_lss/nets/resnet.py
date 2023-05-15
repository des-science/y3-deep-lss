"""
Simple network for testing purposes, this was used in KiDS1000 and is from
https://cosmo-gitlab.phys.ethz.ch/jafluri/arne_handover/-/blob/main/networks/models/small_resnet_partial.py
"""

import tensorflow as tf
from deepsphere import healpy_layers


class ResNetLayers:
    def __init__(
        self,
        # shapes
        output_shape=6,
        # width
        n_base_channels=32,
        # depth
        n_downsampling=3,
        n_cheby=2,
        n_residuals=6,
        # misc
        poly_degree=5,
        norm_kwargs={},
        activation=tf.nn.relu,
    ) -> None:
        self.layers = []

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
        self.layers.append(tf.keras.layers.Dense(output_shape))

    def get_layers(self):
        return self.layers


# def get_network(n_output):
#     bn_kwargs = dict()
#     network = [
#         healpy_layers.HealpyPseudoConv(p=1, Fout=32, activation=tf.nn.relu),
#         healpy_layers.HealpyPseudoConv(p=1, Fout=64, activation=tf.nn.relu),
#         healpy_layers.HealpyPseudoConv(p=1, Fout=128, activation=tf.nn.relu),
#         healpy_layers.HealpyChebyshev(K=5, Fout=256, activation=tf.nn.relu),
#         tf.keras.layers.LayerNormalization(axis=-1),
#         healpy_layers.HealpyPseudoConv(p=1, Fout=256, activation=tf.nn.relu),
#         healpy_layers.HealpyChebyshev(K=5, Fout=256, activation=tf.nn.relu),
#         tf.keras.layers.LayerNormalization(axis=-1),
#         healpy_layers.HealpyPseudoConv(p=1, Fout=256, activation=tf.nn.relu),
#         healpy_layers.Healpy_ResidualLayer(
#             "CHEBY",
#             layer_kwargs={"K": 5, "activation": tf.nn.relu, "use_bias": True},
#             use_bn=True,
#             bn_kwargs=bn_kwargs,
#             norm_type="layer_norm",
#         ),
#         healpy_layers.Healpy_ResidualLayer(
#             "CHEBY",
#             layer_kwargs={"K": 5, "activation": tf.nn.relu, "use_bias": True},
#             use_bn=True,
#             bn_kwargs=bn_kwargs,
#             norm_type="layer_norm",
#         ),
#         healpy_layers.Healpy_ResidualLayer(
#             "CHEBY",
#             layer_kwargs={"K": 5, "activation": tf.nn.relu, "use_bias": True},
#             use_bn=True,
#             bn_kwargs=bn_kwargs,
#             norm_type="layer_norm",
#         ),
#         healpy_layers.Healpy_ResidualLayer(
#             "CHEBY",
#             layer_kwargs={"K": 5, "activation": tf.nn.relu, "use_bias": True},
#             use_bn=True,
#             bn_kwargs=bn_kwargs,
#             norm_type="layer_norm",
#         ),
#         healpy_layers.Healpy_ResidualLayer(
#             "CHEBY",
#             layer_kwargs={"K": 5, "activation": tf.nn.relu, "use_bias": True},
#             use_bn=True,
#             bn_kwargs=bn_kwargs,
#             norm_type="layer_norm",
#         ),
#         healpy_layers.Healpy_ResidualLayer(
#             "CHEBY",
#             layer_kwargs={"K": 5, "activation": tf.nn.relu, "use_bias": True},
#             use_bn=True,
#             bn_kwargs=bn_kwargs,
#             norm_type="layer_norm",
#         ),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.LayerNormalization(axis=-1),
#         tf.keras.layers.Dense(n_output),
#     ]

#     return network
