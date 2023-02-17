"""
Simple network for testing purposes, this was used in KiDS1000 and is from
https://cosmo-gitlab.phys.ethz.ch/jafluri/arne_handover/-/blob/main/networks/models/small_resnet_partial.py
"""

import tensorflow as tf
from deepsphere import healpy_layers


def get_network(n_output):
    bn_kwargs = dict()
    network = [
        healpy_layers.HealpyPseudoConv(p=1, Fout=32, activation=tf.nn.relu),
        healpy_layers.HealpyPseudoConv(p=1, Fout=64, activation=tf.nn.relu),
        healpy_layers.HealpyPseudoConv(p=1, Fout=128, activation=tf.nn.relu),
        healpy_layers.HealpyChebyshev(K=5, Fout=256, activation=tf.nn.relu),
        tf.keras.layers.LayerNormalization(axis=-1),
        healpy_layers.HealpyPseudoConv(p=1, Fout=256, activation=tf.nn.relu),
        healpy_layers.HealpyChebyshev(K=5, Fout=256, activation=tf.nn.relu),
        tf.keras.layers.LayerNormalization(axis=-1),
        healpy_layers.HealpyPseudoConv(p=1, Fout=256, activation=tf.nn.relu),
        healpy_layers.Healpy_ResidualLayer(
            "CHEBY",
            layer_kwargs={"K": 5, "activation": tf.nn.relu, "use_bias": True},
            use_bn=True,
            bn_kwargs=bn_kwargs,
            norm_type="layer_norm",
        ),
        # healpy_layers.Healpy_ResidualLayer(
        #     "CHEBY",
        #     layer_kwargs={"K": 5, "activation": tf.nn.relu, "use_bias": True},
        #     use_bn=True,
        #     bn_kwargs=bn_kwargs,
        #     norm_type="layer_norm",
        # ),
        # healpy_layers.Healpy_ResidualLayer(
        #     "CHEBY",
        #     layer_kwargs={"K": 5, "activation": tf.nn.relu, "use_bias": True},
        #     use_bn=True,
        #     bn_kwargs=bn_kwargs,
        #     norm_type="layer_norm",
        # ),
        # healpy_layers.Healpy_ResidualLayer(
        #     "CHEBY",
        #     layer_kwargs={"K": 5, "activation": tf.nn.relu, "use_bias": True},
        #     use_bn=True,
        #     bn_kwargs=bn_kwargs,
        #     norm_type="layer_norm",
        # ),
        # healpy_layers.Healpy_ResidualLayer(
        #     "CHEBY",
        #     layer_kwargs={"K": 5, "activation": tf.nn.relu, "use_bias": True},
        #     use_bn=True,
        #     bn_kwargs=bn_kwargs,
        #     norm_type="layer_norm",
        # ),
        # healpy_layers.Healpy_ResidualLayer(
        #     "CHEBY",
        #     layer_kwargs={"K": 5, "activation": tf.nn.relu, "use_bias": True},
        #     use_bn=True,
        #     bn_kwargs=bn_kwargs,
        #     norm_type="layer_norm",
        # ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.LayerNormalization(axis=-1),
        tf.keras.layers.Dense(n_output),
    ]

    return network
