import tensorflow as tf

from deepsphere import healpy_layers

from kids1000_analysis import special_layers

"""
This file contains the specifications for the training, e.g. the network layers
"""

# Define if this network is intended for baryons or not
#######################################################

with_bary = False

# get the number of params according to bary
if with_bary:
    n_params = 9
else:
    n_params = 7

# Define the layers
###################

bn_kwargs = dict()
layers = [healpy_layers.HealpyPseudoConv(p=1, Fout=32, activation=tf.nn.relu),
          healpy_layers.HealpyPseudoConv(p=1, Fout=64, activation=tf.nn.relu),
          healpy_layers.HealpyPseudoConv(p=1, Fout=128, activation=tf.nn.relu),
          healpy_layers.HealpyChebyshev(K=5, Fout=256, activation=tf.nn.relu),
          tf.keras.layers.LayerNormalization(axis=-1),
          healpy_layers.HealpyPseudoConv(p=1, Fout=256, activation=tf.nn.relu),
          healpy_layers.HealpyChebyshev(K=5, Fout=256, activation=tf.nn.relu),
          tf.keras.layers.LayerNormalization(axis=-1),
          healpy_layers.HealpyPseudoConv(p=1, Fout=256, activation=tf.nn.relu),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": tf.nn.relu,
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": tf.nn.relu,
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": tf.nn.relu,
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": tf.nn.relu,
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": tf.nn.relu,
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": tf.nn.relu,
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          # mix the disconnected parts
          special_layers.MixingDense(activation=tf.nn.relu),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": tf.nn.relu,
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": tf.nn.relu,
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": tf.nn.relu,
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": tf.nn.relu,
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": tf.nn.relu,
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          healpy_layers.Healpy_ResidualLayer("CHEBY", layer_kwargs={"K": 5, "activation": tf.nn.relu,
                                                                    "use_bias": True},
                                             use_bn=True, bn_kwargs=bn_kwargs, norm_type="layer_norm"),
          tf.keras.layers.Flatten(),
          tf.keras.layers.LayerNormalization(axis=-1),
          tf.keras.layers.Dense(n_params)]
