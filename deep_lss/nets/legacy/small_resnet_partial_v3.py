import tensorflow as tf

from deepsphere import healpy_layers

"""
This file contains the specifications for the training, e.g. the network layers
"""

# quick and dirty layer

class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')

        self.conv2a = tf.keras.layers.Conv1D(filters, kernel_size, strides=1, padding='same', activation="relu")
        self.bn2a = tf.keras.layers.LayerNormalization(axis=-1)

        self.conv2b = tf.keras.layers.Conv1D(filters, kernel_size, strides=1, padding='same', activation="relu")
        self.bn2b = tf.keras.layers.LayerNormalization(axis=-1)

    
    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)
 
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x += input_tensor
        return tf.nn.relu(x)


# Define if this network is intended for baryons or not
#######################################################

with_bary = False

# get the number of params according to bary
if with_bary:
    n_params = 6
else:
    n_params = 4

# tags for training
param_ind = [0, 1, 5, 6] 

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
          ResnetIdentityBlock(16, 256),
          ResnetIdentityBlock(16, 256),
          ResnetIdentityBlock(16, 256),
          ResnetIdentityBlock(16, 256),
          ResnetIdentityBlock(16, 256),
          ResnetIdentityBlock(16, 256),
          tf.keras.layers.Flatten(),
          tf.keras.layers.LayerNormalization(axis=-1),
          tf.keras.layers.Dense(n_params)]
