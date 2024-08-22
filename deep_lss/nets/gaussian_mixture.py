# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created August 2024
Author: Arne Thomsen
"""

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class GaussianMixtureModel:
    def __init__(
        self,
        dim_theta,
        dim_summary,
        num_components,
        num_hidden_layers=2,
        num_hidden_units=128,
        activation="relu",
        full_covariance=True,
    ):
        self.dim_theta = dim_theta
        self.dim_summary = dim_summary
        self.num_components = num_components
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_units = num_hidden_units
        self.activation = activation
        self.full_covariance = full_covariance

        self.mixture_logits_net = self._build_network(output_size=num_components)
        self.loc_net = self._build_network(output_size=num_components * dim_theta)

        if self.full_covariance:
            self.tril_size = (dim_theta * (dim_theta + 1)) // 2
            self.scale_net = self._build_network(output_size=num_components * self.tril_size)
        else:
            self.scale_net = self._build_network(output_size=num_components * dim_theta)
            self.scale_net.add(tf.keras.layers.Activation("softplus"))

    def _build_network(self, output_size):
        layers = [tf.keras.layers.InputLayer(input_shape=(self.dim_summary,))]

        for _ in range(self.num_hidden_layers):
            layers.append(tf.keras.layers.Dense(self.num_hidden_units, activation=self.activation))

        layers.append(tf.keras.layers.Dense(output_size))

        return tf.keras.Sequential(layers)

    def log_prob(self, theta, summary):
        """p(theta | summary)"""

        mixture_logits = self.mixture_logits_net(summary)  # (batch_size, num_components)

        loc = self.loc_net(summary)  # (batch_size, num_components * dim_theta)
        loc = tf.reshape(loc, [-1, self.num_components, self.dim_theta])

        if self.full_covariance:
            scale_tril = self.scale_net(summary)  # (batch_size, num_components * tril_size)
            scale_tril = tf.reshape(scale_tril, [-1, self.num_components, self.tril_size])
            scale_tril = tfp.math.fill_triangular(scale_tril)
            scale_tril = tf.reshape(scale_tril, [-1, self.num_components, self.dim_theta, self.dim_theta])
            component_distribution = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)
        else:
            scale = self.scale_net(summary)  # (batch_size, num_components * dim_theta)
            scale = tf.reshape(scale, [-1, self.num_components, self.dim_theta])
            component_distribution = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

        mixture_distribution = tfd.Categorical(logits=mixture_logits)

        gmm = tfd.MixtureSameFamily(
            mixture_distribution=mixture_distribution, components_distribution=component_distribution
        )

        return gmm.log_prob(theta)
