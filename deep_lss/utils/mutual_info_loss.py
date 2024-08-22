# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created August 2024
Author: Arne Thomsen

This file contains different lower bounds of the mutual information between summary vectors and parameters to be 
inferred, following
Chen et al. 2021 https://arxiv.org/pdf/2010.10079
Jeffrey et al. 2021 https://arxiv.org/pdf/2009.08459
Lanzieri et al. 2024 https://arxiv.org/pdf/2407.10877
"""

import tensorflow as tf

from deep_lss.nets.mlp import MultiLayerPerceptron
from deep_lss.nets.gaussian_mixture import GaussianMixtureModel

# Jensen-Shannon-divergence estimator #################################################################################


def jensen_shannon_divergence(critic, x, theta, m_inner_loop=16, training=True):
    """Following eq. (4) or Appendix B.1 in https://arxiv.org/pdf/2010.10079. In that work, this estimator provides the
    tightest lower bound. Its main disadvantage is the inner loop over m elements, which requires m additional
    evaluations of the network.

    Args:
        critic (tf.keras.Model): The model, which is denoted T in the paper. It takes as inputs [x, theta], where
            x are the uncompressed inputs. Furthermore, the model has to return a scalar value.
        x (tf.Tensor): Batch of input data.
        theta (tf.Tensor): Batch of parameters to be inferred.
        m (int, optional): Length of the inner loop necessary to compute the second term of the loss. Defaults to 16.
            In the paper, this is set to twice the batch size.
        training (bool, optional): Whether the model is in training mode. Defaults to True.
    """
    batch_size = tf.shape(x)[0]

    # sp(-T(θi, S(xi)))
    term_1 = tf.reduce_mean(tf.math.softplus(-critic([x, theta], training=training)))

    term_2 = 0
    for _ in range(m_inner_loop):
        permuted_indices = tf.random.shuffle(tf.range(batch_size))
        permuted_theta = tf.gather(theta, permuted_indices)

        # sp(T(θji, S(xi))) for each pair (θji, S(xi))
        term_2 += tf.reduce_mean(tf.math.softplus(critic([x, permuted_theta], training=training)))

    term_2 /= m_inner_loop

    loss = term_1 + term_2

    return loss


def get_jensen_shannon_critic_from_net(
    summary_net, dim_x, dim_theta, dropout_rate=0.0, num_hidden_units=128, num_layers=2
):
    theta_net = MultiLayerPerceptron(
        output_size=dim_theta, num_hidden_units=num_hidden_units, num_layers=num_layers, dropout_rate=dropout_rate
    )
    critic_net = MultiLayerPerceptron(
        output_size=1, num_hidden_units=num_hidden_units, num_layers=num_layers, dropout_rate=dropout_rate
    )

    in_x = tf.keras.Input(shape=(dim_x,))
    in_theta = tf.keras.Input(shape=(dim_theta,))

    out_summary = summary_net(in_x)
    out_theta = theta_net(in_theta)
    out_critic = critic_net(tf.concat([out_summary, out_theta], axis=-1))

    return tf.keras.Model(inputs=[in_x, in_theta], outputs=out_critic)


def get_jensen_shannon_critic_from_pred(dim_summary, dim_theta, dropout_rate=0.0, num_hidden_units=128, num_layers=2):
    raise NotImplementedError("This function can't be implemented. The critic network must be passed as an argument.")


# distance-correlation estimator ######################################################################################


def safe_norm(x, axis=None, eps=1e-12):
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis) + eps)


def h_tilde(a, b, eps=1e-12):
    # matrix of pairwise differences. The gradient of tf.norm tends to be numerically unstable, so we use safe_norm
    diff_ij = safe_norm(a[:, tf.newaxis] - b[tf.newaxis, :], axis=-1, eps=eps)

    term1 = diff_ij
    term2 = tf.reduce_mean(diff_ij, axis=1, keepdims=True)
    term3 = tf.reduce_mean(diff_ij, axis=0, keepdims=True)
    term4 = tf.reduce_mean(diff_ij)

    return term1 - term2 - term3 + term4


def distance_correlation(summary, theta, training=True, eps=1e-12):

    # compute h_tilde for (theta_i, theta_j) and (S(x_i), S(x_j))
    h_theta = h_tilde(theta, theta, eps=eps)
    h_summary = h_tilde(summary, summary, eps=eps)

    # numerator: sum over i,j h_tilde(theta_i, theta_j) * h_tilde(S(x_i), S(x_j))
    numerator = tf.reduce_mean(h_theta * h_summary)

    # denominator: product of sqrt sums
    sum_h_theta_squared = tf.reduce_mean(tf.square(h_theta))
    sum_h_summary_squared = tf.reduce_mean(tf.square(h_summary))
    denominator = tf.sqrt(sum_h_theta_squared + eps) * tf.sqrt(sum_h_summary_squared + eps)

    loss = -tf.math.divide_no_nan(numerator, denominator + eps)

    return loss


# variational estimator ###############################################################################################


def get_variational_model_from_net(
    summary_net,
    dim_x,
    dim_summary,
    dim_theta,
    num_components=4,
    num_hidden_layers=2,
    num_hidden_units=128,
    activation="relu",
):
    gmm = GaussianMixtureModel(
        dim_theta=dim_theta,
        dim_summary=dim_summary,
        num_components=num_components,
        num_hidden_layers=num_hidden_layers,
        num_hidden_units=num_hidden_units,
        activation=activation,
    )

    in_x = tf.keras.Input(shape=(dim_x,))
    in_theta = tf.keras.Input(shape=(dim_theta,))

    out_summary = summary_net(in_x)
    out = gmm.log_prob(in_theta, out_summary)

    return tf.keras.Model(inputs=[in_x, in_theta], outputs=out)


def get_variational_model_from_summary(
    dim_summary,
    dim_theta,
    num_components=4,
    num_hidden_layers=2,
    num_hidden_units=128,
    activation="relu",
    full_covariance=True,
):
    gmm = GaussianMixtureModel(
        dim_theta=dim_theta,
        dim_summary=dim_summary,
        num_components=num_components,
        num_hidden_layers=num_hidden_layers,
        num_hidden_units=num_hidden_units,
        activation=activation,
        full_covariance=full_covariance,
    )

    in_summary = tf.keras.Input(shape=(dim_summary,))
    in_theta = tf.keras.Input(shape=(dim_theta,))

    out = -gmm.log_prob(in_theta, in_summary)

    return tf.keras.Model(inputs=[in_summary, in_theta], outputs=out)
