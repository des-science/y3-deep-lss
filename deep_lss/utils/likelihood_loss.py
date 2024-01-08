# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created January 2024
Author: Arne Thomsen

Fully supervised loss functions that take a label. The likelihood loss is based off 
https://github.com/tomaszkacprzak/deep_lss/blob/main/deep_lss/networks/losses.py
by Tomasz Kacprzak (itself based off Janis Fluri's implementation).
"""

import tensorflow as tf
import tensorflow_probability as tfp

from deep_lss.utils import summary


@tf.function
def neg_likelihood_loss(predictions, theta_true, n_theta, eps=1e-30, summary_writer=None, training=False):
    """Calculate the negative likelihood loss like in equation (17) in https://arxiv.org/pdf/1906.03156.pdf.

    Args:
        predictions (tf.Tensor): Predictions of the network. The first n_theta values predict the parameter mean and
            the rest is used to build the covariance matrix via the Cholesky decomposition.
        theta_true (tf.Tensor): True parameter values.
        n_theta (int): Number of parameters. This is used to infer the number of predicted matrix elements in the
            Cholesky decomposition.
        eps (float, optional): Small value to ensure that the determinant is not zero, which would be a problem for the
            logarithm. Defaults to 1e-30.
        training (bool, optional): Whether the loss is used for training. If False, no summaries will be written even
            if a summary_writer is supplied. Defaults to True.
        summary_writer (tf.summary.SummaryWriter, optional): The writer used to write tensorboard summaries. Defaults
            to None.

    Returns:
        tf.Tensor: Mean loss value over the batch.
    """

    # number of entries in a triangular matrix (including the diagonal), as used to construct the covariance matrix
    # via the Cholesky decomposition
    n_triang_with_diag = n_theta * (n_theta + 1) // 2

    theta_pred, cov_pred = tf.split(predictions, [n_theta, n_triang_with_diag], axis=1, name="likeloss_split_mean_cov")

    # subtract predictions and labels
    residual = tf.subtract(theta_pred, theta_true, name="likeloss_diff_true_pred")

    # make upper triangular matrix L^T
    upper_triangular = tfp.math.fill_triangular(cov_pred, upper=True, name="likeloss_fill_triangular")

    # Get diagonal
    diag = tf.linalg.diag_part(upper_triangular, name="likeloss_diag_part")

    # add a small number such that the diag is never zero to log it
    diag += eps

    # get log determinant
    # https://math.stackexchange.com/questions/3158303/using-cholesky-decomposition-to-compute-covariance-matrix-determinant
    log_det = tf.reduce_sum(tf.math.log(tf.square(diag)), axis=1)

    # mean over the batch dimension
    mean_log_det = -tf.reduce_mean(log_det, name="likeloss_mean_det")

    # get norm(L^T * residual) (second part of the likelihood loss)
    # https://stats.stackexchange.com/questions/503058/relationship-between-cholesky-decomposition-and-matrix-inversion
    Lt_residual = tf.einsum("ijk,ik->ij", upper_triangular, residual, name="likeloss_Lt_res")
    Lt_residual_norm = tf.reduce_sum(tf.square(Lt_residual), axis=1, name="likeloss_norm_Lt_res")
    mean_Lt_residual_norm = tf.reduce_mean(Lt_residual_norm)

    summary.write_summary("likelihood_log_det_loss", mean_log_det, summary_writer, training)
    summary.write_summary("likelihood_residual_loss", mean_Lt_residual_norm, summary_writer, training)

    neg_likelihood_loss = tf.add(mean_Lt_residual_norm, mean_log_det)

    return neg_likelihood_loss
