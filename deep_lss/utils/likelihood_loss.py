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
from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


def neg_likelihood_loss(
    predictions,
    theta_true,
    n_theta,
    lambda_tikhonov=None,
    eps=1e-30,
    training=False,
    summary_writer=None,
    summary_suffix="",
    img_summary=False,
    xla=False,
):
    """Calculate the negative likelihood loss like in equation (17) in https://arxiv.org/pdf/1906.03156.pdf.

    Args:
        predictions (tf.Tensor): Predictions of the network. The first n_theta values predict the parameter mean and
            the rest is used to build the covariance matrix via the Cholesky decomposition.
        theta_true (tf.Tensor): True parameter values.
        n_theta (int): Number of parameters. This is used to infer the number of predicted matrix elements in the
            Cholesky decomposition.
        lambda_tikhonov (Union[float, tf.Variable], optional): Regularization parameter for the Tikhonov
            regularization. This penalizes predicting the zero matrix as the covariance. When this is a tf.Variable,
            the regularization strength can be varied with a scheduler. It is recommended that tikhonov regularization
            is only applied at the beginning of training, since the two terms of the likelihood loss is constructed to
            balance its two terms and the regularization can skew results. Note that this has been implemented because
            we have observed network training to get stuck close to the zero matrix. Defaults to None, then no
            regularization is applied.
        eps (float, optional): Small value to ensure that the determinant is not zero, which would be a problem for the
            logarithm. Defaults to 1e-30.
        training (bool, optional): Whether the loss is used for training. If False, no summaries will be written even
            if a summary_writer is supplied. Defaults to True.
        summary_writer (tf.summary.SummaryWriter, optional): The writer used to write tensorboard summaries. Defaults
            to None.
        img_summary (bool, optional): Whether to write image summaries of the covariance matrix. Defaults to False.
        xla (bool, optional): Whether to enable XLA just in time compilation. Note that this is incompatible with
            the DeepSphere graph convolutional layers, as they contain unsupported
            SparseDenseMatirxMultiplications. Defaults to False.

    Returns:
        tf.Tensor: Mean loss value over the batch.
    """

    LOGGER.warning(f"Tracing neg_likelihood_loss")

    # number of entries in a triangular matrix (including the diagonal), as used to construct the covariance matrix
    # via the Cholesky decomposition
    n_triang_with_diag = n_theta * (n_theta + 1) // 2

    theta_pred, cov_pred = tf.split(predictions, [n_theta, n_triang_with_diag], axis=1, name="likeloss_split_mean_cov")

    # subtract predictions and labels
    residual = tf.subtract(theta_pred, theta_true, name="likeloss_diff_true_pred")

    # make upper triangular matrix L^T
    upper_triangular = tfp.math.fill_triangular(cov_pred, upper=True, name="likeloss_fill_triangular")

    if img_summary:
        mean_upper_triangular = tf.reduce_mean(upper_triangular, axis=0, keepdims=True)
        upper_triangular_img = tf.expand_dims(mean_upper_triangular, axis=-1)

        mean_cov = tf.matmul(mean_upper_triangular, mean_upper_triangular, transpose_b=True)
        cov_img = tf.expand_dims(mean_cov, axis=-1)

        if not xla:
            summary.write_summary(
                "likelihood_tri_img" + summary_suffix,
                upper_triangular_img,
                summary_writer,
                training,
                summary_type="image",
            )
            summary.write_summary(
                "likelihood_cov_img" + summary_suffix, cov_img, summary_writer, training, summary_type="image"
            )

    # Get diagonal
    diag = tf.linalg.diag_part(upper_triangular, name="likeloss_diag_part")

    # add a small number such that the diag is never zero to log it
    diag += eps

    # get log determinant from the Cholesky decomposition (first part of the likelihood loss)
    # https://math.stackexchange.com/questions/3158303/using-cholesky-decomposition-to-compute-covariance-matrix-determinant
    log_det = tf.reduce_sum(tf.math.log(tf.square(diag)), axis=1)

    # mean over the batch dimension
    mean_log_det = -tf.reduce_mean(log_det, name="likeloss_mean_det")

    # get norm(L^T * residual) (second part of the likelihood loss)
    # https://stats.stackexchange.com/questions/503058/relationship-between-cholesky-decomposition-and-matrix-inversion
    Lt_residual = tf.einsum("ijk,ik->ij", upper_triangular, residual, name="likeloss_Lt_res")
    Lt_residual_norm = tf.reduce_sum(tf.square(Lt_residual), axis=1, name="likeloss_norm_Lt_res")
    mean_Lt_residual_norm = tf.reduce_mean(Lt_residual_norm)

    if not xla:
        summary.write_summary("likelihood_log_det_loss" + summary_suffix, mean_log_det, summary_writer, training)
        summary.write_summary(
            "likelihood_residual_loss" + summary_suffix, mean_Lt_residual_norm, summary_writer, training
        )

    neg_likelihood_loss = tf.add(mean_Lt_residual_norm, mean_log_det)

    if lambda_tikhonov is not None:
        # TODO derive a formula for the covariance matrix A (instead of its Cholesky decomposition L with A = L^T * L)
        frob_norm = tf.linalg.norm(upper_triangular, ord="fro", axis=(-2, -1))
        mean_frob_norm = tf.reduce_mean(frob_norm)
        tikhonov_loss = -lambda_tikhonov * mean_frob_norm
        if not xla:
            summary.write_summary("likelihood_tikhonov_loss" + summary_suffix, tikhonov_loss, summary_writer, training)

        neg_likelihood_loss = tf.add(neg_likelihood_loss, tikhonov_loss)

    return neg_likelihood_loss
