"""
Author: Arne Thomsen

Conditional normalizing flow (RealNVP-style affine coupling layers) for density estimation
of p(theta | summary). Used as a drop-in alternative to GaussianMixtureModel in the
variational mutual information maximization loss.
"""

import numpy as np
import tensorflow as tf


class NormalizingFlowModel:
    """Conditional RealNVP normalizing flow estimating p(theta | summary).

    Uses alternating affine coupling layers where each layer's shift and scale are
    produced by a small MLP conditioned on the summary statistic.

    Numerical stability guarantees:
      - Raw log-scale outputs are clipped to [-log_scale_clip, +log_scale_clip]
      - scale_eps is added after exp() so scale > scale_eps > 0 always
      - log(scale) is therefore always finite
    """

    def __init__(
        self,
        dim_theta,
        dim_summary,
        num_layers=4,
        num_hidden_units=64,
        num_hidden_layers=2,
        activation="relu",
        scale_eps=1e-5,
        log_scale_clip=5.0,
        permute=False,
        theta_shift=None,
        theta_scale=None,
    ):
        self.dim_theta = dim_theta
        self.dim_summary = dim_summary
        self.num_layers = num_layers
        self.scale_eps = scale_eps
        self.log_scale_clip = log_scale_clip

        # Fixed roll-by-1 permutation applied between couplings (volume-preserving, zero parameters).
        # Without it the half-split is static, so dims within the same half (for the standard 6-param
        # target: Om, s8, w0 all sit in the lower half) never condition on each other directly and
        # their correlations must be built up indirectly across layers.
        self.permute = permute
        if permute:
            perm = np.roll(np.arange(dim_theta), 1)
            self._perm = tf.constant(perm, dtype=tf.int32)
            self._inv_perm = tf.constant(np.argsort(perm), dtype=tf.int32)

        # Optional affine standardization mirroring GaussianMixtureModel: the flow operates on
        # z_theta = (theta - theta_shift) / theta_scale. This matters more here than for the GMM:
        # raw theta is a direct INPUT to the coupling MLPs (in the GMM it only enters the NLL
        # quadratic form), and the couplings must also absorb each parameter's scale into their
        # clipped log-scale outputs. log_prob remains the density in physical theta units via the
        # constant log-Jacobian.
        if theta_shift is not None or theta_scale is not None:
            self.theta_shift = tf.constant(theta_shift, dtype=tf.float32)
            self.theta_scale = tf.constant(theta_scale, dtype=tf.float32)
            self.log_jacobian = -tf.reduce_sum(tf.math.log(self.theta_scale))
        else:
            self.theta_shift = None
            self.theta_scale = None
            self.log_jacobian = 0.0

        # d = size of the "lower" half; upper half has size dim_theta - d
        self._d = dim_theta // 2

        self.coupling_nets = []
        for i in range(num_layers):
            if i % 2 == 0:
                # even layer: condition on upper half, transform lower half
                in_size = (dim_theta - self._d) + dim_summary
                out_size = self._d
            else:
                # odd layer: condition on lower half, transform upper half
                in_size = self._d + dim_summary
                out_size = dim_theta - self._d
            self.coupling_nets.append(
                self._build_coupling_net(in_size, out_size, num_hidden_units, num_hidden_layers, activation)
            )

    def _build_coupling_net(self, in_size, out_size, num_hidden_units, num_hidden_layers, activation):
        # Force float32 on the coupling MLPs. When the surrounding model runs under a
        # mixed_bfloat16 policy (the maps training path), Dense layers would otherwise emit
        # bfloat16 shift/log_scale while log_prob/inverse cast theta and summary to float32,
        # producing a dtype-mismatch TypeError in the (z_transform - shift) subtract. The flow's
        # log/exp density math wants float32 regardless, so pin the whole head to it.
        layers = [tf.keras.layers.InputLayer(input_shape=(in_size,), dtype="float32")]
        for _ in range(num_hidden_layers):
            layers.append(tf.keras.layers.Dense(num_hidden_units, activation=activation, dtype="float32"))
        # outputs 2 * out_size: first half = shift, second half = raw log-scale
        layers.append(tf.keras.layers.Dense(2 * out_size, kernel_initializer="glorot_uniform", dtype="float32"))
        return tf.keras.Sequential(layers)

    def log_prob(self, theta, summary):
        """log p(theta | summary), shape (batch_size,)."""
        theta = tf.cast(theta, tf.float32)
        summary = tf.cast(summary, tf.float32)

        if self.theta_shift is not None:
            theta = (theta - self.theta_shift) / self.theta_scale

        z = theta
        log_det_J = tf.zeros(tf.shape(theta)[0], dtype=tf.float32)
        d = self._d

        for i, net in enumerate(self.coupling_nets):
            if self.permute and i > 0:
                z = tf.gather(z, self._perm, axis=-1)
            if i % 2 == 0:
                # pass upper half through; transform lower half conditioned on upper + summary
                z_pass = z[:, d:]
                z_transform = z[:, :d]
                context = tf.concat([z_pass, summary], axis=-1)
                shift, log_scale = self._shift_log_scale(net, context, d)
                scale = tf.exp(log_scale) + self.scale_eps
                z_transform_new = (z_transform - shift) / scale
                log_det_J += -tf.reduce_sum(tf.math.log(scale), axis=-1)
                z = tf.concat([z_transform_new, z_pass], axis=-1)
            else:
                # pass lower half through; transform upper half conditioned on lower + summary
                d2 = self.dim_theta - d
                z_pass = z[:, :d]
                z_transform = z[:, d:]
                context = tf.concat([z_pass, summary], axis=-1)
                shift, log_scale = self._shift_log_scale(net, context, d2)
                scale = tf.exp(log_scale) + self.scale_eps
                z_transform_new = (z_transform - shift) / scale
                log_det_J += -tf.reduce_sum(tf.math.log(scale), axis=-1)
                z = tf.concat([z_pass, z_transform_new], axis=-1)

        # log p(z) under standard normal base distribution
        log_p_base = -0.5 * (
            tf.cast(self.dim_theta, tf.float32) * tf.math.log(2.0 * np.pi) + tf.reduce_sum(tf.square(z), axis=-1)
        )

        return log_p_base + log_det_J + self.log_jacobian

    def _shift_log_scale(self, net, context, out_size):
        out = net(context)  # (B, 2 * out_size)
        shift = out[:, :out_size]
        log_scale = tf.clip_by_value(out[:, out_size:], -self.log_scale_clip, self.log_scale_clip)
        return shift, log_scale

    def inverse(self, z, summary):
        """Invert the flow: z ~ N(0, I) -> theta ~ p(theta | summary), shape (B, dim_theta)."""
        z = tf.cast(z, tf.float32)
        summary = tf.cast(summary, tf.float32)

        theta = z
        d = self._d

        for i in reversed(range(self.num_layers)):
            net = self.coupling_nets[i]
            if i % 2 == 0:
                # even layer transformed the lower half conditioned on the (untouched) upper half
                theta_up = theta[:, d:]
                z_low = theta[:, :d]
                context = tf.concat([theta_up, summary], axis=-1)
                shift, log_scale = self._shift_log_scale(net, context, d)
                scale = tf.exp(log_scale) + self.scale_eps
                theta_low = z_low * scale + shift
                theta = tf.concat([theta_low, theta_up], axis=-1)
            else:
                # odd layer transformed the upper half conditioned on the (untouched) lower half
                theta_low = theta[:, :d]
                z_up = theta[:, d:]
                context = tf.concat([theta_low, summary], axis=-1)
                shift, log_scale = self._shift_log_scale(net, context, self.dim_theta - d)
                scale = tf.exp(log_scale) + self.scale_eps
                theta_up = z_up * scale + shift
                theta = tf.concat([theta_low, theta_up], axis=-1)

            # forward applies the permutation BEFORE coupling i (for i > 0), so invert it AFTER
            if self.permute and i > 0:
                theta = tf.gather(theta, self._inv_perm, axis=-1)

        if self.theta_shift is not None:
            theta = theta * self.theta_scale + self.theta_shift

        return theta

    def mean(self, summary, n_samples=256):
        """Monte Carlo estimate of the posterior mean E[theta | summary], shape (B, dim_theta).

        The flow has no closed-form mean, so it is estimated by sampling z ~ N(0, I), inverting
        the flow to get theta samples, and averaging.
        """
        summary = tf.cast(summary, tf.float32)
        batch_size = tf.shape(summary)[0]

        z = tf.random.normal((batch_size, n_samples, self.dim_theta), dtype=tf.float32)
        summary_tiled = tf.repeat(summary[:, tf.newaxis, :], n_samples, axis=1)

        z_flat = tf.reshape(z, [-1, self.dim_theta])
        summary_flat = tf.reshape(summary_tiled, [-1, self.dim_summary])

        theta_flat = self.inverse(z_flat, summary_flat)
        theta = tf.reshape(theta_flat, [batch_size, n_samples, self.dim_theta])

        return tf.reduce_mean(theta, axis=1)
