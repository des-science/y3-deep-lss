# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created August 2024
Author: Arne Thomsen

Power-spectrum (Cls) preprocessing layers whose fitted statistics are stored as non-trainable
weights inside the TF checkpoint. Passed into the ``MultiLayerPerceptron`` Cls encoder
(``deep_lss.nets.encoders.cls.mlp``) as its ``whitening`` / ``input_transform``.
"""

import numpy as np
import tensorflow as tf
from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


class PCAWhiteningLayer(tf.keras.layers.Layer):
    """Offline PCA whitening stored as non-trainable weights inside the TF checkpoint.

    Call fit() once on the training data (numpy array) before the training loop.
    The fitted mean and projection matrix are saved with the model checkpoint and
    restored automatically at inference time — no separate file needed.

    Output dimension is n_components (< input dimension if truncated), with each
    component having zero mean and unit variance over the training distribution.
    LayerNorm is redundant after this layer and should be disabled in the MLP.
    """

    def __init__(self, n_components, whiten=True, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.whiten = whiten
        self.eps = eps

    def build(self, input_shape):
        n_in = input_shape[-1]
        n_out = min(self.n_components, n_in)
        self.mean_ = self.add_weight("mean", shape=(n_in,), trainable=False, initializer="zeros")
        self.components_ = self.add_weight("components", shape=(n_in, n_out), trainable=False, initializer="zeros")
        super().build(input_shape)

    def fit(self, x, max_samples=200_000):
        """Compute PCA whitening statistics from a (N, n_in) numpy array.

        Subsamples to max_samples rows so covariance estimation stays fast even
        when the full training set is large. 200k samples is more than sufficient
        to estimate an 800×800 covariance matrix accurately.
        """
        if not self.built:
            self.build((None, x.shape[-1]))

        rng = np.random.default_rng(0)
        if x.shape[0] > max_samples:
            idx = rng.choice(x.shape[0], size=max_samples, replace=False)
            x = x[idx]

        x = x.astype(np.float64)
        mean = x.mean(axis=0)
        cov = np.cov(x.T)

        eigvals, eigvecs = np.linalg.eigh(cov)
        # eigh returns ascending order — reverse to descending
        idx = np.argsort(eigvals)[::-1][: self.n_components]
        if self.whiten:
            components = eigvecs[:, idx] / np.sqrt(eigvals[idx] + self.eps)
        else:
            components = eigvecs[:, idx]

        explained = eigvals[np.argsort(eigvals)[::-1]][: self.n_components].sum() / eigvals.sum()
        LOGGER.info(
            f"PCAWhiteningLayer: kept {self.n_components}/{x.shape[1]} components, "
            f"explained variance = {explained:.3f}"
        )

        self.mean_.assign(mean.astype(np.float32))
        self.components_.assign(components.astype(np.float32))

    def call(self, inputs):
        return (inputs - self.mean_) @ self.components_

    def get_config(self):
        config = super().get_config()
        config.update({"n_components": self.n_components, "whiten": self.whiten, "eps": self.eps})
        return config


class AsinhScaleLayer(tf.keras.layers.Layer):
    """Per-feature ``asinh(x / s)`` transform with the scale stored in the TF checkpoint.

    The scale ``s`` is a data-derived, per-feature (per ``(pair, ell-bin)`` column) vector,
    replacing the arbitrary fixed ``1e-10`` knee of the signed-log transform. ``asinh`` is the
    Lupton asinh-magnitude symlog: linear for ``|x| << s``, logarithmic for ``|x| >> s``, smooth
    through zero, sign-preserving, and invertible via ``x = s * sinh(y)``.

    Call fit() once on the raw (untransformed) training Cls before the training loop. The fitted
    scale is a non-trainable weight, so it is saved with the model checkpoint and restored
    automatically at evaluation / inference time — no separate file needed. This guarantees the
    same transform is applied across training, grid/mock/DES evaluation, and any later reload.
    """

    def __init__(self, floor=1e-30, **kwargs):
        super().__init__(**kwargs)
        self.floor = floor

    def build(self, input_shape):
        n_in = input_shape[-1]
        self.scale_ = self.add_weight("scale", shape=(n_in,), trainable=False, initializer="ones")
        super().build(input_shape)

    def fit(self, x, max_samples=200_000):
        """Set the per-feature scale to ``median(|x|, axis=0)`` from a (N, n_in) numpy array.

        Subsamples to max_samples rows (like PCAWhiteningLayer.fit). The floor guards against a
        zero scale on an all-zero feature column.
        """
        if not self.built:
            self.build((None, x.shape[-1]))

        rng = np.random.default_rng(0)
        if x.shape[0] > max_samples:
            idx = rng.choice(x.shape[0], size=max_samples, replace=False)
            x = x[idx]

        scale = np.median(np.abs(x.astype(np.float64)), axis=0)
        scale = np.maximum(scale, self.floor)
        LOGGER.info(
            f"AsinhScaleLayer: per-feature scale median(|x|) over {x.shape[1]} features, "
            f"range [{scale.min():.2e}, {scale.max():.2e}]"
        )
        self.scale_.assign(scale.astype(np.float32))

    def call(self, inputs):
        return tf.math.asinh(inputs / self.scale_)

    def get_config(self):
        config = super().get_config()
        config.update({"floor": self.floor})
        return config
