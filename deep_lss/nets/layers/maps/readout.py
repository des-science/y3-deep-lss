# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created August 2025
Author: Arne Thomsen

Permutation-invariant readouts for the GCNN map branch, and the plumbing that taps one at every
resolution instead of only at the trunk.

The readout is the highest-leverage lever measured in this programme -- ``flatten -> mean`` was
+32.1% paired and ``mean -> mean_std`` a further +5.3% -- and the cheapest, since it acts on the
trunk's final (coarsest) feature map. ``moment_pool`` is the ladder of statistics; the
``*_taps`` helpers let a caller apply it at each scale of the encoder.

Kept out of any particular network body so the summary network (``composite/resnet_summary.py``)
and the multi-resolution encoder (``encoders/maps/gcnn/resnet_multires.py``) can share it -- the
encoder owns the fused seam and therefore has to assemble its own tap list, so both need these.
"""

import tensorflow as tf

from deepsphere import healpy_layers as hp_nn

# Layers that reduce the nside. Used to find the multi-scale readout's tap points; matches the
# reduction bookkeeping in HealpyGCNN.__init__ (healpy_networks.py), minus HealpyPseudoConv_Transpose
# which upsamples and is not part of any encoder in this repo.
_DOWNSAMPLING_LAYERS = (hp_nn.HealpyPool, hp_nn.HealpyPseudoConv, hp_nn.Healpy_ViT)

# Guard on the variance before the sqrt: a dead channel gives variance exactly 0 in fp32, where
# sqrt's gradient is infinite. It also floors std at 1e-3, which is what keeps the standardized
# third/fourth moments finite (see moment_pool).
_VAR_EPS = 1e-6


def moment_pool(x, kind):
    """Permutation-invariant readout of ``(B, n_pix, n_ch)`` conv features over the pixel axis.

    The GCNN trunk runs on footprint pixels only (no padding), so every reduction here is over the
    footprint and the plain mean is the footprint mean.

    Args:
        x (tf.Tensor): conv features, ``(B, n_pix, n_ch)``.
        kind (str): ``"mean"`` -> ``(B, n_ch)``; ``"mean_std"`` -> ``(B, 2*n_ch)``;
            ``"moments"`` -> ``(B, 4*n_ch)``.

    Returns:
        tf.Tensor: the pooled readout.

    ``"moments"`` appends the STANDARDIZED third and fourth central moments (skewness, kurtosis).
    Standardizing is not cosmetic: the raw central moments of a LayerNorm'd feature map span orders
    of magnitude across channels and would reach ``map_norm`` with a conditioning problem the
    mean/std pair does not have. Dividing by ``std**k`` inherits the ``_VAR_EPS`` guard (std >= 1e-3,
    so std**4 >= 1e-12), and a dead channel gives 0/eps**k = 0 rather than a NaN.
    """
    mean = tf.reduce_mean(x, axis=1)  # (B, n_ch)
    if kind == "mean":
        return mean

    centered = x - mean[:, None, :]
    c2 = tf.square(centered)
    variance = tf.reduce_mean(c2, axis=1)
    std = tf.sqrt(variance + _VAR_EPS)
    if kind == "mean_std":
        return tf.concat([mean, std], axis=-1)

    # explicit products rather than tf.pow: same result, cheaper, and c2 is already materialised
    skew = tf.reduce_mean(c2 * centered, axis=1) / (std**3)
    kurt = tf.reduce_mean(c2 * c2, axis=1) / (std**4)
    return tf.concat([mean, std, skew, kurt], axis=-1)


def forward_with_pool_taps(gcnn, x, training):
    """Run a ``HealpyGCNN`` layer by layer, tapping the output of every resolution-reducing layer.

    ``HealpyGCNN`` is a ``Sequential``, so iterating ``gcnn.layers`` reproduces its forward pass
    exactly; the loop exists only to keep the intermediate tensors. Keras' ``Layer.__call__`` drops
    the ``training`` kwarg for layers whose ``call`` does not accept it, so passing it unconditionally
    is safe.

    Args:
        gcnn (HealpyGCNN): the (already built) map branch.
        x (tf.Tensor): input features.
        training (bool): Keras training flag.

    Returns:
        tuple: ``(final_output, pool_taps)`` where ``pool_taps`` holds one tensor per downsampling
        layer, in depth order.
    """
    pool_taps = []
    for layer in gcnn.layers:
        x = layer(x, training=training)
        if isinstance(layer, _DOWNSAMPLING_LAYERS):
            pool_taps.append(x)
    return x, pool_taps


def count_scale_taps(gcnn):
    """How many tensors ``assemble_scale_taps`` will return for this branch (excluding any seam tap).

    Counted from the layer list at construction time so the per-tap LayerNorms can be created up
    front rather than lazily on the first call — that keeps ``summary()`` complete before the trace
    and the checkpoint structure fixed.
    """
    n_down = sum(isinstance(layer, _DOWNSAMPLING_LAYERS) for layer in gcnn.layers)
    return max(n_down - 1, 0) + 1


def assemble_scale_taps(pool_taps, final):
    """Combine downsampling taps with the trunk output into the multi-scale readout's tap list.

    Drops the LAST downsampling tap: the residual body runs immediately after it at the same nside
    and width, so ``final`` already represents that scale and keeping both would pool the same
    resolution twice. When there is no residual body the last tap IS ``final``, and dropping it
    leaves exactly one tensor per scale either way.
    """
    return list(pool_taps[:-1]) + [final]
