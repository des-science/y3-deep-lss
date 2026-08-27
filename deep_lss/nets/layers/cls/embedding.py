# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2024
Author: Arne Thomsen

Builder for the small MLP that embeds the binned power spectra (Cls) before they are fused with
the map features in the summary networks (``deep_lss.nets.composite.resnet_summary`` /
``deep_lss.nets.composite.transformer_summary``), plus ``get_cls_branch_kwargs``, which resolves
a ``cls:`` config block into the constructor arguments those two networks share.
"""

import tensorflow as tf

from deep_lss.utils import configuration


def get_cls_embedding_layers(hidden_layers, dropout_rate=None, activation="relu", dropout_per_layer=True):
    """Build an MLP to embed binned Cls before fusion with map features in the maps+Cls networks.

    Args:
        hidden_layers: List of int widths, e.g. ``[512, 512, 512, 512]``.
            ``None`` or empty list → returns ``[]`` (no embedding).
        dropout_rate: Optional float dropout probability (``None`` → no Dropout at all).
        activation: Activation for hidden Dense layers.
        dropout_per_layer: If ``True`` (default) a Dropout is applied after every Dense+LN block;
            if ``False`` a single Dropout is appended after all hidden layers. Per-layer is the
            intended default; use a lower ``dropout_rate`` than a single trailing Dropout would.
            Note: object-based checkpointing is position-sensitive, so the two placements produce
            incompatible layer orderings — a checkpoint trained with one placement can only be
            restored with the same one. Historical runs trained with the trailing placement (whose
            saved configs predate the ``cls.embedding_dropout_per_layer`` key) must set that key to
            ``false`` to be re-evaluated or resumed.

    Returns:
        List of Keras layers: interleaved Dense + LayerNorm, with Dropout placed either after
        every block (default, ``dropout_per_layer=True``) or once at the end
        (``dropout_per_layer=False``), when ``dropout_rate`` is set.
    """
    if not hidden_layers:
        return []
    layers = []
    for h in hidden_layers:
        layers.append(tf.keras.layers.Dense(h, activation=activation))
        layers.append(tf.keras.layers.LayerNormalization(axis=-1))
        if dropout_rate is not None and dropout_per_layer:
            layers.append(tf.keras.layers.Dropout(dropout_rate))
    if dropout_rate is not None and not dropout_per_layer:
        layers.append(tf.keras.layers.Dropout(dropout_rate))
    return layers


def get_cls_branch_kwargs(cls_conf, msfm_conf, dlss_conf, tfr_n_side, cls_transform):
    """Resolve a ``cls:`` config block into the Cls-branch constructor kwargs.

    ``ResNetSummaryNetwork`` and ``TransformerSummaryNetwork`` take the same five Cls arguments and
    treat them as all-or-nothing: supplying them builds the Cls branch and concatenates it onto the
    map features, omitting them builds the maps-only network. This is the single place that reads
    the block, so training, evaluation and the benchmarks cannot disagree about what a config means
    — a Cls branch built one way at training and another at evaluation restores without complaint
    and silently computes a different function.

    Args:
        cls_conf (dict or None): the run's ``cls:`` config block; ``None`` for a maps-only run.
        msfm_conf (dict): msfm config, for the per-pair ell bounds.
        dlss_conf (dict): deep_lss config, for the per-pair ell bounds.
        tfr_n_side (int): native HEALPix n_side of the TFRecords (the Cls are not downsampled).
        cls_transform (str): "asinh_per_feature" or "log1p_fixed".

    Returns:
        dict: constructor kwargs to splat into the summary network, or ``{}`` when ``cls_conf`` is
        None (which is what makes the network maps-only).
    """
    if cls_conf is None:
        return {}
    _, l_min_per_pair, l_max_per_pair = configuration.get_cls_bounds_per_pair(msfm_conf, dlss_conf)
    return {
        "tfr_n_side": tfr_n_side,
        "n_cls_bins": cls_conf.get("n_bins", 16),
        "l_min_per_pair": l_min_per_pair,
        "l_max_per_pair": l_max_per_pair,
        "cls_embedding_layers": get_cls_embedding_layers(
            cls_conf.get("embedding_layers", [512, 512, 512, 512]),
            dropout_rate=cls_conf.get("embedding_dropout_rate", None),
            dropout_per_layer=cls_conf.get("embedding_dropout_per_layer", True),
        ),
        "cls_transform": cls_transform,
    }
