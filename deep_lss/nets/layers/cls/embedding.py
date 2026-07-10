# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2024
Author: Arne Thomsen

Builder for the small MLP that embeds the binned power spectra (Cls) before they are fused with
the map features in the maps + Cls composite networks
(``deep_lss.nets.composite.resnet_maps_plus_cls`` /
``deep_lss.nets.composite.transformer_maps_plus_cls``).
"""

import tensorflow as tf


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
