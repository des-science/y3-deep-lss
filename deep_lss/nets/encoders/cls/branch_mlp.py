# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Standalone twin of the Cls branch that lives inside the maps+Cls composite networks.

``MultiLayerPerceptron`` is NOT that branch: it is 1024 wide where the branch is 512, it norms
only at the input where the branch norms after every Dense, and it is fed through a PCA rotation
the branch does not have. So the two-point baseline and the Cls half of a maps+Cls run were two
different encoders on the same data vector.

This class closes that gap by building itself out of the SAME two functions the composite networks
use -- ``get_cls_embedding_layers`` and ``get_regression_head`` -- rather than re-typing their
layer stacks. From ``cls_norm`` onward it is the identical function to
``ResNetSummaryNetwork``/``TransformerSummaryNetwork``'s Cls path; only the fusion concat and the
map branch are missing, which is what "standalone" means.

The preprocessing in front of it is numerically equivalent rather than shared, and unavoidably so:
this path reads the pre-binned ``hard_rebinned`` cache and applies ``AsinhScaleLayer``, while the
composite nets bin raw per-ell Cls in-graph with ``ClsBinningAndTransformLayer``. Both take their
bin edges from ``msfm.utils.power_spectra.get_cl_bins(l_min, l_max, n_bins + 1)`` per pair and
both fit the asinh scale as ``median(|x|)`` per feature.

Author: Arne Thomsen
"""

import tensorflow as tf

from deep_lss.nets.heads.regression_head import get_regression_head
from deep_lss.nets.layers.cls.embedding import get_cls_embedding_layers


class ClsBranchMLP(tf.keras.Model):
    """Cls-only summary encoder matching the maps+Cls networks' Cls branch.

    Input contract matches the other Cls encoders: a flat ``(B, n_cls)`` vector of rebinned Cls
    (``n_cls = cls_n_bins * n_pairs``), output ``(B, output_size)``.

    Args:
        output_size (int): Summary dimension (``n_summary``).
        embedding_layers (list[int]): Widths of the embedding MLP. Defaults to the composite
            networks' own default, ``[512, 512, 512, 512]``.
        dropout_rate (float or None): Dropout inside the embedding; ``None`` disables it.
        dropout_per_layer (bool): Dropout after every Dense+LayerNorm block (True, the default and
            what the maps+Cls runs train with) or once after the stack. Checkpoints are
            position-sensitive, so this may not be changed for a restore.
        activation (str): Activation of the embedding's Dense layers.
        input_transform (tf.keras.layers.Layer or None): Per-feature ``AsinhScaleLayer``, fitted
            and passed in by the caller. PCA whitening is deliberately NOT accepted: the branch
            this mirrors does not have one.
        head_dense_layers (list[int] or None): Hidden widths of the regression head. ``None`` gives
            the production head, ``LayerNorm -> Dense(output_size)``.
        head_dropout_rate (float or None): Dropout in the regression head. ``None`` matches the
            production GCNN; the transformer answers this knob the other way.
    """

    def __init__(
        self,
        output_size,
        embedding_layers=None,
        dropout_rate=0.1,
        dropout_per_layer=True,
        activation="relu",
        input_transform=None,
        head_dense_layers=None,
        head_dropout_rate=None,
    ):
        super().__init__()
        self.input_transform = input_transform

        # Mirrors ResNetSummaryNetwork.cls_norm: the branch is normalised on its own before the
        # embedding. There it balances the two branches before the concat; here there is nothing
        # to balance against, but dropping it would change the function this class exists to match.
        self.cls_norm = tf.keras.layers.LayerNormalization(axis=-1, name="cls_norm")

        self.cls_embedding_layers = get_cls_embedding_layers(
            embedding_layers if embedding_layers is not None else [512, 512, 512, 512],
            dropout_rate=dropout_rate,
            activation=activation,
            dropout_per_layer=dropout_per_layer,
        )
        self.regression_head_layers = get_regression_head(
            output_size,
            head_type="dense",
            dense_layers=head_dense_layers,
            activation=activation,
            dropout_rate=head_dropout_rate,
        )

    def call(self, inputs, training=False):
        x = self.input_transform(inputs) if self.input_transform is not None else inputs
        x = self.cls_norm(x, training=training)
        for layer in self.cls_embedding_layers:
            x = layer(x, training=training)
        # The head opens with a Flatten, a no-op on the (B, width) embedding output.
        for layer in self.regression_head_layers:
            x = layer(x, training=training)
        return x
