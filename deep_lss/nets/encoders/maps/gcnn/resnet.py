# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created November 2023
Author: Arne Thomsen
"""

import tensorflow as tf
from deepsphere import healpy_layers

from deep_lss.nets.heads.regression_head import get_regression_head
from deep_lss.nets.layers.maps.input_normalization import EmpiricalInputNormalization
from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


class ResNetLayers:
    """Class used to build the layers of the ResNet network, which was used as the fiducial architecture in Janis'
    KiDS1000 analysis.
    """

    def __init__(
        self,
        out_features,
        # convolutions
        base_channels=32,
        pool_layers=3,
        conv_layers=2,
        residual_layers=6,
        # regression head
        head_type="dense",
        dense_layers=None,
        dropout_rate=None,
        # misc
        poly_degree=5,
        norm_kwargs={},
        activation=tf.nn.relu,
        smoothing_kwargs=None,
        input_norm=False,
        smoothing_external=False,
        spmm_backend="csr",
    ) -> None:
        """Class used to build the layers of the ResNet network, which was used as the fiducial architecture in Janis'
        KiDS1000 analysis.

        Args:
            out_features (int, optional): Output shape of the regression head. This determines the size of the learned
                summary statistics. Defaults to 6.
            base_channels (int, optional): Number of channels after the first layer of the network. This number gets
                multiplied by a factor of two for every downsampling layer. Defaults to 32.
            pool_layers (int, optional): Number of pseudoconvolutions to perform a downsampling of the
                neighboring Healpix pixels. Note that these layers are fairly cheap and their number effectively
                determines how expensive the following (residual) graph convolutions are. Defaults to 3.
            conv_layers (int, optional): Number of Chebyshev convolutions to downsample with. These layers play the
                same role as the pure downsampling layers, just include an additional (Chebyshev) graph convoution.
                Defaults to 2.
            residual_layers (int, optional): Number of residual layers. These are the main graph convolutions. Defaults
                to 6.
            head_type (str, optional): Type of regression head to be used, allowed are "dense" and "conv. Defaults to
                "dense".
            dropout_rate (float, optional): Dropout rate within the regression head. Defaults to None, then it's not
                included.
            poly_degree (int, optional): Degree of the polynomials within the Chebyshev convolutions. Defaults to 5.
            norm_kwargs (dict, optional): Keyword arguments to be passed to the normalization layers. Defaults to {}.
            activation (callable, optional): Non-linear activation function to be used throughout. Defaults to
                tf.nn.relu.
            smoothing_kwargs (dict, optional): Keyword arguments to be passed to the smoothing layer. Defaults to None,
                then no smoothing is performed within the network.
            input_norm (bool, optional): Standardize the smoothed maps with an EmpiricalInputNormalization
                layer placed right after the smoothing front-end (same placement as the transformer map
                encoders): per-channel mean/inv_std measured from training data on fresh runs
                (``compute_input_norm_stats`` via the ``smooth_groups``/``masks``/``load_input_norm_stats``
                interface below) and restored from the checkpoint otherwise. Adds checkpoint variables —
                a checkpoint trained with one setting can only be restored with the same one. Requires
                ``smoothing_kwargs`` (channel count and footprint mask). Defaults to False.
            smoothing_external (bool, optional): Set by ``ResNetMultiResEncoder`` when it builds
                this spec with ``smoothing_kwargs=None`` because smoothing (and input norm) live in
                the encoder instead — silences the missing-smoothing warning. Defaults to False.
            spmm_backend (str, optional): sparse-matmul backend for the (single-res) smoothing layer
                built here ("coo"/"csr"/"gather"; see deepsphere.utils.make_spmm_operator). Defaults
                to "csr" (cuSPARSE; numerically equivalent to "coo" and faster). The graph
                convolutions get their backend from the wrapping HealpyGCNN's own ``spmm_backend``
                (set by BaseModel / the maps+cls composite), not from here.
        """
        self.layers = []

        self.smoothing_layer = None
        self.input_norm_layer = None
        self._input_norm_mask = None

        if smoothing_kwargs is not None:
            if "split_probes" in smoothing_kwargs:
                raise ValueError(
                    "Per-probe smooth_nside (split_probes) is not consumed by ResNetLayers directly — "
                    "run_training dispatches it to ResNetMultiResEncoder, which owns the smoothing and "
                    "builds this spec with smoothing_kwargs=None"
                )
            self.smoothing_layer = healpy_layers.HealpySmoothing(**smoothing_kwargs, spmm_backend=spmm_backend)
            self.layers.append(self.smoothing_layer)
        elif smoothing_external:
            LOGGER.info("Smoothing (and input norm) handled externally by ResNetMultiResEncoder")
        else:
            LOGGER.warning("No smoothing layer is included in the network")

        if input_norm:
            if smoothing_kwargs is None:
                raise ValueError(
                    "input_norm requires smoothing_kwargs (per-channel fwhm for the channel count and the "
                    "footprint mask)"
                )
            # same config geometry as the smoothing front-end (cf. _input_norm_footprint for the
            # transformer encoders); the layer binarizes it to a 0/1 indicator itself
            self._input_norm_mask = smoothing_kwargs.get("mask")
            self.input_norm_layer = EmpiricalInputNormalization(
                len(smoothing_kwargs["fwhm"]), self._input_norm_mask
            )
            self.layers.append(self.input_norm_layer)

        # downsampling and increasing channels
        n_channels = base_channels
        for _ in range(pool_layers):
            self.layers.append(healpy_layers.HealpyPseudoConv(p=1, Fout=n_channels, activation=activation))
            n_channels *= 2

        # downsampling and Chebyshev convolutions
        for _ in range(conv_layers):
            self.layers.append(healpy_layers.HealpyChebyshev(K=poly_degree, Fout=n_channels, activation=activation))
            self.layers.append(tf.keras.layers.LayerNormalization(**{"axis": -1, **norm_kwargs}))
            self.layers.append(healpy_layers.HealpyPseudoConv(p=1, Fout=n_channels, activation=activation))

        # residual Chebyshev convolutions
        for _ in range(residual_layers):
            self.layers.append(
                healpy_layers.Healpy_ResidualLayer(
                    "CHEBY",
                    layer_kwargs={"K": poly_degree, "activation": activation, "use_bias": True},
                    # the delta loss is only compatible with layer and not batch normalization
                    use_bn=True,
                    bn_kwargs=norm_kwargs,
                    norm_type="layer_norm",
                )
            )

        # snapshot conv-only layers before the regression head is appended
        self._conv_layers = list(self.layers)

        # regression head
        regression_head_layers = get_regression_head(
            out_features=out_features,
            head_type=head_type,
            dense_layers=dense_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            poly_degree=poly_degree,
            norm_kwargs=norm_kwargs,
        )
        # head without the leading Flatten — used when Cls are concatenated after flattening
        self._head_layers_no_flatten = regression_head_layers[1:] if head_type == "dense" else regression_head_layers
        self.layers.extend(regression_head_layers)

    def get_layers(self):
        return self.layers

    # --- empirical input-norm interface, mirroring the transformer map encoders
    # (HealpixMapEncoder) so run_training measures the statistics through a single code path.
    # The layer objects below are the SAME instances that live in self.layers, so loading the
    # statistics through this spec reaches the layer inside the built network / composite.

    def smooth_groups(self, maps, training=False):
        """Smoothed maps (fp32) as a one-element list, used to measure the input-norm statistics."""
        return [maps if self.smoothing_layer is None else self.smoothing_layer(maps, training=training)]

    @property
    def masks(self):
        """Footprint mask as a one-element list (aligned with ``smooth_groups``)."""
        return [self._input_norm_mask]

    def load_input_norm_stats(self, stats):
        """Load the single ``(mean, inv_std)`` group into the input-norm layer."""
        (mean, inv_std), = stats
        self.input_norm_layer.load_stats(mean, inv_std)

    def get_conv_layers(self):
        """Return only the graph-convolution layers, without the regression head."""
        return self._conv_layers

    def get_head_layers_no_flatten(self):
        """Return the regression head layers without the leading Flatten layer.

        Used by ResNetMapsPlusCLSNetwork which performs its own flatten-and-concat step before the head.
        Only meaningful for head_type='dense'; for 'conv' heads the full head is returned.
        """
        return self._head_layers_no_flatten
