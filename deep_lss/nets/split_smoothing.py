# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created July 2026
Author: Arne Thomsen

Per-probe Gaussian smoothing at per-probe HEALPix resolutions.

With a single ``HealpySmoothing`` front-end, all channels share one base kernel at the
smallest requested FWHM, so the strongly smoothed clustering channels need
``ceil((fwhm / fwhm_min)^2)`` (~O(100)) sparse matmuls at the full map nside. This module
instead gives each probe its own kernel at its own nside: probes below the output nside are
downsampled in-network (the identical ``tf.math.unsorted_segment_mean`` the msfm pipeline
uses for ``downsample_nside``, which there runs as the last map op — so the result is the
same), smoothed and noise-augmented at the coarse nside with the existing per-channel
repetition scheme, and upsampled back (parent value repeated to its children) so the output
is a single multi-channel map at the output nside and downstream networks are unchanged.

The upsampling is a negligible approximation as long as the probe's smoothing scales are
much larger than the coarse pixel size (e.g. clustering FWHM >= 57' vs 13.7' nside-256
pixels for the 8wl,32gc scale cuts).
"""

import tensorflow as tf

from deepsphere import healpy_layers

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


class PerProbeSmoothing(tf.keras.Model):
    """Smooth each probe's channel block with its own ``HealpySmoothing`` at its own nside.

    Construct under a float32 mixed-precision policy (see ``_build_fp32_smoothing`` in
    ``transformer_networks``) so the sparse kernels stay in float32.

    Args:
        probe_specs (list of dict): one entry per probe, in channel order. Each entry has
            - ``probe`` (str): name, for logging only.
            - ``n_channels`` (int): number of consecutive channels belonging to the probe.
            - ``smoothing_kwargs`` (dict): kwargs for ``HealpySmoothing`` at the probe's nside
              (including per-probe fwhm, white_noise_sigma, and mask at that nside).
            - ``parent_output_idx`` (np.ndarray, optional): fine-to-coarse row map from
              ``configuration.get_smooth_nside_indices``. Present iff the probe's nside is below
              the output nside; drives both the in-network downsampling and the upsampling.
    """

    def __init__(self, probe_specs):
        super().__init__()

        self.probe_names = []
        self.n_channels = []
        self.n_pix_probe = []
        self.smoothing_layers = []
        self.parent_output_idxs = []

        for spec in probe_specs:
            kwargs = spec["smoothing_kwargs"]
            parent_output_idx = spec.get("parent_output_idx", None)
            LOGGER.warning(
                f"PerProbeSmoothing: probe {spec['probe']} with {spec['n_channels']} channels at "
                f"nside={kwargs['nside']}"
                + ("" if parent_output_idx is None else " (downsampled in-network, upsampled after smoothing)")
            )
            self.probe_names.append(spec["probe"])
            self.n_channels.append(spec["n_channels"])
            self.n_pix_probe.append(len(kwargs["indices"]))
            self.smoothing_layers.append(healpy_layers.HealpySmoothing(**kwargs))
            self.parent_output_idxs.append(
                None if parent_output_idx is None else tf.constant(parent_output_idx, dtype=tf.int32)
            )

    def call(self, x, training=False):
        outputs = []
        for smoothing, parent_output_idx, n_pix_coarse, x_probe in zip(
            self.smoothing_layers, self.parent_output_idxs, self.n_pix_probe, tf.split(x, self.n_channels, axis=-1)
        ):
            if parent_output_idx is not None:
                # downsample (B, P_fine, C) -> (B, P_coarse, C) by per-parent averaging, the
                # identical op msfm.grid_pipeline applies for downsample_nside
                x_t = tf.transpose(x_probe, perm=[1, 0, 2])  # (P_fine, B, C)
                x_t = tf.math.unsorted_segment_mean(x_t, parent_output_idx, n_pix_coarse)
                x_probe = tf.transpose(x_t, perm=[1, 0, 2])  # (B, P_coarse, C)

            x_probe = smoothing(x_probe, training=training)

            if parent_output_idx is not None:
                # upsample back by repeating each parent value to its children
                x_probe = tf.gather(x_probe, parent_output_idx, axis=1)

            outputs.append(x_probe)

        return tf.concat(outputs, axis=-1)
