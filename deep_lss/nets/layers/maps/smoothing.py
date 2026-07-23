# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created July 2026
Author: Arne Thomsen

HEALPix Gaussian smoothing for the map branches.

The smoothing front-end itself, ``HealpySmoothing`` (with its ``deepsphere.utils`` helpers
``make_spmm_operator`` / ``split_sparse_dense_matmul`` / ``GaussianNoiseLayer``), lives in
``deepsphere-cosmo-tf2`` — it is a topical fit there, being built on the same sparse graph
operations, and is used unchanged by the DeepSphere GCNNs. It is re-exported here so the map
branches have a single ``deep_lss.nets.layers.maps.smoothing`` entry point; the single-resolution
transformer front-end imports it from this module.

This module only adds the msfm-coupled wrappers that build on ``HealpySmoothing``:

``PerProbeSmoothing`` — with a single ``HealpySmoothing`` front-end, all channels share one base
kernel at the smallest requested FWHM, so the strongly smoothed clustering channels need
``ceil((fwhm / fwhm_min)^2)`` (~O(100)) sparse matmuls at the full map nside. ``PerProbeSmoothing``
instead gives each probe its own kernel at its own nside: probes below the output nside are
downsampled in-network (the identical ``tf.math.unsorted_segment_mean`` the msfm pipeline uses for
``downsample_nside``, which there runs as the last map op — so the result is the same), then
smoothed and noise-augmented at the coarse nside with the existing per-channel repetition scheme.

``PerProbeSmoothing.call`` returns one tensor per probe, each at its OWN nside (finest probe at the
output nside, coarser probes at their coarse nside) — it does NOT upsample coarse probes back to a
common resolution. The multi-resolution transformer encoder consumes these separately, feeding the
coarse probe into the hierarchy at the level that already runs at its nside
(``HealpixMultiResMapEncoder``), so clustering is never upsampled. The transformer encoders,
``PerProbeSmoothing`` and both multi-resolution encoders (including the GCNN
``ResNetMultiResEncoder``) all consume the re-exported ``HealpySmoothing``.
"""

import numpy as np
import tensorflow as tf

from deepsphere.healpy_layers import HealpySmoothing  # re-exported; single source of truth

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)

__all__ = ["HealpySmoothing", "PerProbeSmoothing", "fp32_policy_scope", "group_probe_specs_by_nside"]


class PerProbeSmoothing(tf.keras.Model):
    """Smooth each probe's channel block with its own ``HealpySmoothing`` at its own nside.

    Construct under a float32 mixed-precision policy (see ``fp32_policy_scope`` below) so the
    sparse kernels stay in float32.

    Args:
        probe_specs (list of dict): one entry per probe, in channel order. Each entry has
            - ``probe`` (str): name, for logging only.
            - ``n_channels`` (int): number of consecutive channels belonging to the probe.
            - ``smoothing_kwargs`` (dict): kwargs for ``HealpySmoothing`` at the probe's nside
              (including per-probe fwhm, white_noise_sigma, and mask at that nside).
            - ``parent_output_idx`` (np.ndarray, optional): fine-to-coarse row map from
              ``configuration.get_smooth_nside_indices``. Present iff the probe's nside is below
              the output nside; drives the in-network downsampling to the probe's coarse nside. The
              coarse probe is then injected into the transformer hierarchy at that scale (see
              ``HealpixMultiResMapEncoder``), never upsampled back.
        spmm_backend (str): sparse-matmul backend for every per-probe ``HealpySmoothing`` kernel
            application ("coo"/"csr"/"gather"; see ``deepsphere.utils.make_spmm_operator``).
            Defaults to "csr" (cuSPARSE; numerically equivalent to "coo" up to fp32 tolerance and
            faster). The backend operator is not a checkpointed variable, so switching it does not
            alter the checkpoint object graph.
    """

    def __init__(self, probe_specs, spmm_backend="csr"):
        super().__init__()

        self.probe_names = []
        self.n_channels = []
        self.n_pix_probe = []
        self.probe_nsides = []
        self.probe_indices = []
        self.probe_masks = []
        self.smoothing_layers = []
        self.parent_output_idxs = []

        for spec in probe_specs:
            kwargs = spec["smoothing_kwargs"]
            parent_output_idx = spec.get("parent_output_idx", None)
            LOGGER.warning(
                f"PerProbeSmoothing: probe {spec['probe']} with {spec['n_channels']} channels at "
                f"nside={kwargs['nside']}"
                + ("" if parent_output_idx is None else " (downsampled in-network, kept at coarse nside)")
            )
            self.probe_names.append(spec["probe"])
            self.n_channels.append(spec["n_channels"])
            self.n_pix_probe.append(len(kwargs["indices"]))
            self.probe_nsides.append(int(kwargs["nside"]))
            self.probe_indices.append(kwargs["indices"])
            self.probe_masks.append(kwargs.get("mask"))
            self.smoothing_layers.append(HealpySmoothing(**kwargs, spmm_backend=spmm_backend))
            self.parent_output_idxs.append(
                None if parent_output_idx is None else tf.constant(parent_output_idx, dtype=tf.int32)
            )

    def call(self, x, training=False):
        # Returns one tensor per probe, each at its own nside (coarse probes stay coarse) —
        # see the module docstring. The multi-resolution encoder routes them separately.
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
            outputs.append(x_probe)

        return outputs


def fp32_policy_scope():
    """Context manager that forces the global mixed-precision policy to float32.

    The smoothing front-end (``HealpySmoothing`` / ``PerProbeSmoothing``) reads
    ``tf.keras.mixed_precision.global_policy()`` at construction to pick the dtype of its sparse
    kernel. Under a bf16/fp16 policy that makes the (eager) ``tf.sparse.sparse_dense_matmul`` run
    in low precision, which has no fast cuSPARSE kernel and is ~10x slower (benchmarked). Building
    it in float32 keeps the sparse smoothing fast; the network casts the smoothed maps to the
    body's compute dtype afterwards, so the body still gets the bf16 benefit.
    """
    from contextlib import contextmanager

    @contextmanager
    def _scope():
        prev_policy = tf.keras.mixed_precision.global_policy()
        tf.keras.mixed_precision.set_global_policy("float32")
        try:
            yield
        finally:
            tf.keras.mixed_precision.set_global_policy(prev_policy)

    return _scope()


def group_probe_specs_by_nside(specs):
    """Group ``split_probes`` specs by nside, finest group first.

    Probes sharing an nside are concatenated into one group (same footprint required); the spec
    (and therefore channel) order is preserved within and across groups. Consumed by the
    multi-resolution encoders (transformer ``HealpixMultiResMapEncoder`` and GCNN
    ``ResNetMultiResEncoder``), which take the finest group as the main network input and inject
    the coarser groups at their own scale.

    Args:
        specs (list of dict): the ``split_probes`` spec list from
            ``configuration.get_smoothing_kwargs`` (see ``PerProbeSmoothing``).

    Returns:
        list of dict: one entry per resolution, sorted finest first, with keys
            ``nside`` (int), ``probe_ids`` (spec indices), ``n_channels`` (summed),
            ``indices`` (shared footprint pixel ids), and ``masks`` (per-probe mask list).
    """
    nside_to_group = {}
    groups = []
    for i, spec in enumerate(specs):
        sk = spec["smoothing_kwargs"]
        g_nside = int(sk["nside"])
        if g_nside not in nside_to_group:
            g = {"nside": g_nside, "probe_ids": [], "n_channels": 0, "indices": None, "masks": []}
            nside_to_group[g_nside] = g
            groups.append(g)
        g = nside_to_group[g_nside]
        g["probe_ids"].append(i)
        g["n_channels"] += int(spec["n_channels"])
        idx = np.asarray(sk["indices"])
        if g["indices"] is None:
            g["indices"] = idx
        elif not np.array_equal(g["indices"], idx):
            raise ValueError(f"probes sharing nside {g_nside} have different footprints.")
        g["masks"].append(sk.get("mask"))

    # finest group first
    groups.sort(key=lambda g: g["nside"], reverse=True)
    return groups
