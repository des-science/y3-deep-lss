# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created July 2026
Author: Arne Thomsen

Shared plumbing for the multi-resolution map encoders — the transformer
``HealpixMultiResMapEncoder`` and the GCNN ``ResNetMultiResEncoder``. Both smooth each probe at
its own nside (``PerProbeSmoothing``), group the probes by resolution
(``group_probe_specs_by_nside``), optionally standardize each group with its own
``EmpiricalInputNormalization``, and expose the ``smooth_groups`` / ``masks`` /
``load_input_norm_stats`` interface through which ``run_training`` measures the empirical
input-norm statistics. Only the trainable body (nested transformer with injections vs split
``HealpyGCNN`` segments with a concat+Dense fusion) differs, so that part stays in the concrete
classes.
"""

import numpy as np
import tensorflow as tf

from deep_lss.nets.layers.maps.smoothing import (
    PerProbeSmoothing,
    fp32_policy_scope,
    group_probe_specs_by_nside,
)
from deep_lss.nets.layers.maps.input_normalization import EmpiricalInputNormalization


class MultiResEncoderMixin:
    """Smoothing / grouping / input-norm plumbing shared by the multi-resolution map encoders.

    The two ``_init_*`` helpers are called from the concrete ``__init__`` at the same positions
    the previously inlined code occupied (smoothing before, input norm after the trainable body),
    so the attribute layout — and with it the checkpoint object graph — is unchanged.
    """

    def _init_smoothing_and_groups(self, smoothing_kwargs, spmm_backend="csr"):
        """Build ``self.smoothing`` (fp32 ``PerProbeSmoothing``) and the resolution groups.

        Returns the group list (finest first, see ``group_probe_specs_by_nside``), also stored as
        ``self._groups`` for ``smooth_groups``. ``spmm_backend`` selects the sparse-matmul backend
        for the per-probe smoothing kernels; both multi-res encoders forward the value they were
        built with (the app defaults it to "csr" via the net config).
        """
        if "split_probes" not in smoothing_kwargs:
            raise ValueError(f"{type(self).__name__} requires a split_probes smoothing spec.")
        specs = smoothing_kwargs["split_probes"]

        # sparse smoothing kept in float32 (no fast bf16 cuSPARSE kernel) — see fp32_policy_scope
        with fp32_policy_scope():
            self.smoothing = PerProbeSmoothing(specs, spmm_backend=spmm_backend)

        self._groups = group_probe_specs_by_nside(specs)
        return self._groups

    def _init_group_input_norm(self, input_norm):
        """Build the per-group ``EmpiricalInputNormalization`` layers (or None attributes).

        One layer per resolution group with the group's own concatenated footprint mask, already
        at the group's nside — no upsampling. Built fp32 by the layer itself; runs on the
        fp32-smoothed maps.
        """
        self.input_norms = None
        self._group_masks = None
        if input_norm:
            self.input_norms = []
            self._group_masks = []
            for g in self._groups:
                if any(m is None for m in g["masks"]):
                    mask = None
                else:
                    mask = np.concatenate([np.asarray(m) for m in g["masks"]], axis=-1)
                self.input_norms.append(EmpiricalInputNormalization(g["n_channels"], mask))
                self._group_masks.append(mask)

    def smooth_groups(self, maps, training=False):
        """Per-resolution-group smoothed maps (fp32), used to measure the input-norm statistics."""
        probe_tensors = self.smoothing(maps, training=training)
        out = []
        for g in self._groups:
            parts = [probe_tensors[i] for i in g["probe_ids"]]
            out.append(parts[0] if len(parts) == 1 else tf.concat(parts, axis=-1))
        return out

    @property
    def masks(self):
        """Per-group footprint masks in group order (aligned with ``smooth_groups``)."""
        return self._group_masks

    def load_input_norm_stats(self, stats):
        """Load a list of ``(mean, inv_std)`` per group into the per-group input-norm layers."""
        for norm, (mean, inv_std) in zip(self.input_norms, stats):
            norm.load_stats(mean, inv_std)
