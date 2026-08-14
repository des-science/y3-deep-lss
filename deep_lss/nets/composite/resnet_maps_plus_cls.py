# Copyright (C) 2025 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created June 2025
Author: Arne Thomsen

Network that concatenates DeepSphere map features with binned angular power spectra (Cls).
The DeepSphere GCNN map branch is built from ``ResNetLayers`` conv/head layer lists (the only
encoder exposing ``get_conv_layers`` / ``get_head_layers_no_flatten``), hence ``ResNet`` in the
name; the transformer-branch counterpart is
``deep_lss.nets.composite.transformer_maps_plus_cls.TransformerMapsPlusCLSNetwork``.

Architecture:
  1. HealpyGCNN processes the HEALPix maps → flatten (or mean-pool over pixels, map_pool="mean")
     [→ linear Dense(map_feature_dim)] → map_norm (LN)                     (map branch)
  2. ClsBinningAndTransformLayer bins + gathers + sign-log-transforms Cls
     → cls_norm (LN) → cls_embedding MLP (Dense→LN × N)                    (Cls branch)
  3. Concatenate both branches
  4. regression_head (LN + hidden Dense layers + output)
"""

import tensorflow as tf

from deepsphere import HealpyGCNN

from msfm.utils import logger

from deep_lss.nets.layers.cls.binning import ClsBinningAndTransformLayer
from deep_lss.nets.layers.maps.readout import (
    assemble_scale_taps,
    count_scale_taps,
    forward_with_pool_taps,
    moment_pool,
)

LOGGER = logger.get_logger(__file__)


class ResNetMapsPlusCLSNetwork(tf.keras.Model):
    """Maps + Cls combined network.

    Processes HEALPix maps with a DeepSphere HealpyGCNN, then concatenates the
    Cls branch (per-pair binned, sign-log-transformed, encoded by a small MLP)
    to the flattened GCNN output before the regression head.  Each branch is
    independently LayerNorm'd; the Cls embedding further processes the Cls
    features before fusion.
    """

    def __init__(
        self,
        conv_layers=None,
        cls_embedding_layers=None,
        regression_head_layers=None,
        n_side=None,
        tfr_n_side=None,
        indices=None,
        n_neighbors=None,
        max_batch_size=None,
        initial_Fin=None,
        n_cls_bins=None,
        l_min_per_pair=None,
        l_max_per_pair=None,
        cls_transform="asinh_per_feature",
        map_feature_dim=None,
        map_encoder=None,
        map_pool=None,
        map_pool_multiscale=False,
        spmm_backend="csr",
    ):
        """
        Args:
            conv_layers (list): Graph-convolution layers (ResNetLayers.get_conv_layers()).
                Mutually exclusive with ``map_encoder``.
            cls_embedding_layers (list): MLP layers that encode the Cls branch before fusion
                (get_cls_embedding_layers()).  Pass ``[]`` to skip the embedding.
            regression_head_layers (list): Dense head layers without the leading Flatten
                (ResNetLayers.get_head_layers_no_flatten()).
            n_side (int): HEALPix n_side of the input maps (after any downsampling/smoothing)
                used to build the GCNN graph.
            tfr_n_side (int): Native HEALPix n_side of the TFRecords, i.e. the simulation
                resolution. The Cls stored in the TFRecords are not downsampled, so
                ``n_ell = 3 * tfr_n_side``.
            indices (np.ndarray): 1-D array of HEALPix NEST pixel indices in the footprint.
            n_neighbors (int): Number of neighbours for the HealpyGCNN graph.
            max_batch_size (int): Pre-allocated max batch size for sparse-dense matmul splits.
            initial_Fin (int): Number of input map channels (z-bins).
            n_cls_bins (int): Number of ell bins per cross pair.
            l_min_per_pair (list[float]): Per-pair lower bin edge (from scales config).
            l_max_per_pair (list[float]): Per-pair upper bin edge = l_max_eff (from scales config).
            cls_transform (str): Cls transform, forwarded to ClsBinningAndTransformLayer
                ("asinh_per_feature" or "log1p_fixed").
            map_feature_dim (int, optional): Bottleneck width of the map branch at the fusion
                point, mirroring the transformer composite: a linear Dense projects the flattened
                GCNN features (~1e5-dim) down to this width before map_norm, so both branches
                meet the concatenation at comparable dimensionality. ``None`` (default) keeps the
                legacy behavior of concatenating the raw flattened features. Note: object-based
                checkpointing is structure-sensitive — a checkpoint trained without the projection
                can only be restored with ``map_feature_dim`` unset, and vice versa.
            map_encoder (tf.keras.Model, optional): Prebuilt map branch returning
                ``(B, n_pix_reduced, n_ch)`` conv features — the multi-resolution
                ``ResNetMultiResEncoder`` (per-probe ``smooth_nside``). Mutually exclusive with
                ``conv_layers``/``n_side``/``indices``/``initial_Fin``; smoothing and input norm
                then live inside the encoder. Like ``map_feature_dim``, this is a separate
                checkpoint lineage: a multi-res checkpoint restores only with the matching
                ``smooth_nside`` mapping set, and vice versa.
            map_pool (str, optional): map-branch readout. ``None`` (default) flattens the
                ``(B, n_pix, n_ch)`` conv features to ``(B, n_pix*n_ch)`` (legacy). ``"mean"`` instead
                mean-pools over the pixel axis to ``(B, n_ch)``, mirroring the transformer map encoder's
                masked-mean token pool — a permutation-invariant readout that replaces the ~1e5->dim
                linear crush with a small ``n_ch->map_feature_dim`` Dense (set ``map_feature_dim`` for the
                "+ small Dense" variant; leave it ``None`` to use the raw pooled ``n_ch`` vector). The GCNN
                trunk runs on footprint pixels only (no padding), so the plain mean is the footprint mean.
                Separate checkpoint lineage (changes ``map_projection``'s input width): restore only with
                the same ``map_pool``.

                ``"mean_std"`` concatenates the per-channel standard deviation ACROSS the pixel axis
                onto the mean, giving ``(B, 2*n_ch)``. Motivation: ``"mean"`` keeps only the first
                moment, so it discards all across-sky variance — and since the GCNN trunk is a stack of
                LOCAL graph convolutions, a plain mean readout is approximately "the average of local
                statistics". For a field whose cosmological information is largely non-Gaussian that
                variance is signal, and a per-channel spatial variance is a direct second-moment
                statistic the head would otherwise have to do without. Compute is negligible (one extra
                reduction); the cost is that the fused width doubles, so it also rebalances the map
                branch against the Cls embedding — see ``map_feature_dim`` if that needs pinning back.
                Separate checkpoint lineage from ``"mean"`` (different readout width).

                ``"moments"`` appends the STANDARDIZED third and fourth central moments on top of
                ``"mean_std"``, giving ``(B, 4*n_ch)``. The moments of the convergence field are the
                classical non-Gaussian statistic in weak lensing and the third is the one that breaks
                the Om-sigma8 degeneracy the variance alone leaves, so this is the next term in the
                same expansion rather than an arbitrary extension. See ``moment_pool``.
            map_pool_multiscale (bool): apply ``map_pool``'s reduction at EVERY resolution of the map
                branch instead of only at the trunk output, concatenating the results. Requires a
                pooled ``map_pool`` (the flatten readout has no scale structure to tap).

                Taps are the output of each downsampling stage plus the trunk output — for the
                multi-res encoder the fused seam is included as the finest tap, since that is where
                both probes first meet. Each tap is reduced independently and gets its OWN
                LayerNorm before the concatenation: the taps have very unequal widths (the trunk is
                typically 8-16x the seam), and a single LayerNorm over the concatenated vector would
                take its statistics almost entirely from the widest tap and squash the fine scales —
                which are exactly the ones carrying the non-Gaussian signal this readout exists to
                keep. Per-tap normalization costs ~2*sum(widths) parameters, i.e. nothing.

                Only meaningful on an encoder that actually convolves at several resolutions (the
                U-net schedule: ``pool_layers`` 1, ``conv_layers`` 4, ``conv_widen``). On the default
                trunk the first stages are pure strided pseudo-convs with no graph convolution until
                nside 64, so the fine taps would pool barely-processed downsampled maps.
                Separate checkpoint lineage (readout width, and the per-tap norms are new variables).
        """
        super().__init__()

        if (map_encoder is None) == (conv_layers is None):
            raise ValueError("pass exactly one of conv_layers (single-res GCNN) or map_encoder (multi-res)")
        if map_encoder is not None and any(a is not None for a in (n_side, indices, initial_Fin)):
            raise ValueError("map_encoder owns the map branch — n_side/indices/initial_Fin must be None")
        if map_pool not in (None, "mean", "mean_std", "moments"):
            raise ValueError(f"map_pool must be None (flatten), 'mean', 'mean_std' or 'moments', got {map_pool!r}")
        if map_pool_multiscale and map_pool is None:
            raise ValueError(
                "map_pool_multiscale requires a pooled map_pool ('mean', 'mean_std' or 'moments'); "
                "the flatten readout has no scale structure to tap"
            )
        self.map_pool = map_pool
        self.map_pool_multiscale = map_pool_multiscale

        # keep the single-res attribute creation order unchanged (object-based checkpointing is
        # structure-sensitive; a None attribute is untracked, so the map_encoder path adds no
        # variables to the single-res object graph and vice versa)
        self.gcnn = (
            HealpyGCNN(
                nside=n_side,
                indices=indices,
                layers=conv_layers,
                n_neighbors=n_neighbors,
                max_batch_size=max_batch_size,
                initial_Fin=initial_Fin,
                spmm_backend=spmm_backend,
            )
            if map_encoder is None
            else None
        )
        self.map_encoder = map_encoder

        self.cls_layer = ClsBinningAndTransformLayer(
            n_ell=3 * tfr_n_side,
            n_bins=n_cls_bins,
            l_min_per_pair=l_min_per_pair,
            l_max_per_pair=l_max_per_pair,
            cls_transform=cls_transform,
        )

        # Optional map-branch bottleneck: linear, like the transformer's final Dense(num_outputs)
        # projection; the LayerNorm below then normalises the projected features.
        self.map_projection = (
            tf.keras.layers.Dense(map_feature_dim, name="map_projection") if map_feature_dim is not None else None
        )

        # Per-tap LayerNorm for the multi-scale readout, one per scale, applied BEFORE the taps are
        # concatenated. The taps have very unequal widths (trunk typically 8-16x the seam), so a
        # single norm over the concatenation would draw its statistics almost entirely from the
        # widest tap. Left as None when multiscale is off, which keeps it untracked and leaves every
        # existing checkpoint lineage's object graph unchanged.
        if map_pool_multiscale:
            n_taps = (
                1 + count_scale_taps(map_encoder.gcnn_post) if map_encoder is not None else count_scale_taps(self.gcnn)
            )
            self.scale_norms = [
                tf.keras.layers.LayerNormalization(axis=-1, name=f"scale_norm_{i}") for i in range(n_taps)
            ]
        else:
            self.scale_norms = None

        # Separate LayerNorm per branch so the high-dimensional map features and the
        # compact Cls features are independently normalised before the embedding / concatenation.
        self.map_norm = tf.keras.layers.LayerNormalization(axis=-1, name="map_norm")
        self.cls_norm = tf.keras.layers.LayerNormalization(axis=-1, name="cls_norm")

        self.cls_embedding_layers = cls_embedding_layers
        self.regression_head_layers = regression_head_layers

        dense_widths = [layer.units for layer in cls_embedding_layers if hasattr(layer, "units")]
        cls_out_dim = dense_widths[-1] if dense_widths else self.cls_layer.n_cls_flat
        LOGGER.warning(
            f"ResNetMapsPlusCLSNetwork: map_pool={map_pool or 'None (flatten)'}, "
            f"map_pool_multiscale={map_pool_multiscale}"
            f"{f' ({len(self.scale_norms)} scale taps)' if map_pool_multiscale else ''}, map_feature_dim="
            f"{map_feature_dim if map_feature_dim is not None else 'None (raw flattened GCNN features)'}, "
            f"n_cls_bins={n_cls_bins}, n_z_cross={len(l_max_per_pair)}, "
            f"cls_flat_dim={self.cls_layer.n_cls_flat}, "
            f"cls_emb_dim={cls_out_dim} ({'embedding' if cls_embedding_layers else 'no embedding'})"
        )

    def call(self, inputs, training=False):
        """Forward pass.

        Args:
            inputs (tuple): ``(maps, cls)`` where
                - maps: float tensor ``(batch, n_pix, n_channels)``
                - cls:  float tensor ``(batch, n_ell, n_z_cross)``  (raw per-ell values)
            training (bool): Keras training flag.

        Returns:
            tf.Tensor: Summary statistics, shape ``(B, out_features)``.
        """
        maps, cls = inputs

        # Map branch: GCNN (or multi-res encoder) → pool/flatten → (project) → normalise
        map_branch = self.map_encoder if self.map_encoder is not None else self.gcnn
        if self.map_pool_multiscale:
            # Multi-scale readout: reduce at every resolution, not just the trunk. The multi-res
            # encoder assembles its own taps (it owns the seam, the finest point at which both probes
            # are present); the single-res branch is walked here.
            if self.map_encoder is not None:
                x, scale_taps = self.map_encoder(maps, training=training, return_taps=True)
            else:
                x, pool_taps = forward_with_pool_taps(self.gcnn, maps, training=training)
                scale_taps = assemble_scale_taps(pool_taps, x)
            # zip truncates silently, and a dropped tap is a smaller readout that still trains and
            # still scores -- exactly the kind of mislabelled result that is unrecoverable later.
            if len(scale_taps) != len(self.scale_norms):
                raise ValueError(
                    f"multi-scale readout got {len(scale_taps)} taps but {len(self.scale_norms)} norms were "
                    "built at construction; count_scale_taps disagrees with the forward pass"
                )
            # each scale normalised on its own before the concat -- see __init__ for why
            x_flat = tf.concat(
                [
                    norm(moment_pool(t, self.map_pool), training=training)
                    for norm, t in zip(self.scale_norms, scale_taps)
                ],
                axis=-1,
            )
        else:
            x = map_branch(maps, training=training)  # (batch, n_pix_reduced, n_ch)
            if self.map_pool is not None:
                # permutation-invariant readout mirroring the transformer's masked-mean token pool:
                # reduce over the (footprint-only) pixel axis instead of flattening + linear-crushing.
                x_flat = moment_pool(x, self.map_pool)  # (B, n_ch), (B, 2*n_ch) or (B, 4*n_ch)
            else:
                x_flat = tf.reshape(x, (tf.shape(x)[0], -1))  # (B, n_map_flat)
        if self.map_projection is not None:
            x_flat = self.map_projection(x_flat)  # (B, map_feature_dim); small n_ch->dim Dense when pooled
        x_flat = self.map_norm(x_flat, training=training)

        # Cls branch: per-pair bin + log transform → normalise → embed
        cls_flat = self.cls_layer(cls, training=training)  # (B, n_bins * n_z_cross)
        cls_flat = self.cls_norm(cls_flat, training=training)
        for layer in self.cls_embedding_layers:
            cls_flat = layer(cls_flat, training=training)  # (B, emb_width) after last Dense+LN

        # Concatenate and pass through the regression head
        x = tf.concat([x_flat, cls_flat], axis=-1)
        for layer in self.regression_head_layers:
            x = layer(x, training=training)
        return x
