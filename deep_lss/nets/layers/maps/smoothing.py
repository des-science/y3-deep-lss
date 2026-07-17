# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created July 2026
Author: Arne Thomsen

HEALPix Gaussian smoothing for the map branches.

This module vendors ``HealpySmoothing`` (and its two ``deepsphere.utils`` helpers,
``split_sparse_dense_matmul`` and ``GaussianNoiseLayer``) from ``deepsphere-cosmo-tf2`` so the
smoothing front-end no longer has to be imported from deepsphere. The copy is faithful to
the deepsphere original (same sparse-kernel construction, per-channel repetition scheme, and
white-noise augmentation); only the imports changed. The transformer encoders and
``PerProbeSmoothing`` — and therefore both multi-resolution encoders, including the GCNN
``ResNetMultiResEncoder`` — use this vendored copy; the single-resolution GCNN and legacy layer
specs (``ResNetLayers``, vit, one_d_conv) still build ``deepsphere.healpy_layers.HealpySmoothing``
directly. Keep the two in sync (the diff must stay imports-only).

``PerProbeSmoothing`` builds on top of it: with a single ``HealpySmoothing`` front-end, all
channels share one base kernel at the smallest requested FWHM, so the strongly smoothed clustering
channels need ``ceil((fwhm / fwhm_min)^2)`` (~O(100)) sparse matmuls at the full map nside.
``PerProbeSmoothing`` instead gives each probe its own kernel at its own nside: probes below the
output nside are downsampled in-network (the identical ``tf.math.unsorted_segment_mean`` the msfm
pipeline uses for ``downsample_nside``, which there runs as the last map op — so the result is the
same), then smoothed and noise-augmented at the coarse nside with the existing per-channel
repetition scheme.

``PerProbeSmoothing.call`` returns one tensor per probe, each at its OWN nside (finest probe at the
output nside, coarser probes at their coarse nside) — it does NOT upsample coarse probes back to a
common resolution. The multi-resolution transformer encoder consumes these separately, feeding the
coarse probe into the hierarchy at the level that already runs at its nside
(``HealpixMultiResMapEncoder``), so clustering is never upsampled.
"""

import os
from typing import Optional, Union

import numpy as np
import healpy as hp
import tensorflow as tf
from sklearn.neighbors import BallTree
from tqdm import tqdm

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


@tf.function
def split_sparse_dense_matmul(sparse_tensor, dense_tensor, n_splits=1):
    """
    Splits axis 1 of the dense_tensor such that tensorflow can handle the size of the computation.
    :param sparse_tensor: Input sparse tensor of rank 2.
    :param dense_tensor: Input dense tensor of rank 2.
    :param n_splits: Integer number of splits applied to axis 1 of dense_tensor.

    For reference, the error message to be avoided is:

    'Cannot use GPU when output.shape[1] * nnz(a) > 2^31 [Op:SparseTensorDenseMatMul]

    Call arguments received by layer "chebyshev" (type Chebyshev):
    • input_tensor=tf.Tensor(shape=(208, 7264, 128), dtype=float32)
    • training=False'
    """
    if n_splits > 1:
        print(
            f"Tracing... Due to tensor size, tf.sparse.sparse_dense_matmul is executed over {n_splits} splits."
            f" Beware of the resulting performance penalty."
        )
        dense_splits = tf.split(dense_tensor, n_splits, axis=1)
        result = []
        for dense_split in dense_splits:
            result.append(tf.sparse.sparse_dense_matmul(sparse_tensor, dense_split))
        result = tf.concat(result, axis=1)
    else:
        result = tf.sparse.sparse_dense_matmul(sparse_tensor, dense_tensor)

    return result


class GaussianNoiseLayer(tf.keras.layers.Layer):
    """
    A layer that adds Gaussian noise to the input, where the standard deviation of the Gaussian can be set channel-wise
    """

    def __init__(self, stddev, **kwargs):
        super(GaussianNoiseLayer, self).__init__(**kwargs)
        self.stddev = tf.convert_to_tensor(stddev, dtype=tf.float32)

    def build(self, input_shape):
        if len(self.stddev.shape) == 0:
            self.stddev = tf.ones((input_shape[-1],)) * self.stddev
        elif self.stddev.shape[0] != input_shape[-1]:
            raise ValueError("Length of stddev does not match the number of input channels")

    def call(self, inputs):
        noise = tf.random.normal(
            shape=tf.shape(inputs),
            mean=0.0,
            stddev=tf.cast(self.stddev, inputs.dtype),
            dtype=inputs.dtype,
        )

        return inputs + noise


class HealpySmoothing(tf.keras.Model):
    """
    A layer that smoothes a Healpix map with a Gaussian kernel.

    Vendored from ``deepsphere.healpy_layers.HealpySmoothing`` (see the module docstring).
    """

    def __init__(
        self,
        # pixels
        nside: int,
        indices: np.ndarray,
        nest: bool = True,
        mask: Optional[tf.Tensor] = None,
        # smoothing
        fwhm: Optional[Union[int, float, list]] = None,
        fwhm_base: Optional[Union[int, float]] = None,
        sigma: Optional[Union[int, float, list]] = None,
        n_sigma_support: Union[int, float] = 3,
        arcmin: bool = True,
        per_channel_repetitions: Optional[Union[list, np.ndarray]] = None,
        white_noise_sigma: Optional[Union[int, float, list]] = None,
        # computational
        data_path: Optional[str] = None,
        max_batch_size: Optional[int] = None,
    ) -> None:
        """
        Initialize the sparse kernel tensor with which the maps are smoothed.
        Note that the smoothing is always done with a single base sigma. When different smoothing scales are specified
        for the different input channels, that kernel is applied repeatedly to channels which require a larger
        smoothing scale, by exploiting the fact that the convolution of two Gaussians with standard deviations sigma_1
        and sigma_2 is a Gaussian with sigma_3 = sqrt(sigma_1^2 + sigma_2^2). This implementation saves GPU memory, as
        the sparse kernel matrix can grow to be very large.
        :param nside: The healpy nside of the input.
        :param indices: 1d array of indices, corresponding to the pixel ids of the input map footprint.
        :param nest: Whether the maps are stored in healpix NEST ordering. Defaults to True, which is
                     always the case for DeepSphere networks.
        :param mask: Boolean tensor of shape (n_indices, 1) or (n_indices, n_channels)
                     that indicates which part of the patch defined by the indices is actually populated. Defaults to
                     None, then no additional masking is applied and the maps bleed into the zero padding.
        :param fwhm: FWHM of the Gaussian smoothing kernel. Can be either a single or per channel number. In the latter
                     case, the smoothing scale of the kernel is chosen as the smallest value and the rest achieved by
                     smoothing repeatedly. Defaults to None, then sigma needs to be specified.
        :param fwhm_base: Optional base kernel FWHM used together with a per channel fwhm list. The kernel is built at
                          this scale and applied ceil((fwhm / fwhm_base)^2) times per channel, instead of deriving the
                          base from min(fwhm). A value below min(fwhm) reduces the overshoot introduced by the ceil,
                          at the cost of more repetitions. Must satisfy fwhm_base <= min(fwhm). Defaults to None, then
                          min(fwhm) is used as before.
        :param sigma: Identical functionality as the fwhm argument, but specifies the standard deviation of the
                      Gaussian smoothing kernel instead. Defaults to None, then fwhm needs to be specified.
        :param n_sigma_support: Determines the radius from which the smoothing is calculated. Specifically, this value
                                determines which nearest neighbors are included. Defaults to 3, then roughly 99.7% of
                                the Gaussian probability mass is accounted for.
        :param arcmin: Whether fwhm and sigma are specified in arcmin or radian. Defaults to True.
        :param per_channel_repetitions: When a single value is specified for fwhm or sigma, this argument determines
                                        the per channel number of times the smoothing kernel is applied. Defaults to
                                        None.
        :param white_noise_sigma: Standard deviation of the white noise to add to the smoothed map. This is done to
                                  destroy information above some l_max, which has to be chosen according to the fwhm
                                  and the map type under consideration.
        :param data_path: Path where the sparse kernel tensor is stored to, and if available, loaded from. Defaults to
                          None, then the sparse kernel tensor is neither saved nor loaded.
        :param max_batch_size: Maximal batch size this network is supposed to handle. This determines the number of
                               splits in the tf.sparse.sparse_dense_matmul operation, which are subsequently applied
                               independent of the actual batch size. Defaults to None, then an attempt is made to infer
                               this from the input, which may cause an error.
        """
        super(HealpySmoothing, self).__init__()

        # pixels
        self.nside = nside
        self.indices = indices
        self.nest = nest
        self.mask = mask

        # smoothing
        assert fwhm is not None or sigma is not None, f"One of fwhm and sigma has to be specified"
        assert fwhm is None or sigma is None, f"Only one of fwhm and sigma can be specified"
        assert fwhm_base is None or isinstance(
            fwhm, (list, np.ndarray)
        ), f"fwhm_base requires a per channel fwhm list"

        self.fwhm = fwhm
        self.fwhm_base = fwhm_base
        self.sigma = sigma
        self.n_sigma_support = n_sigma_support
        self.arcmin = arcmin
        self.per_channel_repetitions = per_channel_repetitions
        self.white_noise_sigma = white_noise_sigma
        self.data_path = data_path
        self.max_batch_size = max_batch_size
        self.layer_compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype

        if self.fwhm == 0.0 or self.sigma == 0.0:
            self.do_smoothing = False
            print(f"The layer implements the identity, smoothing is disabled")
        else:
            self.do_smoothing = True

            if isinstance(self.fwhm, (list, np.ndarray)):
                assert (
                    self.per_channel_repetitions is None
                ), f"per_channel_repetitions can't be specified when fwhm is a list, since it is then inferred"

                self.fwhm = np.array(self.fwhm)

                # smallest smoothing scale from which the others are derived by looping, unless an
                # explicit (finer) base kernel scale is requested
                if self.fwhm_base is not None:
                    assert self.fwhm_base <= np.min(self.fwhm), (
                        f"fwhm_base ({self.fwhm_base}) must not exceed the smallest per channel fwhm "
                        f"({np.min(self.fwhm)})"
                    )
                    fwhm_min = self.fwhm_base
                else:
                    fwhm_min = np.min(self.fwhm)

                # ceil to be conservative, square because Gaussian variances are added (not stds)
                self.per_channel_repetitions = np.ceil((self.fwhm / fwhm_min) ** 2).astype(int)
                self.fwhm = fwhm_min

            elif isinstance(self.sigma, (list, np.ndarray)):
                assert (
                    self.per_channel_repetitions is None
                ), f"per_channel_repetitions can't be specified when sigma is a list, since it is then inferred"

                self.sigma = np.array(self.sigma)
                sigma_min = np.min(self.sigma)
                self.per_channel_repetitions = np.ceil((self.sigma / sigma_min) ** 2).astype(int)
                self.sigma = sigma_min

            elif isinstance(self.per_channel_repetitions, list):
                self.per_channel_repetitions = np.array(self.per_channel_repetitions)

            # internally, the smoothing is always done with sigma
            if self.sigma is None:
                self.sigma = self.fwhm / np.sqrt(8 * np.log(2))

            # angle conversions
            if self.arcmin:
                self.sigma_arcmin = self.sigma
                self.sigma_rad = self._arcmin_to_rad(self.sigma_arcmin)
            else:
                self.sigma_rad = self.sigma
                self.sigma_arcmin = self._rad_to_arcmin(self.sigma_rad)

            self.fwhm_arcmin = self.sigma_arcmin * np.sqrt(8 * np.log(2))

            # derived attributes
            self.n_indices = len(indices)
            self.kernel_func = lambda r: np.exp(-0.5 / self.sigma_rad**2 * r**2)
            with np.printoptions(precision=2):
                self.file_label = f"-nside{self.nside}-sigma{self.sigma_arcmin:4.2f}-n_sigma{n_sigma_support}"

                if self.per_channel_repetitions is not None:
                    per_channel_factor = np.sqrt(self.per_channel_repetitions)
                    print(f"Using the per channel smoothing repetitions {self.per_channel_repetitions}")
                    print(
                        f"Using the per channel smoothing scales "
                        f"sigma = {per_channel_factor * self.sigma_arcmin} arcmin, "
                        f"fwhm = {per_channel_factor * self.fwhm_arcmin} arcmin"
                    )
                else:
                    print(
                        f"Using the per channel smoothing scale sigma = {self.sigma_arcmin:4.2f} arcmin, "
                        f" fwhm = {self.fwhm_arcmin:4.2f} arcmin"
                    )

                if self.data_path is not None:
                    try:
                        self.ind_coo = np.load(os.path.join(self.data_path, f"ind_coo{self.file_label}.npy"))
                        self.val_coo = np.load(os.path.join(self.data_path, f"val_coo{self.file_label}.npy"))
                        print(f"Successfully loaded sparse kernel indices and values from {self.data_path}")
                    except FileNotFoundError:
                        self._build_tree()
                        self._build_kernel()
                else:
                    self._build_tree()
                    self._build_kernel()

            self._build_sparse_tensor()
            print(f"Successfully created the sparse kernel tensor")

        # white noise
        if self.white_noise_sigma is not None:
            print(f"Adding white noise with sigma {self.white_noise_sigma} to the smoothed map")
            self.white_noise_layer = GaussianNoiseLayer(self.white_noise_sigma)

            if mask is None:
                print(
                    f"Warning, you're adding white noise to the maps but haven't provided a mask! The noise will "
                    f"extend to the padding"
                )
        else:
            self.white_noise_layer = None

    def build(self, input_shape: tuple) -> None:
        """
        Checks whether the input shape is compatible with the initialized layer. Note that the sparse-dense matrix
        multiplication might be split into multiple operations, depending on the nonzero entries in the sparse kernel
        matrix and batch dimension.
        :param input_shape: Shape of the input, which is expected to be (n_batch, n_indices, n_channels).
        """
        if self.do_smoothing:
            # batch dimension
            if self.max_batch_size is not None:
                self.n_batch = self.max_batch_size
            elif input_shape[0] is not None:
                self.n_batch = input_shape[0]
            else:
                self.n_batch = None
                print(
                    f"Since the batch size cannot be inferred from the input shape and max_batch_size is not "
                    f"available, no sparse-dense matmul splits are performed, which may cause an error."
                )

            # map dimensions
            assert self.n_indices == input_shape[1]
            self.n_channels = input_shape[2]

            if self.per_channel_repetitions is not None:
                assert (
                    len(self.per_channel_repetitions) == self.n_channels
                ), f"The list per_channel_repetitions has to have length {self.n_channels}"

                assert (
                    self.per_channel_repetitions.dtype == int
                ), f"The list per_channel_repetitions has to contain integers only"

            if self.mask is not None:
                self.mask = tf.cast(self.mask, dtype=self.layer_compute_dtype)
                if tf.rank(self.mask).numpy() == 1:
                    self.mask = tf.expand_dims(self.mask, axis=0)
                    self.mask = tf.expand_dims(self.mask, axis=-1)
                elif tf.rank(self.mask).numpy() == 2:
                    self.mask = tf.expand_dims(self.mask, axis=0)

                assert (
                    self.mask.shape[1] == self.n_indices
                ), f"The mask has to have shape (1, n_indices, 1) or (1, n_indices, n_channels)"

            self.n_matmul_splits = 1
            # check if we need to split the matmul
            if self.n_batch is not None:
                while not (
                    # tf.split only does even splits for integer arguments
                    (self.n_batch % self.n_matmul_splits == 0)
                    and
                    # due to the int32 limitation of tf.sparse.sparse_dense_matmul
                    (self.n_matmul_splits >= self.n_batch * len(self.sparse_kernel.indices) / 2**31)
                ):
                    self.n_matmul_splits += 1

            if self.white_noise_layer is not None:
                self.white_noise_layer.build(input_shape)

            print(f"Successfully built the smoothing layer")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Calls the layer on the input tensor.
        :param inputs: Tensor of shape (n_batch, n_indices, n_channels).
        :return: Smoothed output tensor of identical shape as input.
        """
        if self.do_smoothing:
            # (n_indices, n_batch, n_channels)
            indices_first = tf.transpose(inputs, (1, 0, 2))

            # list of (n_indices, n_batch)
            separate_channels = tf.unstack(indices_first, axis=2)

            stack = []
            for i, single_channel in enumerate(separate_channels):
                if self.per_channel_repetitions is not None:
                    for _ in range(self.per_channel_repetitions[i]):
                        single_channel = split_sparse_dense_matmul(
                            self.sparse_kernel, single_channel, self.n_matmul_splits
                        )
                else:
                    single_channel = split_sparse_dense_matmul(
                        self.sparse_kernel, single_channel, self.n_matmul_splits
                    )

                stack.append(single_channel)

            # (n_indices, n_batch, n_channels)
            channels_last = tf.stack(stack, axis=2)

            # (n_batch, n_indices, n_channels)
            channels_last = tf.transpose(channels_last, (1, 0, 2))

            if self.white_noise_layer is not None:
                channels_last = self.white_noise_layer(channels_last)

            if self.mask is not None:
                channels_last *= self.mask

            return channels_last

        else:
            return inputs

    def _build_tree(self) -> None:
        """
        Builds a BallTree to find the nearest neighbors of each pixel. The number of neighbors is determined by the
        radius n_sigma_support * sigma. The maximum number of neighbors is determined by the pixel with the most
        neighbors within that radius. The Gaussian smoothing kernel is evaluated at the distances to the neighbors.
        """
        print(
            f"Creating tree for {self.n_indices} pixels and radius n_sigma_support * sigma = "
            f"{self.sigma_arcmin * self.n_sigma_support:4.2f} arcmin"
        )

        lon, lat = hp.pix2ang(self.nside, ipix=self.indices, nest=self.nest, lonlat=True)
        theta = np.stack([np.radians(lat), np.radians(lon)], axis=1)

        tree = BallTree(theta, metric="haversine")

        # determine the maximum number of neighbors
        inds_r = tree.query_radius(theta, r=self.sigma_rad * self.n_sigma_support)
        n_neighbours = [len(i) for i in inds_r]
        self.max_neighbors = np.max(n_neighbours)
        print(f"The maximal number of neighbors within that radius is {self.max_neighbors}")

        # find the per pixel k nearest neighbors
        n_theta_splits = 100
        theta_split = np.array_split(theta, n_theta_splits)
        list_dist_k, list_inds_k = [], []
        for theta_ in tqdm(theta_split, total=n_theta_splits, desc="querying the tree"):
            dist_k, inds_k = tree.query(theta_, k=self.max_neighbors, return_distance=True, sort_results=True)
            list_dist_k.append(dist_k)
            list_inds_k.append(inds_k)

        dist_k = np.concatenate(list_dist_k, axis=0)
        self.inds_k = np.concatenate(list_inds_k, axis=0, dtype=np.int64)
        self.kernel_k = self.kernel_func(dist_k).astype(np.float32)

    def _build_kernel(self) -> None:
        """
        Builds the indices and values of the coo sparse kernel matrix as dense arrays, which may be stored to disk.
        """
        # row, all of the pixels in the patch
        inds_r = tf.constant(np.arange(self.n_indices), dtype=tf.int64)
        inds_r = tf.expand_dims(inds_r, axis=-1)
        inds_r = tf.repeat(inds_r, self.max_neighbors, axis=1)

        # column, all of the pixels that we want to sum over
        inds_c = tf.constant(self.inds_k, dtype=tf.int64)

        # shape (n_non_zero, 2)
        self.ind_coo = tf.concat([tf.reshape(inds_r, (-1, 1)), tf.reshape(inds_c, (-1, 1))], axis=1)

        # shape(n_non_zero,)
        self.val_coo = tf.reshape(self.kernel_k, (-1,))

        if self.data_path is not None:
            np_ind_coo = self.ind_coo.numpy()
            np_val_coo = self.val_coo.numpy()
            print(
                f"Storing sparse kernel indices ({np_ind_coo.nbytes/1e9:4.2f} GB, dtype {np_ind_coo.dtype}) and "
                f"values ({np_val_coo.nbytes/1e9:4.2f} GB, dtype {np_val_coo.dtype})"
            )

            os.makedirs(self.data_path, exist_ok=True)
            np.save(os.path.join(self.data_path, f"ind_coo{self.file_label}.npy"), np_ind_coo)
            np.save(os.path.join(self.data_path, f"val_coo{self.file_label}.npy"), np_val_coo)

    def _build_sparse_tensor(self) -> None:
        """Builds the tf.sparse.SparseTensor from the dense indices and values."""
        self.val_coo = tf.cast(self.val_coo, dtype=self.layer_compute_dtype)

        self.sparse_kernel = tf.sparse.SparseTensor(
            indices=self.ind_coo,
            values=self.val_coo,
            dense_shape=(self.n_indices, self.n_indices),
        )
        self.sparse_kernel = tf.sparse.reorder(self.sparse_kernel)

        # the kernel entries within rows have to sum to one
        col_sum = tf.sparse.reduce_sum(self.sparse_kernel, axis=1, output_is_sparse=False)
        self.sparse_kernel = self.sparse_kernel / tf.expand_dims(col_sum, axis=0)

        del self.ind_coo
        del self.val_coo

    @staticmethod
    def _rad_to_arcmin(theta):
        return theta / np.pi * (180 * 60)

    @staticmethod
    def _arcmin_to_rad(theta):
        return theta * np.pi / (60 * 180)


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
    """

    def __init__(self, probe_specs):
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
            self.smoothing_layers.append(HealpySmoothing(**kwargs))
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
