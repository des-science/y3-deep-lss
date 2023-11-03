# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created November 2023
Author: Arne Thomsen

Additional network layers.
"""

import os

import numpy as np
import healpy as hp
import tensorflow as tf

from sklearn.neighbors import BallTree
from typing import Union, Optional

from msfm.utils import logger, scales

LOGGER = logger.get_logger(__file__)
np.set_printoptions(precision=1)


class HealpySmoothingLayer(tf.keras.layers.Layer):
    """
    A layer that performs a Gaussian smoothing on a Healpix map.
    """

    def __init__(
        self,
        # pixels
        n_side: int,
        indices: np.ndarray,
        nest: bool = True,
        mask: Optional[tf.Tensor] = None,
        # smoothing
        fwhm: Optional[Union[int, float, list]] = None,
        sigma: Optional[Union[int, float, list]] = None,
        n_sigma_support: Union[int, float] = 3,
        arcmin: bool = True,
        per_channel_repetitions: Optional[Union[list, np.ndarray]] = None,
        # computational
        data_path: Optional[str] = None,
    ) -> None:
        """Create the sparse kernel tensor with which the maps are smoothed.

        Note that the smoothing is always done with a single base sigma. When different smoothing scales are specified
        for the different input channels,that kernel is applied repeatedly to channels which require a larger smoothing
        scale, by exploiting the fact that the convolution of two Gaussians with standard deviations sigma_1 and
        sigma_2 is a Gaussian with sigma_3 = sqrt(sigma_1^2 + sigma_2^2). This implementation saves GPU memory, as the
        sparse kernel matrix can grow to be very large.

        Args:
            n_side (int): The healpy n_side of the input.
            indices (np.ndarray): 1d array of indices, corresponding to the pixel ids of the input map footprint.
            nest (bool, optional): Whether the maps are stored in healpix NEST ordering. Defaults to True, which is
                always the case for DeepSphere networks.
            mask (Optional[tf.Tensor], optional): Boolean tensor of shape (n_indices, 1) or (n_indices, n_channels)
                that indicates which part of the patch defined by the indices is actually populated. Defaults to None,
                then no additional masking is applied and the maps bleed into the zero padding.
            fwhm (Optional[Union[int, float]], optional): FWHM of the Gaussian smoothing kernel. Can be either a single
                or per channel number. In the latter case, the smoothing scale of the kernel is chosen as the smallest
                value and the rest achieved by smoothing repeatedly. Defaults to None, then sigma needs to be
                specified.
            sigma (Optional[Union[int, float]], optional): Identical functionality as the fwhm argument, but specifies
                the standard deviation of the Gaussian smoothing kernel instead. Defaults to None, then fwhm needs to
                be specified.
            n_sigma_support (Union[int, float], optional): Determines the radius from which the smoothing is
                calculated. Specifically, this value determines which nearest neighbors are included. Defaults to 3,
                then roughly 99.7% of the Gaussian probability mass is accounted for.
            arcmin (bool, optional): Whether fwhm and sigma are specified in arcmin or radian. Defaults to True.
            per_channel_repetitions (Optional[Union[list, np.ndarray]], optional): When a single value is specified for
                fwhm or sigma, this argument determines the per channel number of times the smoothing kernel is
                applied. Defaults to None.
            data_path (Optional[str], optional): Path where the sparse kernel tensor is stored to, and if available,
                loaded from. Defaults to None, then the sparse kernel tensor is neither saved nor loaded.
        """
        super(HealpySmoothingLayer, self).__init__()

        # pixels
        self.n_side = n_side
        self.indices = indices
        self.nest = nest
        self.mask = mask

        # smoothing
        assert fwhm is not None or sigma is not None, f"One of fwhm and sigma has to be specified"
        assert fwhm is None or sigma is None, f"Only one of fwhm and sigma can be specified"

        self.fwhm = fwhm
        self.sigma = sigma
        self.n_sigma_support = n_sigma_support
        self.arcmin = arcmin
        self.per_channel_repetitions = per_channel_repetitions
        self.data_path = data_path

        if self.fwhm == 0.0 or self.sigma == 0.0:
            self.do_smoothing = False
            LOGGER.warning(f"The layer implements the identity, smoothing is disabled")
        else:
            self.do_smoothing = True

            if isinstance(self.fwhm, (list, np.ndarray)):
                assert (
                    self.per_channel_repetitions is None
                ), f"per_channel_repetitions can't be specified when fwhm is a list, since it is then inferred"

                self.fwhm = np.array(self.fwhm)

                # smallest smoothing scale from which the others are derived by looping
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

            # internally, the smoothing is always done with sigma
            if self.sigma is None:
                self.sigma = self.fwhm / np.sqrt(8 * np.log(2))

            # angle conversions
            if self.arcmin:
                self.sigma_arcmin = self.sigma
                self.sigma_rad = scales.arcmin_to_rad(self.sigma_arcmin)
            else:
                self.sigma_rad = self.sigma
                self.sigma_arcmin = scales.rad_to_arcmin(self.sigma_rad)

            self.fwhm_arcmin = self.sigma_arcmin * np.sqrt(8 * np.log(2))

            # derived attributes
            self.n_indices = len(indices)
            self.kernel_func = lambda r: np.exp(-0.5 / self.sigma_rad**2 * r**2)
            self.file_label = f"-n_side{self.n_side}-sigma{self.sigma_arcmin:4.2f}-n_sigma{n_sigma_support}"

            if self.per_channel_repetitions is not None:
                per_channel_factor = np.sqrt(self.per_channel_repetitions)
                LOGGER.info(f"Using the per channel repetitions {self.per_channel_repetitions}")
                LOGGER.info(
                    f"Using the per channel smoothing scales "
                    f"sigma = {per_channel_factor * self.sigma_arcmin} arcmin, "
                    f"fwhm = {per_channel_factor * self.fwhm_arcmin} arcmin"
                )
            else:
                LOGGER.info(
                    f"Using the per channel smoothing scales sigma = {self.sigma_arcmin:4.2f} arcmin, "
                    f" fwhm = {self.fwhm_arcmin:4.2f} arcmin"
                )

            if self.data_path is not None:
                try:
                    self.ind_coo = np.load(os.path.join(self.data_path, f"ind_coo{self.file_label}.npy"))
                    self.val_coo = np.load(os.path.join(self.data_path, f"val_coo{self.file_label}.npy"))
                    LOGGER.info(f"Successfully loaded sparse kernel indices and values from {self.data_path}")
                except FileNotFoundError:
                    self._build_tree()
                    self._build_kernel()
            else:
                self._build_tree()
                self._build_kernel()

            self._build_sparse_tensor()
            LOGGER.info(f"Successfully created the sparse kernel tensor")

    def build(self, input_shape: tuple):
        """Checks whether the input shape is compatible with the initialized layer. Note that the sparse-dense matrix
        multiplication might be split into multiple operations, depending on the nonzero entries in the sparse kernel
        matrix and batch dimension.

        Args:
            input_shape (tuple): Shape of the input, which is expected to be (n_batch, n_indices, n_channels).
        """
        if self.do_smoothing:
            # check shapes
            self.n_batch = input_shape[0]
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
                self.mask = tf.cast(self.mask, dtype=tf.float32)
                if tf.rank(self.mask).numpy() == 1:
                    self.mask = tf.expand_dims(self.mask, axis=0)
                    self.mask = tf.expand_dims(self.mask, axis=-1)
                elif tf.rank(self.mask).numpy() == 2:
                    self.mask = tf.expand_dims(self.mask, axis=0)

                assert (
                    self.mask.shape[1] == self.n_indices
                ), f"The mask has to have shape (1, n_indices, 1) or (1, n_indices, n_channels)"

            # check if we need to split the matmul
            self.n_matmul_splits = 1
            while not (
                # tf.split only does even splits for integer arguments
                (self.n_batch % self.n_matmul_splits == 0)
                and
                # due to the int32 limitation of tf.sparse.sparse_dense_matmul
                (self.n_matmul_splits >= self.n_batch * len(self.sparse_kernel.indices) / 2**31)
            ):
                self.n_matmul_splits += 1

            LOGGER.info(f"Successfully built the layer")

    def call(self, inputs: tf.Tensor):
        """Calls the layer on the input tensor.

        Args:
            inputs (tf.Tensor): Tensor of shape (n_batch, n_indices, n_channels).

        Returns:
            tf.Tensor: Smoothed output tensor of identical shape.
        """
        if self.do_smoothing:
            # (n_channels, n_indices, n_batch), for efficient sparse-dense matmul
            channels_first = tf.transpose(inputs, (2, 1, 0))

            channels_first_smoothed = []
            for i, single_channel in enumerate(channels_first):
                if self.per_channel_repetitions is not None:
                    for _ in range(self.per_channel_repetitions[i]):
                        single_channel = self._split_sparse_dense_matmul(self.sparse_kernel, single_channel)
                else:
                    single_channel = self._split_sparse_dense_matmul(self.sparse_kernel, single_channel)

                channels_first_smoothed.append(single_channel)

            channels_first_smoothed = tf.stack(channels_first_smoothed, axis=0)
            channels_last = tf.transpose(channels_first_smoothed, (2, 1, 0))

            if self.mask is not None:
                channels_last *= self.mask

            return channels_last

        else:
            return inputs

    def _build_tree(self):
        """Builds a BallTree to find the nearest neighbors of each pixel. The number of neighbors is determined by the
        radius n_sigma_support * sigma. The maximum number of neighbors is determined by the pixel with the most
        neighbors within that radius. The Gaussian smoothing kernel is evaluated at the distances to the neighbors.
        """
        LOGGER.info(
            f"Creating tree for {self.n_indices} pixels and radius n_sigma_support * sigma = "
            f"{self.sigma_arcmin * self.n_sigma_support:4.2f} arcmin"
        )

        lon, lat = hp.pix2ang(self.n_side, ipix=self.indices, nest=self.nest, lonlat=True)
        theta = np.stack([np.radians(lat), np.radians(lon)], axis=1)

        tree = BallTree(theta, metric="haversine")

        # determine the maximum number of neighbors
        inds_r = tree.query_radius(theta, r=self.sigma_rad * self.n_sigma_support)
        n_neighbours = [len(i) for i in inds_r]
        self.max_neighbors = np.max(n_neighbours)
        LOGGER.info(f"The maximal number of neighbors within that radius is {self.max_neighbors}")

        # find the per pixel k nearest neighbors
        n_theta_steps = 100
        theta_split = np.array_split(theta, n_theta_steps)
        list_dist_k, list_inds_k = [], []
        for theta_ in LOGGER.progressbar(theta_split, at_level="info", total=n_theta_steps, desc="querying the tree"):
            dist_k, inds_k = tree.query(theta_, k=self.max_neighbors, return_distance=True, sort_results=True)
            list_dist_k.append(dist_k)
            list_inds_k.append(inds_k)

        dist_k = np.concatenate(list_dist_k, axis=0)
        self.inds_k = np.concatenate(list_inds_k, axis=0, dtype=np.int64)
        self.kernel_k = self.kernel_func(dist_k).astype(np.float32)

    def _build_kernel(self):
        """Builds the indices and values of the coo sparse kernel matrix as dense tensors, which may be stored to disk."""
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
            LOGGER.info(
                f"Storing sparse kernel indices ({np_ind_coo.nbytes/1e9:4.2f} GB, dtype {np_ind_coo.dtype}) and "
                f"values ({np_val_coo.nbytes/1e9:4.2f} GB, dtype {np_val_coo.dtype})"
            )
            np.save(os.path.join(self.data_path, f"ind_coo{self.file_label}.npy"), np_ind_coo)
            np.save(os.path.join(self.data_path, f"val_coo{self.file_label}.npy"), np_val_coo)

    def _build_sparse_tensor(self):
        """Builds the tf.sparse.SparseTensor from the dense indices and values."""
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

    def _split_sparse_dense_matmul(self, sparse_tensor, dense_tensor):
        """Splits axis 1 of the dense_tensor such that TensorFlow can handle the size of the computation.

        For reference, the error message to be avoided is:

        'Cannot use GPU when output.shape[1] * nnz(a) > 2^31 [Op:SparseTensorDenseMatMul]

        Call arguments received by layer "chebyshev" (type Chebyshev):
        • input_tensor=tf.Tensor(shape=(208, 7264, 128), dtype=float32)
        • training=False'

        Args:
            sparse_tensor (tf.sparse.SparseTensor): Input sparse tensor of rank 2.
            dense_tensor (tf.Tensor): Input dense tensor of rank 2.

        Returns:
            tf.Tensor: A dense rank 2 tensor computed by the matrix product.
        """

        if self.n_matmul_splits > 1:
            LOGGER.info(
                f"Tracing... Due to tensor size, tf.sparse.sparse_dense_matmul is executed over {self.n_matmul_splits} "
                f"splits. Beware of the resulting performance penalty."
            )
            dense_splits = tf.split(dense_tensor, self.n_matmul_splits, axis=1)
            result = []
            for dense_split in dense_splits:
                result.append(tf.sparse.sparse_dense_matmul(sparse_tensor, dense_split))
            result = tf.concat(result, axis=1)
        else:
            result = tf.sparse.sparse_dense_matmul(sparse_tensor, dense_tensor)

        return result
