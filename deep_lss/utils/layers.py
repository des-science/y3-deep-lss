# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created November 2023
Author: Arne Thomsen
"""

import os

import numpy as np
import healpy as hp
import tensorflow as tf

from sklearn.neighbors import BallTree
from typing import Union, Optional

from msfm.utils import logger, scales

LOGGER = logger.get_logger(__file__)


class HealpixSmoothingLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        # pixels
        n_side: int,
        indices: np.ndarray,
        nest: bool = True,
        # smoothing
        fwhm: Optional[Union[int, float]] = None,
        sigma: Optional[Union[int, float]] = None,
        n_sigma_support: Union[int, float] = 3,
        arcmin: bool = True,
        per_channel_repetitions: Optional[Union[list, np.ndarray]] = None,
        # computational
        data_path: Optional[str] = None,
    ):
        super(HealpixSmoothingLayer, self).__init__()

        # pixels
        self.n_side = n_side
        self.indices = indices
        self.nest = nest

        # smoothing
        assert fwhm is not None or sigma is not None, f"One of fwhm and sigma has to be specified"
        assert fwhm is None or sigma is None, f"Only one of fwhm and sigma can be specified"

        self.fwhm = fwhm
        self.sigma = sigma
        self.n_sigma_support = n_sigma_support
        self.arcmin = arcmin
        self.per_channel_repetitions = per_channel_repetitions

        # computational
        self.data_path = data_path

        if self.fwhm == 0.0 or self.sigma == 0.0:
            self.do_smoothing = False
            LOGGER.warning(f"Smoothing is disabled")
        else:
            self.do_smoothing = True

            # internally, we only use sigma
            if sigma is None:
                sigma = fwhm / np.sqrt(8 * np.log(2))

            if arcmin:
                self.sigma_arcmin = self.sigma
                self.sigma_rad = scales.arcmin_to_rad(self.sigma_arcmin)
            else:
                self.sigma_rad = self.sigma
                self.sigma_arcmin = scales.rad_to_arcmin(self.sigma_rad)

            # derived attributes
            self.n_indices = len(indices)
            self.kernel_func = lambda r: np.exp(-0.5 / self.sigma_rad**2 * r**2)

            self.file_label = f"-n_side{self.n_side}-sigma{self.sigma_arcmin}-n_sigma{n_sigma_support}"

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

    def build(self, input_shape):
        if self.do_smoothing:
            # check shapes
            self.n_batch = input_shape[0]
            assert self.n_indices == input_shape[1]
            self.n_channels = input_shape[2]

            if self.per_channel_repetitions is not None:
                assert (
                    len(self.per_channel_repetitions) == self.n_channels
                ), f"The list per_channel_repetitions has to have length {self.n_channels}"
                assert all(
                    [isinstance(item, int) for item in self.per_channel_repetitions]
                ), f"The list per_channel_repetitions has to contain integers only"

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

    def call(self, inputs):
        if self.do_smoothing:
            # (batch, nodes, channels)

            channels_first = tf.transpose(inputs, (2, 1, 0))

            channels_smoothed = []
            for i, single_channel in enumerate(channels_first):
                if self.per_channel_repetitions is not None:
                    for _ in range(self.per_channel_repetitions[i]):
                        single_channel = self._split_sparse_dense_matmul(self.sparse_kernel, single_channel)

                else:
                    single_channel = self._split_sparse_dense_matmul(self.sparse_kernel, single_channel)

                channels_smoothed.append(single_channel)

            channels_first = tf.stack(channels_smoothed, axis=0)
            channels_last = tf.transpose(channels_first, (2, 1, 0))

            return channels_last

        else:
            return inputs

    def _build_tree(self):
        LOGGER.info(
            f"Creating tree for {self.n_indices} pixels and radius {self.sigma_arcmin * self.n_sigma_support} arcmin"
        )

        lon, lat = hp.pix2ang(self.n_side, ipix=self.indices, nest=self.nest, lonlat=True)
        theta = np.stack([np.radians(lat), np.radians(lon)], axis=1)

        tree = BallTree(theta, metric="haversine")

        # determine the maximum number of neighbors
        inds_r, dist_r = tree.query_radius(
            theta,
            r=self.sigma_rad * self.n_sigma_support,
            return_distance=True,
            sort_results=True,
        )
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

        np_ind_coo = self.ind_coo.numpy()
        np_val_coo = self.val_coo.numpy()
        LOGGER.info(
            f"Storing sparse kernel indices ({np_ind_coo.nbytes/1e9:4.2f} GB, dtype {np_ind_coo.dtype}) and "
            f"values ({np_val_coo.nbytes/1e9:4.2f} GB, dtype {np_val_coo.dtype})"
        )
        np.save(os.path.join(self.data_path, f"ind_coo{self.file_label}.npy"), np_ind_coo)
        np.save(os.path.join(self.data_path, f"val_coo{self.file_label}.npy"), np_val_coo)

    def _build_sparse_tensor(self):
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
