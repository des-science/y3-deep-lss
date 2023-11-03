import os

import numpy as np
import tensorflow as tf

from deep_lss.utils import distribute
from msfm.grid_pipeline import GridPipeline
from msfm.utils import logger

LOGGER = logger.get_logger(__file__)

tf.config.run_functions_eagerly(False)
tf.config.set_soft_device_placement(False)

# @tf.function
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
        # print(
        #     f"Tracing... Due to tensor size, tf.sparse.sparse_dense_matmul is executed over {n_splits} splits."
        #     f" Beware of the resulting performance penalty."
        # )
        dense_splits = tf.split(dense_tensor, n_splits, axis=1)
        result = []
        for dense_split in dense_splits:
            result.append(tf.sparse.sparse_dense_matmul(sparse_tensor, dense_split))
        result = tf.concat(result, axis=1)
    else:
        result = tf.sparse.sparse_dense_matmul(sparse_tensor, dense_tensor)

    return result


# dataset
tfr_pattern = "/pscratch/sd/a/athomsen/DESY3/v5/linear_bias/tfrecords/grid/DESy3_grid_???.tfrecord"
n_steps = 100
n_batch = 50

grid_pipeline = GridPipeline(
    with_lensing=True,
    with_clustering=True,
    apply_norm=True,
)

dset = grid_pipeline.get_dset(
    tfr_pattern=tfr_pattern,
    local_batch_size=n_batch,
    n_readers=16,
    n_prefetch=3,
)

for dv_batch, _, _ in dset.take(1):
    pass
n_data_vec_pix = dv_batch.shape[1]
n_channels = dv_batch.shape[2]

LOGGER.warning(f"Built the dataset")

# kernel
kernel_dir = "/pscratch/sd/a/athomsen/run_files/debug/smoothing_tree"
ind_coo = np.load(os.path.join(kernel_dir, "ind_coo.npy"))
val_kernel = np.load(os.path.join(kernel_dir, "val_kernel.npy"))

sparse_kernel = tf.sparse.SparseTensor(
    indices=ind_coo, values=val_kernel, dense_shape=(n_data_vec_pix, n_data_vec_pix)
)
sparse_kernel = tf.sparse.reorder(sparse_kernel)

op_size = len(sparse_kernel.indices) * n_batch
LOGGER.info(f"{op_size < 2**31}, {op_size}, {2**31}")

# normalize
col_sum = tf.sparse.reduce_sum(sparse_kernel, axis=1, output_is_sparse=False)
sparse_kernel = sparse_kernel / tf.expand_dims(col_sum, axis=0)
LOGGER.warning(f"Built the sparse kernel matrix")

perform_smoothing = False

for dv_batch, cosmo_batch, index_batch in LOGGER.progressbar(dset.take(n_steps), at_level="info", total=n_steps):
    if perform_smoothing:
        # loop over channels
        dv_smooth = []
        for i in range(n_channels):
            if i == 0:
                n_loops = 4
            else:
                n_loops = 1

            # select channel
            dv_batch_last = tf.transpose(dv_batch[..., i], (1, 0))

            for j in range(n_loops):
                dv_batch_last = tf.sparse.sparse_dense_matmul(sparse_kernel, dv_batch_last)
                # dv_batch_last = split_sparse_dense_matmul(sparse_kernel, dv_batch_last, 8)

            dv_batch_first = tf.transpose(dv_batch_last, (1, 0))
            dv_smooth.append(dv_batch_first)

        dv_smooth = tf.stack(dv_smooth, axis=-1)

