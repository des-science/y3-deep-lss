import tensorflow as tf
import os, argparse, warnings, yaml

from datetime import datetime
from time import time
from contextlib import nullcontext

from msfm.fiducial_pipeline import FiducialPipeline
from msfm.grid_pipeline import GridPipeline
from msfm.utils import logger, input_output, files, parameters, tfrecords

from deep_lss.utils import distribute

LOGGER = logger.get_logger(__file__)

strategy = distribute.get_strategy("mirrored")
# strategy = distribute.get_strategy("horovod")
# strategy = distribute.get_strategy("multi_worker_mirrored")

# tfr_pattern = "/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/fiducial/DESy3_fiducial_*.tfrecord"
tfr_pattern = "/pscratch/sd/a/athomsen/DESY3/v6/linear_bias/tfrecords/grid/DESy3_grid_*.tfrecord"

pipe = GridPipeline(
    # pipe = FiducialPipeline(
    conf="/global/homes/a/athomsen/multiprobe-simulation-forward-model/configs/v6/linear_bias.yaml",
    with_lensing=True,
    with_clustering=True,
    apply_norm=False,
    params=["Om", "s8", "w0", "Aia", "n_Aia"],
)


# def dataset_fn(input_context):
#     ic(input_context.num_input_pipelines)
#     ic(input_context.input_pipeline_id)

#     dset = pipe.get_dset(
#         tfr_pattern=tfr_pattern,
#         # distribution
#         input_context=input_context,
#         local_batch_size=20,
#         n_prefetch=0,
#         n_readers=1,
#         file_name_shuffle_buffer=None,
#         examples_shuffle_buffer=None,
#         # n_readers=1,
#         # file_name_shuffle_buffer=16,
#         # examples_shuffle_buffer=64,
#     )

#     return dset

n_readers = 3
batch_size = 7

n_noise = 3


def dataset_fn(input_context):
    dset = tf.data.Dataset.list_files(tfr_pattern, shuffle=False)

    ic(input_context.num_input_pipelines)
    ic(input_context.input_pipeline_id)

    dset = dset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    # dset = dset.shard(4, input_context.input_pipeline_id)
    # dset = dset.shard(4, 0)
    # dset = dset.interleave(
    #     tf.data.TFRecordDataset,
    #     cycle_length=n_readers,
    #     num_parallel_calls=tf.data.AUTOTUNE,
    #     deterministic=True,
    # )
    # dset = dset.map(
    #     lambda serialized_example: tfrecords.parse_inverse_grid(
    #         serialized_example,
    #         n_noise,
    #         # dimensions
    #         pipe.n_dv_pix,
    #         pipe.n_z_metacal,
    #         pipe.n_z_maglim,
    #         pipe.n_all_params,
    #         # map types
    #         True,
    #         False,
    #     ),
    #     num_parallel_calls=tf.data.AUTOTUNE,
    # )
    # dset = dset.map(lambda element: element["i_sobol"])
    dset = dset.batch(batch_size)

    return dset


dist_dset = strategy.distribute_datasets_from_function(dataset_fn)
dist_iter = iter(dist_dset)

# breakpoint()

n_steps = 1
for step in LOGGER.progressbar(range(n_steps), at_level="info", total=n_steps):
    # dv_batch, index = next(dist_iter)
    # dv_batch, cosmo_batch, index = next(dist_iter)

    # print(index[0])

    element = next(dist_iter)
    print(element)
