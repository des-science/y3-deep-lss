# this exists for benchmark purposes    

import tensorflow as tf
from contextlib import nullcontext

from deep_lss.utils import distribute
from msfm import fiducial_pipeline
from msfm.utils import logger

LOGGER = logger.get_logger(__file__)

global_batch_size = 64
tfr_pattern = "/pscratch/sd/a/athomsen/DESY3/v2/fiducial/DESy3_fiducial_???.tfrecord"
target_params = ["Om", "s8"]
n_steps = 500
profile = False

strategy = distribute.get_strategy(True)
local_batch_size = distribute.get_local_batch_size(strategy, global_batch_size)

def dataset_fn(input_context):
        # dset = fiducial_pipeline.get_fiducial_dset(
        dset = fiducial_pipeline.get_fiducial_multi_noise_dset(
            tfr_pattern=tfr_pattern,
            params=target_params,
            local_batch_size=local_batch_size,
            n_noise=3,
            # relevant for performance
            is_cached=True,
            n_readers=32,
            n_prefetch=3,
            file_name_shuffle_buffer=128,
            examples_shuffle_buffer=10,
            # distribution
            input_context=input_context,
        )
        return dset

dist_dset = strategy.distribute_datasets_from_function(dataset_fn)
dist_iter = iter(dist_dset)

for step in LOGGER.progressbar(range(1, n_steps + 1), at_level="info", total=n_steps):
    with tf.profiler.experimental.Trace("step", step_num=step, _r=1) if profile else nullcontext():
        dv_batch, index_batch = next(dist_iter)

        # profile
        if profile and step == 190:
            LOGGER.info(f"Starting to profile")
            tf.profiler.experimental.start("/pscratch/sd/a/athomsen/benchmarks")
        if profile and step == 195:
            LOGGER.info(f"Stopping to profile")
            tf.profiler.experimental.stop()

