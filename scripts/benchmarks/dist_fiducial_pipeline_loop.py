# this exists for benchmark purposes

import tensorflow as tf
from contextlib import nullcontext

from deep_lss.utils import distribute
from msfm.fiducial_pipeline import FiducialPipeline
from msfm.utils import logger

LOGGER = logger.get_logger(__file__)

tfr_pattern = "/pscratch/sd/a/athomsen/DESY3/v3/fiducial/DESy3_fiducial_???.tfrecord"
n_steps = 200
global_batch_size = 16
profile = False

_, _ = distribute.check_devices()
strategy = distribute.get_strategy(True)
local_batch_size = distribute.get_local_batch_size(strategy, global_batch_size)

fiducial_pipeline = FiducialPipeline(
    # params=["Om", "s8"],
    # params=["Om", "s8", "bg", "n_bg"],
    # params=["Om", "s8", "Aia", "n_Aia"],
    params=["Om", "s8", "Aia", "n_Aia", "bg", "n_bg"],
    with_lensing=True,
    with_clustering=True,
    apply_norm=True,
)


# like https://www.tensorflow.org/tutorials/distribute/input#tfdistributestrategydistribute_datasets_from_function
def dataset_fn(input_context):
    dset = fiducial_pipeline.get_dset(
        # dset = fiducial_pipeline.get_multi_noise_dset(
        # n_noise=3,
        tfr_pattern=tfr_pattern,
        local_batch_size=local_batch_size,
        is_cached=False,
        n_readers=4,
        n_prefetch=None,
        file_name_shuffle_buffer=16,
        examples_shuffle_buffer=None,
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
