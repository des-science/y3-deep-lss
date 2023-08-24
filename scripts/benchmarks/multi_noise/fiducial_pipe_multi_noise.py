# this exists for benchmark purposes

import tensorflow as tf
import psutil, os

from contextlib import nullcontext
from datetime import datetime
from time import time

from deep_lss.utils import distribute
from msfm.fiducial_pipeline import FiducialPipeline
from msfm.utils import logger

LOGGER = logger.get_logger(__file__)

tfr_pattern = "/pscratch/sd/a/athomsen/DESY3/v3/fiducial/DESy3_fiducial_???.tfrecord"
log_file = f"/pscratch/sd/a/athomsen/run_files/benchmarks/multi_noise/new/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
global_batch_size = 16
n_steps = 300
n_gpus = len(tf.config.list_physical_devices("GPU"))
profile = True
multi_noise = False

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

dset_kwargs = {
    "tfr_pattern": tfr_pattern,
    "local_batch_size": local_batch_size,
    "is_cached": False,
    "n_readers": 16,
    # "n_prefetch": 0,
    "n_prefetch": 5,
    # "is_eval": False,
    "file_name_shuffle_buffer": 16,
    "examples_shuffle_buffer": None,
}

LOGGER.info(dset_kwargs)


def dataset_fn(input_context):
    if multi_noise:
        LOGGER.warning("multi noise")
        dset = fiducial_pipeline.get_multi_noise_dset(
            n_noise=3,
            **dset_kwargs,
            input_context=input_context,
        )

    else:
        LOGGER.warning("single noise")
        dset = fiducial_pipeline.get_dset(
            i_noise=0,
            **dset_kwargs,
            input_context=input_context,
        )

    return dset


dist_dset = strategy.distribute_datasets_from_function(dataset_fn)
dist_iter = iter(dist_dset)

summary_writer = tf.summary.create_file_writer(log_file)

t_prev = time()
for step in LOGGER.progressbar(range(n_steps), at_level="info", total=n_steps):
    with tf.profiler.experimental.Trace("step", step_num=step, _r=1) if profile else nullcontext():
        _, _ = next(dist_iter)

        # profile
        if profile and step == 200:
            LOGGER.info(f"Starting to profile")
            tf.profiler.experimental.start(os.path.join(log_file, "profiler"))
        if profile and step == 205:
            LOGGER.info(f"Stopping to profile")
            tf.profiler.experimental.stop()

        with summary_writer.as_default():
            t_now = time()
            tf.summary.scalar("step_time", t_now - t_prev, step=step)
            t_prev = t_now

            if step % 10 == 0:
                for i in range(n_gpus):
                    # GPU, in per cent
                    mem_info = tf.config.experimental.get_memory_info(f"/GPU:{i}")
                    tf.summary.scalar(f"GPU_{i}_mem", mem_info["current"] / (4 * 10**8), step=step)
                    # tf.summary.scalar(f"GPU_{i}_mem_peak", mem_info["peak"]/(4*10**8), step=step)

                    # CPU, in per cent
                    tf.summary.scalar(f"CPU_mem", psutil.virtual_memory().percent, step=step)

        step += 1

LOGGER.warning(f"terminated successfully")
