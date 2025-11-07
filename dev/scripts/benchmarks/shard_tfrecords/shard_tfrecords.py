import tensorflow as tf
from deep_lss.utils import distribute
from msfm.fiducial_pipeline import FiducialPipeline

TFR_PATTERN = "/pscratch/sd/a/athomsen/DESY3/v5/linear_bias/tfrecords/fiducial/DESy3_fiducial_???.tfrecord"

if __name__ == "__main__":
    strategy = distribute.get_strategy(True)

    if isinstance(strategy, tf.distribute.MultiWorkerMirroredStrategy):
        task_id = strategy.cluster_resolver.task_id
        tf.print("task_id", task_id)
        
        n_gpus = len(tf.config.list_physical_devices("GPU"))
        tf.print("n_gpus", n_gpus)
    else:
        task_id = 0

    def dataset_fn(input_context):
        dset = tf.data.Dataset.list_files(TFR_PATTERN, shuffle=False)
        # dset = dset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)

        return dset

    dist_dset = strategy.distribute_datasets_from_function(dataset_fn)

    for batch in dist_dset:
        tf.print(task_id, batch)
