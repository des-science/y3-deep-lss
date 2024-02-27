# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created December 2022
Author: Arne Thomsen

Adapted from
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/base_model.py
by Janis Fluri, 
the main difference is that here, the distribution happens via tf.distribute.Strategy instead of horovod. Furthermore,
checkpointing is handled differently.
"""

import tensorflow as tf
import horovod.tensorflow as hvd
import os, warnings

from deepsphere import HealpyGCNN

from deep_lss.utils.distribute import HorovodStrategy
from msfm.utils import logger

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


class BaseModel(object):
    """
    This is a base model that provides a minimal training step and methods to restore and save the model.
    """

    def __init__(
        self,
        network,
        input_shape=None,
        optimizer=None,
        optimizer_kwargs={},
        summary_dir=None,
        checkpoint_dir=None,
        restore_checkpoint=False,
        max_checkpoints=3,
        init_step=0,
        strategy=None,
        # DeepSphere
        n_side=None,
        indices=None,
        n_neighbors=20,
        max_batch_size=None,
        initial_Fin=None,
    ):
        """Initializes a base model

        Args:
            network (Union[list, tf.keras.Sequential]): The underlying network of the model. Can be a list of layers,
                then either a regular tf.keras.Sequential or HealpyGCNN model is initialized.
            input_shape (tf.tensor, optional): Input shape of the network, necessary if one wants to restore the
                model. Defaults to None.
            optimizer (tf.keras.optimizers.Optimizer, optional): Optimizer of the model. Defaults to None.
            optimizer_kwargs (dict, optional): Additional keyword arguments passed to the optimizer. Defaults to {}.
            summary_dir (str, optional): Directory to save the summaries. Defaults to None.
            checkpoint_dir (str, optional): Directory where to save the weights and optimizer. Defaults to None.
            restore_checkpoint (boo, optional): Whether to restore the network from a checkpoint, or initialize it.
                Defaults to False.
            max_checkpoints (int, optional): The maximum number of checkpoints to keep. Older ones are automatically
                deleted by the CheckpointManager.
            init_step (int, optional): Initial step. Defaults to 0.
            strategy (Union[tf.distribute.Strategy, deep_lss.utils.distribute.HorovodStrategy], optional):
                The distribution strategy the model was created within. Defaults to None, then training is local.
            n_side (int): The healpy n_side of the input.
            indices (np.ndarray): 1d array of indices, corresponding to the pixel ids of the input map footprint.
            n_neighbors (int, optional): Number of neighbors considered when building the graph, currently supported
                values are: 8, 20, 40 and 60. Defaults to 20.
            max_batch_size (int, optional): Maximal batch size this network is supposed to handle. This determines the
                number of splits in the tf.sparse.sparse_dense_matmul operation, which are subsequently applied
                independent of the actual batch size. Defaults to None, then no such precautions are taken, which may
                cause an error.
            initial_Fin (int, optional) Initial number of input features. Defaults to None, then like for
                max_batch_size, there are no precautions taken.
        """

        # get the network
        if isinstance(network, list):
            if (n_side is None) and (indices is None):
                LOGGER.info("Initializing with a normal Sequential model")
                network = tf.keras.Sequential(layers=network)
            elif (n_side is not None) and (indices is not None):
                LOGGER.info("Initializing with a HealpyGCNN model")
                network = HealpyGCNN(
                    nside=n_side,
                    indices=indices,
                    layers=network,
                    n_neighbors=n_neighbors,
                    max_batch_size=max_batch_size,
                    initial_Fin=initial_Fin,
                )
            else:
                raise ValueError(f"n_side = {n_side} and indices = {indices} have to be both None or both not None")
        elif isinstance(network, tf.keras.Sequential):
            LOGGER.info("Initializing with a normal Sequential model")
        else:
            raise ValueError(f"Invalid network {network} was passed")

        # get the network
        self.network = network

        # save additional variables
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.summary_dir = summary_dir
        self.checkpoint_dir = checkpoint_dir
        self.restore_from_checkpoint = restore_checkpoint
        self.max_checkpoints = max_checkpoints
        self.init_step = init_step
        self.strategy = strategy

        # set up the optimizer
        if isinstance(self.optimizer, tf.keras.optimizers.Optimizer):
            pass
        elif self.optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(**optimizer_kwargs)
        elif self.optimizer == "adam":
            self.optimizer = tf.keras.optimizers.Adam(**optimizer_kwargs)
        elif self.optimizer == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(**optimizer_kwargs)
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer} is not implemented")

        # build the network
        if self.input_shape is not None:
            self.build_network(input_shape=self.input_shape)
            self.print_summary()

        # set the step
        self.train_step = tf.Variable(self.init_step, trainable=False, name="GlobalStep", dtype=tf.int64)
        tf.summary.experimental.set_step(self.train_step)

        # set up the checkpointing
        if self.checkpoint_dir is not None:
            if isinstance(self.strategy, (tf.distribute.MultiWorkerMirroredStrategy, HorovodStrategy)):
                if not self.is_chief():
                    self.checkpoint_dir = self.create_temp_dir(self.checkpoint_dir)

                    # copy over the existing checkpoints from the chief to the temporary directories
                    chief_dir = tf.io.gfile.join(self.checkpoint_dir, "..")
                    self.copy_chief_to_temp_dir(chief_dir, self.checkpoint_dir)
                    LOGGER.info(
                        f"Copied over the chief's checkpoints to the temporary directory {self.checkpoint_dir}"
                    )

            # always create the checkpoint directory
            tf.io.gfile.makedirs(self.checkpoint_dir)

            self.checkpoint = tf.train.Checkpoint(
                network=self.network, optimizer=self.optimizer, train_step=self.train_step
            )
            self.checkpoint_manager = tf.train.CheckpointManager(
                self.checkpoint,
                self.checkpoint_dir,
                max_to_keep=self.max_checkpoints,
                checkpoint_name="ckpt",
                step_counter=self.train_step,
            )
            self.n_init_checkpoints = len(self.checkpoint_manager.checkpoints)
        else:
            self.checkpoint = None
            self.checkpoint_manager = None
            self.n_init_checkpoints = None

        # restore model
        if self.restore_from_checkpoint:
            self.restore_model()
        elif (self.checkpoint_manager is not None) and (self.n_init_checkpoints != 0):
            LOGGER.warning(
                f"The model can not be saved when it is initialized from scratch with a non-empty checkpoint directory"
            )
        else:
            LOGGER.info(f"The network is initialized from scratch.")

        # set up summary writer
        if self.summary_dir is not None:
            if isinstance(self.strategy, HorovodStrategy) and not self.is_chief():
                self.summary_dir = self.create_temp_dir(self.summary_dir)
            else:
                tf.io.gfile.makedirs(self.summary_dir)
            self.summary_writer = tf.summary.create_file_writer(self.summary_dir)
        else:
            self.summary_writer = None

    def update_step(self):
        """
        Increments the train step of the model by 1
        """
        self.train_step.assign(self.train_step + 1)

    def set_step(self, step):
        """Sets the current training step of the model to a given value

        Args:
            step (int): The new step
        """
        self.train_step.assign(step)

    def get_step(self):
        """Returns the current training step

        Returns:
            int: A regular integer.
        """
        if isinstance(self.strategy, tf.distribute.MirroredStrategy):
            step = self.strategy.gather(self.train_step, axis=0)[0].numpy()
        elif isinstance(self.strategy, tf.distribute.MultiWorkerMirroredStrategy):
            step = self.train_step.numpy()
        else:
            step = self.train_step.numpy()

        return step

    def save_model(self):
        """Saves the model with the CheckpointManager

        Raises:
            ValueError: If there's no checkpoint directory.
            Exception: When the model is initialized from scratch, but the given checkpoint directory is non-empty.
        """

        if self.checkpoint_dir is None:
            raise ValueError("No checkpoint directory was declared during the init of the model, it can not be saved.")

        if not self.restore_from_checkpoint and self.n_init_checkpoints != 0:
            raise Exception(
                f"The specified checkpoint directory {self.checkpoint_dir} was not empty at initialization, can not"
                f" save a model initialized from scratch there."
            )

        # save the model
        self.checkpoint_manager.save()
        LOGGER.info(f"Successfully saved the model in {self.checkpoint_manager.directory}")

        # clean up the temoporary checkpoints of the non-chief workers
        if isinstance(self.strategy, (tf.distribute.MultiWorkerMirroredStrategy, HorovodStrategy)):
            if not self.is_chief():
                tf.io.gfile.rmtree(self.checkpoint_dir)

            LOGGER.info(f"Deleted the temporary checkpoint directory {self.checkpoint_dir}")

    def restore_model(self):
        """Restores the model from a checkpoint using the CheckpointManager that picks the most recent checkpoint.

        Raises:
            ValueError: If there's no checkpoint directory or it's empty.
        """

        if self.checkpoint_dir is None:
            raise ValueError(f"No checkpoint directory was given, the network can not be restored.")

        if len(self.checkpoint_manager.checkpoints) == 0:
            raise ValueError(f"A non empty checkpoint_dir {self.checkpoint_dir} has to be passed")

        restore_dir = self.checkpoint_manager.restore_or_initialize()
        LOGGER.info(f"Network successfully restored from checkpoint {restore_dir}.")

    def build_network(self, input_shape):
        """Builds the internal HealpyGCNN with a given input shape

        Args:
            input_shape (tuple): Input shape of the netork, may contain None (like for the batch dimension)
        """
        self.network.build(input_shape=input_shape)

    def print_summary(self, **kwargs):
        """Prints the summary of the internal network

        Args:
            kwargs: passed to HealpyGCNN.summary
        """
        self.network.summary(**kwargs)

    def write_summary(self, label, value, summary_type="scalar"):
        # this is part of the model graph, so has to be executed with every step. An additional condition like
        # step % log_every_n_steps == 0 is therefore not feasible
        if self.summary_writer is not None:
            with self.summary_writer.as_default():
                if summary_type == "scalar":
                    tf.summary.scalar(label, value)
                elif summary_type == "histogram":
                    tf.summary.histogram(label, value)
                elif summary_type == "image":
                    tf.summary.image(label, value)
                else:
                    raise ValueError(f"Invalid summary type {summary_type} was passed")

    def create_temp_dir(self, chief_dir):
        """For a distribution strategy with multiple workers, the non-chief workers need to create temporary files.

        Args:
            chief_dir (str): The directory of the chief worker, which is always one level above the temporary ones.

        Returns:
            str: The temporary directory associated with the worker.
        """
        assert not self.is_chief(), f"Only the non-chief workers should create temporary directories"

        assert isinstance(
            self.strategy, (tf.distribute.MultiWorkerMirroredStrategy, HorovodStrategy)
        ), f"Invalid strategy {self.strategy} was passed, should be MultiWorkerMirroredStrategy or HorovodStrategy"

        # set up temporary directories for the non-chief workers
        temp_dir = tf.io.gfile.join(chief_dir, "temp_worker_" + str(self.strategy.cluster_resolver.task_id))
        tf.io.gfile.makedirs(temp_dir)

        return temp_dir

    def copy_chief_to_temp_dir(self, chief_dir, temp_dir):
        """For a distribution strategy with multiple workers, copy the contents of the chief's directory to the
        workers's temporary ones.

        Args:
            chief_dir (str): The directory of the chief worker, which is always one level above the temporary ones.
            temp_dir (str): As set up by self.create_temp_dir, the temporary directory associated with the worker.
        """
        # copy over the checkpoints from the chief to the temporary directories of the non-chief workers
        chief_files = tf.io.gfile.listdir(chief_dir)
        for chief_file in chief_files:
            full_chief_file = tf.io.gfile.join(chief_dir, chief_file)

            if os.path.isfile(full_chief_file):
                full_temp_file = tf.io.gfile.join(temp_dir, chief_file)
                tf.io.gfile.copy(full_chief_file, full_temp_file, overwrite=True)

    def delete_temp_dir(self, temp_dir):
        pass

    def delete_temp_summaries(self):
        """Only one copy of the TensorBoard summary is needed, so it can be deleted after training for non-chief
        workers.
        """
        if isinstance(self.strategy, HorovodStrategy) and not self.is_chief():
            tf.io.gfile.rmtree(self.summary_dir)
            LOGGER.info(f"Deleted the temporary summary directory {self.summary_dir}")

    def is_chief(self):
        """Within the tf.distribute.MultiWorkerStrategy, whether the worker is the chief or not. Adapted from
        https://www.tensorflow.org/tutorials/distribute/multi_worker_with_ctl#checkpoint_saving_and_restoring

        Raises:
            AttributeError: If called for a model that is not distributed with tf.distribute.MultiWorkerStrategy

        Returns:
            bool: Whether the worker is the chief or not.
        """

        if isinstance(self.strategy, tf.distribute.MultiWorkerMirroredStrategy):
            task_type = self.strategy.cluster_resolver.task_type
            task_id = self.strategy.cluster_resolver.task_id
            cluster_spec = self.strategy.cluster_resolver.cluster_spec()

            return task_type == "chief" or (
                task_type == "worker" and task_id == 0 and "chief" not in cluster_spec.as_dict()
            )

        elif isinstance(self.strategy, HorovodStrategy):
            return hvd.rank() == 0

        else:
            raise AttributeError(
                f"The concept of chief only makes sense for tf.distribute.MultiWorkerMirroredStrategy, but this model "
                f"is set up with {self.strategy}"
            )

    def horovod_broadcast_variables(self):
        """Broadcast the network and optimizer variables from the chief to all other workers. This is only relevant
        for Horovod, as the builtin strategies do this under the hood.
        """
        hvd.broadcast_variables(self.network.weights, root_rank=0)
        hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

    def train_step(
        self,
        input_tensor,
        loss_function,
        input_labels=None,
        clip_by_value=None,
        clip_by_norm=None,
        clip_by_global_norm=None,
        l2_norm_weight=None,
    ):
        # non distributed
        if self.strategy is None:
            return self.base_train_step(
                input_tensor=input_tensor,
                loss_function=loss_function,
                input_labels=input_labels,
                clip_by_value=clip_by_value,
                clip_by_norm=clip_by_norm,
                clip_by_global_norm=clip_by_global_norm,
                l2_norm_weight=l2_norm_weight,
            )

        # distributed
        elif isinstance(self.strategy, tf.distribute.Strategy):
            return self.distributed_train_step(
                input_tensor=input_tensor,
                loss_function=loss_function,
                input_labels=input_labels,
                clip_by_value=clip_by_value,
                clip_by_norm=clip_by_norm,
                clip_by_global_norm=clip_by_global_norm,
                l2_norm_weight=l2_norm_weight,
            )

        else:
            raise ValueError(f"Invalid strategy {self.strategy} was passed")

    def base_train_step(
        self,
        input_tensor,
        loss_function,
        input_labels=None,
        clip_by_value=None,
        clip_by_norm=None,
        clip_by_global_norm=None,
        l2_norm_weight=None,
    ):
        """A base train step given a loss funtion and an input tensor. The method evaluates the network and performs a
        single gradient decent step. Note that it should be wrapped in a tf.function. If multiple clippings are
        requested, the order will be:
            * by value
            * by norm
            * by global norm

        Args:
            input_tensor (tf.tensor): The input to the network
            loss_function (callable): The loss function, a callable that takes predictions of the network (and if
                provided, the input_labels) as input and returns a loss
            input_labels (tf.tensor, optional): Labels of the input_tensor. Defaults to None.
            clip_by_value (tf.tensor, optional): Clip the gradients by given 1d array of values into the interval
                [value[0], value[1]]. Defaults to None (no clipping).
            clip_by_norm (tf.tensor, optional): Clip the gradients by norm. Defaults to None (no clipping).
            clip_by_global_norm (tf.tensor, optional): Clip the gradients by global norm. Defaults to None (no
                clipping).
            l2_norm_weight (float, optional): Weight for the L2 norm of the trainable weights. Defaults to None
                (no regularization).
        """
        LOGGER.warning("Performing a base_train_step in python instead of a tf.function")
        trainable_variables = self.network.trainable_variables

        with tf.GradientTape() as tape:
            predictions = self.network(input_tensor, training=True)

            # compute the loss
            if input_labels is None:
                loss = loss_function(predictions)
            else:
                loss = loss_function(predictions, input_labels)
            self.write_summary("loss", loss)

            # handle the l2 norm
            if l2_norm_weight is not None:
                l2_loss = tf.linalg.global_norm(trainable_variables)
                self.write_summary("l2_loss", l2_loss)

                loss = loss + l2_norm_weight * l2_loss

        # NOTE distributed delta loss, get the global gradients on the level of the tape for Horovod
        if isinstance(self.strategy, HorovodStrategy):
            tape = hvd.DistributedGradientTape(tape)

        gradients = tape.gradient(loss, trainable_variables)

        # NOTE distribute delta loss, get global gradients on the level of the gradients for the builtin strategies
        if isinstance(self.strategy, tf.distribute.Strategy):
            gradients = tf.distribute.get_replica_context().all_reduce("MEAN", gradients)

        # clip the gradients
        if clip_by_value is not None:
            gradients = [tf.clip_by_value(g, clip_by_value[0], clip_by_value[1]) for g in gradients]
        if clip_by_norm is not None:
            gradients = [tf.clip_by_norm(g, clip_by_norm) for g in gradients]

        glob_norm = tf.linalg.global_norm(gradients)

        self.write_summary("global_grad_norm", glob_norm)

        if clip_by_global_norm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, clip_by_global_norm, use_norm=glob_norm)

        # apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        # update the step
        self.update_step()

        return loss

    def distributed_train_step(
        self,
        input_tensor,
        loss_function,
        input_labels=None,
        clip_by_value=None,
        clip_by_norm=None,
        clip_by_global_norm=None,
        l2_norm_weight=None,
    ):
        """A distributed train step to be used in conjunction with a tf.distribute.Strategy like in
        https://www.tensorflow.org/tutorials/distribute/custom_training.
        Note that this method is not needed when training is distributed with Horovod.

        The method evaluates the network and performs a single gadient decent step. Note it should be wrapped in a
        tf.function. If multiple clippings are requested, the order will be:
            * by value
            * by norm
            * by global norm

        For correct normalization of the loss over multiple replicas/GPUs, the local batch size has to be the same
        accross the replicas, which is the case when the global batch size is divisible by the number of replicas.
        Note that there's no additional check for this.

        Args:
            input_tensor (tf.tensor): The input to the network
            loss_function (callable): The loss function, a callable that takes predictions of the network (and if
                provided, the input_labels) as input and returns a loss
            input_labels (tf.tensor, optional): Labels of the input_tensor. Defaults to None.
            clip_by_value (tf.tensor, optional): Clip the gradients by given 1d array of values into the interval
                [value[0], value[1]]. Defaults to None (no clipping).
            clip_by_norm (tf.tensor, optional): Clip the gradients by norm. Defaults to None (no clipping).
            clip_by_global_norm (tf.tensor, optional): Clip the gradients by global norm. Defaults to None (no
                clipping).
            l2_norm_weight (float, optional): Weight for the L2 norm of the trainable weights. Defaults to None
                (no regularization).
        """
        # the means here are taken over the local batches
        local_losses = self.strategy.run(
            self.base_train_step,
            args=(
                input_tensor,
                loss_function,
                input_labels,
                clip_by_value,
                clip_by_norm,
                clip_by_global_norm,
                l2_norm_weight,
            ),
        )

        # the mean of means is equal to the overall mean if the subgroups all have the same number of samples
        # https://en.wikipedia.org/wiki/Grand_mean
        global_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, local_losses, axis=None)
        LOGGER.warning(
            f"The distributed_train_step makes the assumption that the global batch size is divisible by the number"
            f" of replicas, ensure that this is the case"
        )

        self.write_summary("global_loss", global_loss)

        return global_loss

    def __call__(self, input_tensor, training=False, numpy=False, layer=None, *args, **kwargs):
        """Calls the network underlying the model

        Args:
            input_tensor (tf.tensor, np.ndarray): the tensor (or array) to call on
            training (bool, optional): Whether we are training or evaluating (e.g. necessary for batch norm). Defaults
                to False.
            numpy (bool, optional): Return a numpy array instead of a tensor. Defaults to False.
            layer (int, optional): Propagate only up to this layer, can be -1. Defaults to None.

        Returns:
            tf.tensor, np.ndarray: Tensor or array, depending on the numpy argument
        """
        if layer is None:
            preds = self.network(input_tensor, training=training, *args, **kwargs)
        else:
            preds = input_tensor
            for layer in self.network.layers[:layer]:
                preds = layer(preds)

        if numpy:
            return preds.numpy()
        else:
            return preds

    @tf.function
    def tf_call(self, input_tensor, training=False, *args, **kwargs):
        """Calls the network underlying the model as a tf.function

        Args:
            input_tensor (tf.tensor, np.ndarray): the tensor (or array) to call on
            training (bool, optional): Whether we are training or evaluating (e.g. necessary for batch norm). Defaults
                to False.

        Returns:
            tf.tensor, np.ndarray: Tensor or array, depending on the numpy argument
        """
        LOGGER.warning(f"Tracing tf_call")

        preds = self.network(input_tensor, training=training, *args, **kwargs)

        return preds
