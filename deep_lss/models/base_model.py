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
        xla=False,
        z_bank_size=None,
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
            xla (bool, optional): Whether to enable XLA just in time compilation. Note that this is incompatible with
                the DeepSphere graph convolutional layers, as they contain unsupported
                SparseDenseMatirxMultiplications. Defaults to False.
            z_bank_size (int, optional): Size of the memory bank for the z regularization. Defaults to None, then no
                memory bank is used.
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
        elif isinstance(network, (tf.keras.Sequential, tf.keras.Model)):
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
        self.xla = xla
        self.z_bank_size = z_bank_size
        self.z_bank = None
        self.z_bank_index = None

        # set up the optimizer
        if isinstance(self.optimizer, (tf.keras.optimizers.Optimizer, tf.keras.optimizers.legacy.Optimizer)):
            pass
        elif self.optimizer is None:
            self.optimizer = tf.keras.optimizers.legacy.Adam(**optimizer_kwargs)
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

    def increment_step(self):
        """
        Increments the train step of the model by 1
        """
        self.train_step.assign(self.train_step + 1)

    def change_step(self, delta):
        """
        Increments the train step of the model by a given value

        Args:
            delta (int): The value to increment the step by
        """
        self.train_step.assign_add(delta)

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
            if self.strategy.num_replicas_in_sync == 1:
                step = self.strategy.gather(self.train_step, axis=0).numpy()
            else:
                # step = self.strategy.gather(self.train_step, axis=0)[0].numpy()
                step = int(self.strategy.experimental_local_results(self.train_step)[0].numpy())
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

    def restore_model_from_checkpoint_path(self, checkpoint_path):
        """Restores the model from a concrete checkpoint passed as a function argument.
        This should have a format like checkpoint_dir/ckpt-10 for the 10th checkpoint.

        Raises:
            ValueError: If there's no checkpoint directory or it's empty.
        """

        if self.checkpoint_dir is None:
            raise ValueError(f"No checkpoint directory was given, the network can not be restored.")

        self.checkpoint.restore(checkpoint_path)
        LOGGER.info(f"Network successfully restored from checkpoint {checkpoint_path}.")

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

    def write_summary(self, label, value, summary_type="scalar", skip=False):
        # this is part of the model graph, so has to be executed with every step. An additional condition like
        # step % log_every_n_steps == 0 is therefore not feasible
        if (self.summary_writer is not None) and (not skip):
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

    def _compute_vicreg_loss(self, z):
        """Compute VICReg variance and covariance loss terms for standardizing features.

        This implements the variance and covariance terms from VICReg https://arxiv.org/abs/2105.04906 to encourage
        the features to have unit variance and zero covariance between dimensions.
        In the paper, they penalize a hinge loss to make sure tha variance is greater than one. Here, penalize any
        deviations from one to standardize the features.

        Args:
            z (tf.tensor): Features from the penultimate layer, shape (batch_size, feature_dim)

        Returns:
            tf.tensor: Scalar loss combining variance and covariance regularization
        """
        batch_size = tf.cast(tf.shape(z)[0], tf.float32)
        feature_dim = tf.cast(tf.shape(z)[1], tf.float32)

        z_centered = z - tf.reduce_mean(z, axis=0, keepdims=True)

        # variance loss: penalizes deviations from std = 1 (standardization)
        std = tf.sqrt(tf.reduce_mean(tf.square(z_centered), axis=0) + 1e-4)
        var_loss = tf.reduce_mean(tf.square(std - 1.0))

        # covariance loss: encourages off-diagonal elements to be 0
        cov_matrix = tf.matmul(z_centered, z_centered, transpose_a=True) / (batch_size - 1)
        cov_loss = tf.reduce_sum(tf.square(cov_matrix)) - tf.reduce_sum(tf.square(tf.linalg.diag_part(cov_matrix)))
        # normalize by number of off-diagonal elements to keep scale consistent with var_loss
        cov_loss = cov_loss / (feature_dim**2 - feature_dim)

        return var_loss + cov_loss

    def _compute_mmd_loss(self, z, interpretable=False):
        """Compute Maximum Mean Discrepancy loss between features and standard Gaussian.

        This penalizes deviations from a standard Gaussian distribution N(0, I) using the biased MMD estimator
        with RBF kernel. The biased estimator includes diagonal terms and is preferred for numerical stability
        (always non-negative) while being asymptotically equivalent to the unbiased version.

        Uses dimension-aware bandwidths that scale with sqrt(feature_dim) to account for the typical distances
        in high-dimensional Gaussian distributions.

        Args:
            z (tf.tensor): Features from the penultimate layer, shape (batch_size, feature_dim)
            interpretable (bool): If True, includes the k_gg term so minimum loss is exactly zero

        Returns:
            tf.tensor: Scalar MMD loss
        """
        batch_size = tf.shape(z)[0]
        feature_dim = tf.shape(z)[1]

        # sample from standard Gaussian with the same shape
        z_gaussian = tf.random.normal(shape=tf.shape(z))

        # dimension-aware bandwidth scaling: typical distances in d-dimensional Gaussian scale as sqrt(d)
        dim_scale = tf.sqrt(tf.cast(feature_dim, tf.float32))

        def rbf_kernel(x, y):
            """Compute RBF kernel matrix with dimension-aware bandwidths."""
            xx = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
            yy = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)
            xy = tf.matmul(x, y, transpose_b=True)
            distances = xx - 2 * xy + tf.transpose(yy)
            # for numerical stability
            distances = tf.nn.relu(distances)

            # scale bandwidths by sqrt(feature_dim)
            bandwidths = [0.1 * dim_scale, 1.0 * dim_scale, 10.0 * dim_scale]

            kernel_matrix = tf.zeros_like(distances)
            for bandwidth in bandwidths:
                kernel_matrix += tf.exp(-distances / (2 * bandwidth**2))
            return kernel_matrix / len(bandwidths)

        # compute kernel matrices
        k_zz = rbf_kernel(z, z)
        k_zg = rbf_kernel(z, z_gaussian)
        if interpretable:
            k_gg = rbf_kernel(z_gaussian, z_gaussian)

        # compute MMD^2
        batch_size_f = tf.cast(batch_size, tf.float32)
        mmd_loss = tf.reduce_sum(k_zz) / (batch_size_f * batch_size_f) - 2 * tf.reduce_sum(k_zg) / (
            batch_size_f * batch_size_f
        )

        # with this term, the minimum loss is zero. Otherwise, it can become negative
        if interpretable:
            mmd_loss = mmd_loss + tf.reduce_sum(k_gg) / (batch_size_f * batch_size_f)

        return mmd_loss

    def _compute_sw_loss(self, z, num_projections=None):
        """Compute Sliced Wasserstein distance between features and standard Gaussian.

        Projects the distribution onto random 1D lines where Wasserstein distance has a closed-form
        solution via sorting.

        Args:
            z (tf.tensor): Features from the penultimate layer, shape (batch_size, feature_dim)
            num_projections (int): Number of random projection lines. If None, defaults to max(512, feature_dim).

        Returns:
            tf.tensor: Scalar SW loss (squared)
        """
        feature_dim = tf.shape(z)[1]

        if num_projections is None:
            num_projections = tf.maximum(512, feature_dim)

        z_gaussian = tf.random.normal(shape=tf.shape(z))

        # generate random projection vectors on the unit sphere
        projections = tf.random.normal(shape=(feature_dim, num_projections))
        projections = tf.math.l2_normalize(projections, axis=0)

        # project both distributions
        projected_z = tf.matmul(z, projections)
        projected_gaussian = tf.matmul(z_gaussian, projections)

        # sort and compute quantile matching loss
        sorted_z = tf.sort(projected_z, axis=0)
        sorted_gaussian = tf.sort(projected_gaussian, axis=0)

        sw_loss = tf.reduce_mean(tf.square(sorted_z - sorted_gaussian))

        return sw_loss

    def _update_and_get_z_bank(self, z):
        """Updates the memory bank with the current batch features and returns the concatenated features.

        Args:
            z (tf.tensor): Features from the current batch, shape (batch_size, feature_dim)
        Returns:
            tuple: (z_loss_input, z_scale) where z_loss_input is the concatenation of z_features and z_bank,
                   and z_scale is the scaling factor for the loss.
        """
        if self.z_bank_size is None:
            return z, 1.0

        if self.z_bank is None:
            LOGGER.info(f"Initializing z memory bank with size {self.z_bank_size}")
            feature_dim = z.shape[-1]
            self.z_bank = tf.Variable(
                tf.random.normal((self.z_bank_size, feature_dim), dtype=z.dtype),
                trainable=False,
                name="z_bank",
            )
            self.z_bank_index = tf.Variable(0, trainable=False, name="z_bank_index", dtype=tf.int64)

        # update the bank
        batch_size = tf.shape(z)[0]
        indices = (self.z_bank_index + tf.range(batch_size, dtype=tf.int64)) % self.z_bank_size
        update_indices = tf.expand_dims(indices, 1)
        self.z_bank.scatter_nd_update(update_indices, z)
        self.z_bank_index.assign((self.z_bank_index + tf.cast(batch_size, tf.int64)) % self.z_bank_size)

        # concatenate the bank to the features
        z_loss_input = tf.concat([z, self.z_bank.value()], axis=0)

        # scaling factor to compensate for diluted gradients
        z_scale = (tf.cast(batch_size, tf.float32) + tf.cast(self.z_bank_size, tf.float32)) / tf.cast(
            batch_size, tf.float32
        )

        return z_loss_input, z_scale

    def train_step(
        self,
        input_tensor,
        loss_function,
        input_labels=None,
        clip_by_value=None,
        clip_by_norm=None,
        clip_by_global_norm=None,
        l2_norm_weight=None,
        z_weight=None,
        z_type=None,
        z_layer="last",
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
                z_weight=z_weight,
                z_type=z_type,
                z_layer=z_layer,
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
                z_weight=z_weight,
                z_type=z_type,
                z_layer=z_layer,
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
        z_weight=None,
        z_type=None,
        z_layer="last",
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
            z_weight (float, optional): Weight for the regularization of features z in the penultimate layer.
                Defaults to None (no regularization).
            z_type (str, optional): Type of regularization for z features, either "cov" (VICReg variance and
                covariance terms) or "mmd" (Maximum Mean Discrepancy penalty for standard Gaussian). Defaults to None.
            z_layer (str, optional): Layer to compute z features for regularization. "penultimate" or "last".
                Defaults to "last".
        """
        LOGGER.warning("Performing a base_train_step in python instead of a tf.function")

        if not hasattr(self, "trainable_variables"):
            self.trainable_variables = self.network.trainable_variables

        with tf.GradientTape() as tape:
            predictions = self.network(input_tensor, training=True)

            # compute the loss
            if input_labels is None:
                loss = loss_function(predictions)
            else:
                loss = loss_function(predictions, input_labels)
            self.write_summary("loss", loss, skip=self.xla)

            # handle the l2 norm
            if l2_norm_weight is not None:
                l2_loss = tf.linalg.global_norm(self.trainable_variables)
                self.write_summary("l2_loss", l2_loss, skip=self.xla)

                loss = loss + l2_norm_weight * l2_loss

            # handle the z regularization
            if z_weight is not None:
                if z_layer == "penultimate":
                    z_features = input_tensor
                    for layer in self.network.layers[:-1]:
                        z_features = layer(z_features, training=True)
                elif z_layer == "last":
                    z_features = predictions
                else:
                    raise ValueError(f"Invalid z_layer '{z_layer}', must be 'penultimate' or 'last'")

                # memory bank
                z_input, z_scale = self._update_and_get_z_bank(z_features)

                if z_type == "cov":
                    LOGGER.info("Using VICReg covariance loss for z regularization")
                    z_loss = self._compute_vicreg_loss(z_input)
                elif z_type == "mmd":
                    LOGGER.info("Using MMD loss for z regularization")
                    z_loss = self._compute_mmd_loss(z_input)
                elif z_type == "sw":
                    LOGGER.info("Using Sliced Wasserstein loss for z regularization")
                    z_loss = self._compute_sw_loss(z_input)
                else:
                    raise ValueError(f"Invalid z_type {z_type}. Must be 'cov', 'mmd', or 'sw'.")

                self.write_summary(f"z_{z_type}_loss", z_loss, skip=self.xla)

                loss = loss + z_weight * z_scale * z_loss

            # mixed precision
            if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                loss = self.optimizer.get_scaled_loss(loss)

        # NOTE distributed delta loss, get the global gradients on the level of the tape for Horovod
        if isinstance(self.strategy, HorovodStrategy):
            tape = hvd.DistributedGradientTape(tape)

        gradients = tape.gradient(loss, self.trainable_variables)

        # NOTE distribute delta loss, get global gradients on the level of the gradients for the builtin strategies
        if isinstance(self.strategy, tf.distribute.Strategy):
            gradients = tf.distribute.get_replica_context().all_reduce("MEAN", gradients)

        if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            gradients = self.optimizer.get_unscaled_gradients(gradients)

        # clip the gradients
        if clip_by_value is not None:
            gradients = [tf.clip_by_value(g, clip_by_value[0], clip_by_value[1]) for g in gradients]
        if clip_by_norm is not None:
            gradients = [tf.clip_by_norm(g, clip_by_norm) for g in gradients]

        glob_norm = tf.linalg.global_norm(gradients)
        self.write_summary("global_grad_norm", glob_norm, skip=self.xla)

        if clip_by_global_norm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, clip_by_global_norm, use_norm=glob_norm)

        # apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # update the step
        self.increment_step()

        # log the learning rate
        current_learning_rate = self.optimizer.learning_rate
        if callable(current_learning_rate):
            current_learning_rate = current_learning_rate(self.train_step)
        self.write_summary("learning_rate", current_learning_rate, skip=self.xla)

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
        z_weight=None,
        z_type=None,
        z_layer="last",
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
            z_weight (float, optional): Weight for the regularization of features z in the penultimate layer.
                Defaults to None (no regularization).
            z_type (str, optional): Type of regularization for z features, either "cov" (VICReg variance and
                covariance terms) or "mmd" (Maximum Mean Discrepancy penalty for standard Gaussian). Defaults to None.
            z_layer (str, optional): Layer to compute z features for regularization. "penultimate" or "last".
                Defaults to "last".
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
                z_weight,
                z_type,
                z_layer,
            ),
        )

        # the mean of means is equal to the overall mean if the subgroups all have the same number of samples
        # https://en.wikipedia.org/wiki/Grand_mean
        LOGGER.warning(
            f"The distributed_train_step makes the assumption that the global batch size is divisible by the number"
            f" of replicas, ensure that this is the case"
        )
        global_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, local_losses, axis=None)
        self.write_summary("global_loss", global_loss, skip=self.xla)

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
