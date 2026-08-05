# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created January 2024
Author: Arne Thomsen

To train over the grid part of the CosmoGrid with the
    - Mean Squared Error (MSE)
    - Likelihood loss (see https://arxiv.org/abs/1906.03156)
    - mutual information loss (see Section 7.3 in https://arxiv.org/pdf/2009.08459).
"""

import warnings
import tensorflow as tf

from deepsphere import HealpyGCNN

from msfm.utils import logger
from deep_lss.utils import likelihood_loss, mutual_info_loss
from deep_lss.utils.distribute import HorovodStrategy
from deep_lss.models.base_model import BaseModel
from deep_lss.utils.configuration import get_backend_floatx

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


class GridLossModel(BaseModel):
    """
    This class subclasses the BaseModel to employ a HealpyGCNN with the information maximizing delta loss, which trains
    at the fiducial and its perturbations.
    """

    def __init__(
        self,
        network,
        # DeepSphere
        n_side=None,
        indices=None,
        n_neighbors=20,
        max_batch_size=None,
        initial_Fin=None,
        spmm_backend="csr",
        # general
        input_shape=None,
        optimizer=None,
        optimizer_kwargs={},
        summary_dir=None,
        checkpoint_dir=None,
        restore_checkpoint=False,
        max_checkpoints=3,
        init_step=0,
        z_bank_size=None,
        strategy=None,
        xla=False,
        summary_every=1,
    ):
        """Initializes a graph convolutional neural network using the healpy pixelization scheme.

        Args:
            network (Union[list, tf.keras.Sequential]): The underlying network of the model. Can be a list of layers,
                then either a regular tf.keras.Sequential or HealpyGCNN model is initialized.
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
            input_shape (tf.tensor, optional): Input shape of the network, necessary if one wants to restore the model.
                Defaults to None.
            optimizer (tf.keras.optimizers.Optimizer, optional): Optimizer of the model. Defaults to None, which loads
                Adam.
            optimizer_kwargs (dict, optional): Keyword arguments passed to the optimizer. Defaults to {}.
            summary_dir (str, optional): Directory to save the summaries. Defaults to None.
            checkpoint_dir (str, optional): Directory where to save the weights and optimizer. Defaults to None.
            restore_checkpoint (bool, optional): Whether to restore the network from a checkpoint, or initialize it.
                Defaults to False.
            max_checkpoints (int, optional): Maximum number of checkpoints to keep. Defaults to 3.
            init_step (int, optional): Initial step. Defaults to 0.
            z_bank_size (int, optional): Size of the memory bank for the z regularization. Defaults to None, then no
                memory bank is used.
            strategy (Union[tf.distribute.Strategy, deep_lss.utils.distribute.HorovodStrategy], optional):
                The distribution strategy the model was created within. Defaults to None, then training is local.
            xla (bool, optional): Whether to enable XLA just in time compilation. Note that this is incompatible with
                the DeepSphere graph convolutional layers, as they contain unsupported
                SparseDenseMatirxMultiplications. Defaults to False.
            summary_every (int, optional): Write TensorBoard summaries every N training steps. Defaults to 1.
        """

        # init the base model
        super(GridLossModel, self).__init__(
            network=network,
            input_shape=input_shape,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            summary_dir=summary_dir,
            checkpoint_dir=checkpoint_dir,
            restore_checkpoint=restore_checkpoint,
            max_checkpoints=max_checkpoints,
            init_step=init_step,
            strategy=strategy,
            xla=xla,
            summary_every=summary_every,
            # DeepSphere
            n_side=n_side,
            indices=indices,
            n_neighbors=n_neighbors,
            max_batch_size=max_batch_size,
            initial_Fin=initial_Fin,
            spmm_backend=spmm_backend,
            z_bank_size=z_bank_size,
        )
        LOGGER.info("Initialized the GridLossModel")

    def setup_grid_loss_step(
        self,
        loss="mutual_info",
        # shapes
        batch_size=None,
        dim_theta=None,
        dim_x=None,
        dim_channels=None,
        # mutual information loss
        dim_summary=None,
        mutual_info_estimator="variational",
        mutual_info_kwargs={},
        # mse loss
        label_std=None,
        # likelihood loss
        lambda_tikhonov=None,
        # gradient clipping + regularization
        clip_by_value=None,
        clip_by_norm=None,
        clip_by_global_norm=10.0,
        l2_norm_weight=None,
        z_weight=None,
        z_type=None,
        z_layer="last",
        # misc
        img_summary=False,
        xla=False,
    ):
        """Set up the training step for the grid model.

        Args:
            dim_theta (int, optional): The number of cosmological parameters making up the label. Defaults to None.
            batch_size (int, optional): The batch size. Defaults to None.
            dim_x (int, optional): Input dimension of the network, must be provided if the network is not a
                HealpyGCNN. Defaults to None.
            dim_channels (int, optional): The number of channels. Defaults to None.
            loss (str, optional): The type of loss function to use. Defaults to "mutual_info".
            dim_summary (int, optional): The dimensionality of the summary. This is only a free parameter for the
                mutual information loss. Defaults to None.
            mutual_info_estimator (str, optional): The estimator to use for mutual information loss. Defaults to
                "variational", which produced the best results on tests on the Cls.
            mutual_info_kwargs (dict, optional): Additional keyword arguments for the mutual information estimator like
                makeup of the Gaussian Mixture Model. Defaults to {}.
            label_std (np.ndarray | tf.Tensor, optional): Per-parameter standard deviation of the labels, of
                shape (dim_theta,). Only used by the "mse" loss to weight each parameter's squared error by
                1/std**2, which standardizes the loss across parameters that live on different scales (this is
                equivalent at the optimum to standardizing the labels while keeping predictions in physical
                units). Defaults to None, then an unweighted MSE is used.
            lambda_tikhonov (float, optional): Regularization parameter for the Tikhonov regularization in the
                likelihood loss. Defaults to None, then no regularization is applied.
            clip_by_value (tf.tensor, optional): Clip the gradients by given 1d array of values into the interval
                [value[0], value[1]]. Defaults to None (no clipping).
            clip_by_norm (tf.tensor, optional): Clip the gradients by norm. Defaults to None (no clipping).
            clip_by_global_norm (tf.tensor, optional): Clip the gradients by global norm. Defaults to 10.0.
            l2_norm_weight (float, optional): Weight for the L2 norm of the trainable weights. Defaults to None
                (no regularization).
            z_weight (float | dict | None, optional): Weight(s) for z-feature regularization. For
                ``z_type="vicreg"`` this is a dict ``{variance: float|None, covariance: float|None,
                invariance: float|None}`` (each term independently weighted; setting a term to None disables
                it). For ``z_type="mmd"`` or ``"sw"`` it is a single float. When the invariance term is set,
                ``grid_train_step`` requires the ``i_sobol`` and ``i_signal`` index tensors as additional
                inputs, and the memory bank (``z_bank_size``) must be disabled. Defaults to None.
            z_type (str, optional): Type of regularization for z features. One of ``"vicreg"``, ``"mmd"``,
                ``"sw"``. Defaults to None.
            z_layer (str, optional): Layer to compute z features for regularization. "penultimate" or "last".
                Defaults to "last".
            img_summary (bool, optional): Whether to write image summaries of the covariance matrix. Defaults to False.
            xla (bool, optional): Whether to enable XLA just in time compilation. Defaults to False.

        Raises:
            ValueError: If an invalid strategy is passed.

        Note:
            - If the loss type is "mse", pass ``label_std`` so the per-parameter squared errors are weighted
              by 1/std**2 (the cosmological parameters live on different scales); otherwise the loss is
              dominated by the large-scale parameters.
            - If the loss type is "likelihood", the number of parameters (dim_theta) must be passed.
        """

        if self.xla:
            LOGGER.warning("Using XLA just in time compilation")

        # the VICReg invariance term needs per-sample (i_sobol, i_signal) ids; derived once here so the
        # downstream closure builders, the early-error check, and the startup log all stay consistent.
        uses_invariance = z_type == "vicreg" and isinstance(z_weight, dict) and z_weight.get("invariance") is not None

        if uses_invariance and self.z_bank_size is not None:
            raise NotImplementedError(
                "The VICReg invariance term (z_weight['invariance']) is not yet supported in conjunction with "
                "the z memory bank (z_bank_size). The bank stores features without pair_ids, so positive "
                "matches against bank entries cannot be made. Set one or the other to None."
            )

        # surface which z-regularization terms are active at startup so it's obvious from the run header
        if z_weight is None:
            LOGGER.info("z regularization: none")
        elif z_type == "vicreg":
            assert isinstance(z_weight, dict), (
                f"For z_type='vicreg', z_weight must be a dict with keys 'variance', 'covariance', "
                f"'invariance' (each a float or None); got {type(z_weight).__name__}."
            )
            active = [f"{name}={z_weight.get(name)}" for name in ("variance", "covariance", "invariance")]
            bank_str = f", bank_size={self.z_bank_size}" if self.z_bank_size is not None else ""
            LOGGER.warning(f"z regularization active: vicreg ({', '.join(active)}, layer={z_layer}{bank_str})")
        else:
            bank_str = f", bank_size={self.z_bank_size}" if self.z_bank_size is not None else ""
            LOGGER.warning(f"z regularization active: {z_type} (weight={z_weight}, layer={z_layer}{bank_str})")

        vali_loss_kwargs = {}
        if loss == "mse":
            distributed = isinstance(
                self.strategy, (tf.distribute.MirroredStrategy, tf.distribute.MultiWorkerMirroredStrategy)
            )

            if label_std is not None:
                # weight each parameter's squared error by 1/std**2 so parameters on different scales
                # contribute equally; equivalent at the optimum to standardizing the labels while keeping
                # predictions in physical units. Guard against zero-variance (fixed) parameters.
                std = tf.constant(label_std, dtype=get_backend_floatx())
                std = tf.where(tf.equal(std, 0.0), tf.ones_like(std), std)
                w = 1.0 / std

                if distributed:
                    # to be compatible with the delta loss, the loss is averaged per replica
                    loss_fn = lambda preds, theta, training=True: (1.0 / batch_size) * tf.reduce_sum(
                        tf.reduce_mean(tf.square(w * (preds - theta)), axis=-1)
                    )
                else:
                    loss_fn = lambda preds, theta, training=True: tf.reduce_mean(tf.square(w * (preds - theta)))

                LOGGER.warning(f"Using the Mean Squared Error, label-std-weighted (label_std={label_std})")
            else:
                if distributed:
                    # to be compatible with the delta loss, the loss is averaged per replica
                    loss_fn = lambda preds, theta, training=True: (
                        1.0 / batch_size
                    ) * tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)(preds, theta)
                else:
                    loss_fn = lambda preds, theta, training=True: tf.keras.losses.MeanSquaredError(
                        reduction=tf.keras.losses.Reduction.AUTO
                    )(preds, theta)

                LOGGER.warning("Using the Mean Squared Error. Note that the labels should be normalized!")

            # for MSE the network predictions are the point estimates of the parameters, so the app's
            # vali_mse metric (preds vs cosmos, physical units) just needs the identity here.
            self.vali_posterior_mean_fn = lambda preds: preds

        elif loss == "likelihood":
            assert dim_theta is not None, "n_theta must be passed for the likelihood loss"

            # analogously to the delta loss, the per replica averaging of the likelihood loss is done in
            # likelihood_loss.py, so no distinction between distributed and non-distributed training is necessary here
            def loss_fn(preds, theta, training=True, summary_suffix=""):
                return likelihood_loss.neg_likelihood_loss(
                    preds,
                    theta,
                    dim_theta,
                    lambda_tikhonov,
                    training=training,
                    summary_writer=self.summary_writer,
                    summary_suffix=summary_suffix,
                    img_summary=img_summary,
                    xla=self.xla,
                )

            vali_loss_kwargs["summary_suffix"] = "_vali"

            LOGGER.warning("Using the likelihood loss")

        elif loss == "mutual_info":
            assert dim_theta is not None, "n_theta must be passed for the mutual information loss"

            if dim_summary is None:
                dim_summary = 2 * dim_theta
                LOGGER.warning(f"The dimensionality of the summary is set to {dim_summary}")

            # see Section 7.3 in https://arxiv.org/pdf/2009.08459
            if mutual_info_estimator == "variational":
                self.variational_head = mutual_info_loss.get_variational_model_from_summary(
                    dim_summary, dim_theta, **mutual_info_kwargs
                )

                if self.checkpoint_manager is not None:
                    LOGGER.warning("Mutual info loss, overwriting the checkpoint manager")
                    self.checkpoint = tf.train.Checkpoint(
                        network=self.network,
                        optimizer=self.optimizer,
                        variational_head=self.variational_head,
                        train_step=self.train_step,
                    )
                    self.checkpoint_manager = tf.train.CheckpointManager(
                        self.checkpoint,
                        self.checkpoint_dir,
                        max_to_keep=self.max_checkpoints,
                        checkpoint_name="ckpt",
                        step_counter=self.train_step,
                    )
                if self.restore_from_checkpoint:
                    LOGGER.warning("Mutual info loss, restoring the model again from within setup_grid_loss_step")
                    self.restore_model()

                self.trainable_variables = self.variational_head.trainable_variables + self.network.trainable_variables

                loss_fn = lambda preds, theta, training=True: tf.reduce_mean(
                    self.variational_head([preds, theta], training=training)
                )
                self.vali_posterior_mean_fn = self.variational_head.estimator.mean

            # see https://arxiv.org/pdf/2010.10079
            elif mutual_info_estimator == "distance_correlation":
                loss_fn = lambda preds, theta, training=True: mutual_info_loss.distance_correlation(
                    preds, theta, training=training
                )

            # see https://arxiv.org/pdf/2010.10079
            elif mutual_info_estimator == "jensen_shannon":
                # critic_net = mutual_info_loss.get_jensen_shannon_critic(self, dim_x, dim_theta, **mutual_info_kwargs)
                # loss_fn = lambda x, theta: mutual_info_loss.jensen_shannon_divergence(
                #     critic_net, x, theta, training=True
                # )
                raise NotImplementedError(
                    f"Mutual information loss type {mutual_info_estimator} is not implemented. this loss function is "
                    f"fundamentally different since one needs to pass the network itself as well, not just its "
                    f"predictions well, not just its predictions. This is not compatible with the current setup."
                    f"In any case, this loss is much slower than the others because of the inner loop."
                )

            else:
                raise ValueError(f"Invalid mutual_info_estimator {mutual_info_estimator} was passed")

            LOGGER.warning(f"Using the mutual information loss with the {mutual_info_estimator} estimator")

        # to use the same loss function sepearately, without the need to perform the training step
        self.vali_loss_fn = lambda preds, theta: loss_fn(preds, theta, training=False, **vali_loss_kwargs)

        # this isn't strictly necessary and could be removed
        if isinstance(self.network, HealpyGCNN):
            input_shape = (batch_size, len(self.network.indices_in), dim_channels)
        elif dim_x is not None:
            if dim_channels is not None:
                input_shape = (batch_size, dim_x, dim_channels)
            else:
                input_shape = (batch_size, dim_x)
        else:
            input_shape = None

        if input_shape is not None:
            current_float = get_backend_floatx()
            label_shape = (batch_size, dim_theta)
            input_signature = [
                tf.TensorSpec(shape=input_shape, dtype=current_float),
                tf.TensorSpec(shape=label_shape, dtype=current_float),
            ]
            if uses_invariance:
                # extra (i_sobol, i_signal) inputs forwarded by the training loop, used as positive-pair ids
                index_shape = (batch_size,)
                input_signature.extend([tf.TensorSpec(shape=index_shape, dtype=tf.int64)] * 2)
            tf_kwargs = {"input_signature": input_signature}
        else:
            tf_kwargs = {}

        # not distributed via tensorflow builtin
        if (self.strategy is None) or isinstance(self.strategy, HorovodStrategy):

            if not uses_invariance:

                @tf.function(jit_compile=self.xla, **tf_kwargs)
                def grid_train_step(x, theta):
                    LOGGER.warning("Tracing grid_train_step")
                    loss = self.base_train_step(
                        input_tensor=x,
                        input_labels=theta,
                        loss_function=loss_fn,
                        # gradient clipping + regularization
                        clip_by_value=clip_by_value,
                        clip_by_norm=clip_by_norm,
                        clip_by_global_norm=clip_by_global_norm,
                        l2_norm_weight=l2_norm_weight,
                        z_weight=z_weight,
                        z_type=z_type,
                        z_layer=z_layer,
                    )

                    return loss

            else:

                @tf.function(jit_compile=self.xla, **tf_kwargs)
                def grid_train_step(x, theta, i_sobol, i_signal):
                    LOGGER.warning("Tracing grid_train_step (with VICReg invariance)")
                    # pair_ids is forwarded as a tuple — stacking happens inside base_train_step where
                    # the tensors are no longer wrapped in PerReplica (cf. distributed grid_train_step).
                    loss = self.base_train_step(
                        input_tensor=x,
                        input_labels=theta,
                        loss_function=loss_fn,
                        # gradient clipping + regularization
                        clip_by_value=clip_by_value,
                        clip_by_norm=clip_by_norm,
                        clip_by_global_norm=clip_by_global_norm,
                        l2_norm_weight=l2_norm_weight,
                        z_weight=z_weight,
                        z_type=z_type,
                        z_layer=z_layer,
                        pair_ids=(i_sobol, i_signal),
                    )

                    return loss

        # distributed via tensorflow builtin
        elif isinstance(self.strategy, tf.distribute.Strategy):
            # passing an input_signature like above for a distributed dset leads the following error:
            # AttributeError: 'PerReplica' object has no attribute 'dtype'
            # Instead do like https://www.tensorflow.org/tutorials/distribute/input#using_the_element_spec_property
            if not uses_invariance:

                @tf.function
                def grid_train_step(x, theta):
                    LOGGER.warning("Tracing distributed grid_train_step")
                    global_loss = self.distributed_train_step(
                        input_tensor=x,
                        input_labels=theta,
                        loss_function=loss_fn,
                        # gradient clipping + regularization
                        clip_by_value=clip_by_value,
                        clip_by_norm=clip_by_norm,
                        clip_by_global_norm=clip_by_global_norm,
                        l2_norm_weight=l2_norm_weight,
                        z_weight=z_weight,
                        z_type=z_type,
                        z_layer=z_layer,
                    )

                    return global_loss

            else:

                @tf.function
                def grid_train_step(x, theta, i_sobol, i_signal):
                    LOGGER.warning("Tracing distributed grid_train_step (with VICReg invariance)")
                    # pair_ids is forwarded as a tuple of PerReplica tensors — strategy.run distributes each
                    # leaf independently, and the stack into a (B, 2) tensor happens inside base_train_step
                    # where the per-replica unwrapping has already taken place.
                    global_loss = self.distributed_train_step(
                        input_tensor=x,
                        input_labels=theta,
                        loss_function=loss_fn,
                        # gradient clipping + regularization
                        clip_by_value=clip_by_value,
                        clip_by_norm=clip_by_norm,
                        clip_by_global_norm=clip_by_global_norm,
                        l2_norm_weight=l2_norm_weight,
                        z_weight=z_weight,
                        z_type=z_type,
                        z_layer=z_layer,
                        pair_ids=(i_sobol, i_signal),
                    )

                    return global_loss

        else:
            raise ValueError(f"Invalid strategy {self.strategy} was passed")

        LOGGER.info(f"Set up the training step of the {loss} loss")
        self.grid_train_step = grid_train_step
        # tells the training loop whether to forward the (i_sobol, i_signal) tensors
        self.grid_train_step_uses_pair_ids = uses_invariance
