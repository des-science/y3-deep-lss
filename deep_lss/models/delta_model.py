# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created December 2022
Author: Arne Thomsen, Janis Fluri

Closely follows
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/delta_model.py
by Janis Fluri, 
the main difference is that I don't work with horovod and instead use the builtin distribution strategies of
TensorFlow.
"""

import warnings
import tensorflow as tf

from deepsphere import HealpyGCNN

from msfm.utils import logger
from deep_lss.utils import delta_loss
from deep_lss.models.base_model import BaseModel

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


class DeltaLossModel(BaseModel):
    """
    This class subclasses the BaseModel to employ a HealpyGCNN with the information maximizing delta loss.
    """

    def __init__(
        self,
        network,
        n_side,
        indices,
        n_neighbors=20,
        input_shape=None,
        optimizer=None,
        summary_dir=None,
        checkpoint_dir=None,
        restore_checkpoint=False,
        max_checkpoints=3,
        init_step=0,
    ):
        """Initializes a graph convolutional neural network using the healpy pixelization scheme.

        Args:
            network (list): A list of layers that make up the neural network.
            n_side (int): The healpy n_side of the input.
            indices (np.ndarray): 1d array of inidices, corresponding to the pixel ids of the input of the NN.
            n_neighbors (int, optional): Number of neighbors considered when building the graph, currently supported
                values are: 8, 20, 40 and 60. Defaults to 20.
            input_shape (tf.tensor, optional): Input shape of the network, necessary if one wants to restore the model.
                Defaults to None.
            optimizer (tf.keras.optimizers.Optimizer, optional): Optimizer of the model. Defaults to None, which loads
                Adam.
            summary_dir (str, optional): Directory to save the summaries. Defaults to None.
            checkpoint_dir (str, optional): Directory where to save the weights and optimizer. Defaults to None.
            restore_checkpoint (bool, optional): Whether to restore the network from a checkpoint, or initialize it.
                Defaults to False.
            init_step (int, optional): Initial step. Defaults to 0.
        """

        # get the network
        if n_side is None and indices is None:
            LOGGER.info("Initializing DeltaLossModel with a normal Sequential model")
            network = tf.keras.Sequential(layers=network)
        else:
            LOGGER.info("Initializing DeltaLossModel with a HealpyGCNN model")
            network = HealpyGCNN(nside=n_side, indices=indices, layers=network, n_neighbors=n_neighbors)

        # init the base model
        super(DeltaLossModel, self).__init__(
            network=network,
            input_shape=input_shape,
            optimizer=optimizer,
            summary_dir=summary_dir,
            checkpoint_dir=checkpoint_dir,
            restore_from_checkpoint=restore_checkpoint,
            max_checkpoints=max_checkpoints,
            init_step=init_step,
        )
        LOGGER.info(f"Initialized the DeltaLossModel")

    def setup_delta_loss_step(
        self,
        n_params,
        n_same,
        off_sets,
        # shapes
        n_points=1,
        n_input=None,
        n_channels=1,
        # regularization
        force_params_value=0.0,
        force_params_weight=1.0,
        jac_weight=100.0,
        jac_cond_weight=None,
        cov_weight=False,
        # summary statistic
        n_output=None,
        n_partial=None,
        weights=None,
        no_correlations=False,
        # gradient clipping
        clip_by_value=None,
        clip_by_norm=None,
        clip_by_global_norm=5.0,
        l2_norm_weight=None,
        # numerical stability
        use_log_det=True,
        tikhonov_regu=False,
        eps=1e-32,
        # tf.summary
        img_summary=False,
    ):
        """This sets up a function that performs one training step with the delta loss, which tries to maximize the
        information of the summary statistics. Note  it needs the maps need to be ordered in a specific way:
        * The shape of the predictions is (n_points * n_same * (2 * n_params + 1), n_output)
        * If one splits the predictions into (2 * n_params + 1) parts among the first axis one has the following scheme:
            * The first part was generated with the unperturbed parameters
            * The second part was generated with parameters where off_sets[0] was subtracted from the first param
            * The third part was generated with parameters where off_sets[0] was added from to first param
            * The fourth part was generated with parameters where off_sets[1] was subtracted from the second param
            * and so on
        The training step function that is set up will only work if the input has a shape:
        (n_points * n_same * ( 2 * n_params + 1), len(indices), n_channels)

        If multiple clippings are requested, the ordering is:
            * by value
            * by norm
            * by global norm

        Args:
            n_params (int): Number of underlying (cosmological) model parameters.
            n_same (int): Number of (uperturbed) summaries coming from the same parameter set, this is the same as the
                (base) batch size
            off_sets (np.ndarray): The off-sets used to perturb the original (fiducial) parameters. These are used as
                the finite differences in the computation of the Jacobian.
            n_points (int, optional): Number of different "fiducial" parameters. Defaults to 1.
            n_input (int, optional): Input dimension of the network, must be provided if the network is not a
                HealpyGCNN. Defaults to None.
            n_channels (int, optional): Number of channels. Defaults to 1.
            force_params_value (float, np.ndarray, optional): Either None or a set of parameters with shape
                (n_points, 1, n_output) which is used to compute a square loss of the unperturbed summaries. It is
                useful to set this for example to zeros such that the network does not produces arbitrary high summary
                values. Defaults to None, float inputs are broadcast to the appropriate shape.
            force_params_weight (float, optional): The weight of the square loss of force_params_value. Defaults to 1.0.
            jac_weight (float, optional): The weight of the Jacobian loss, which forces the Jacobian of the summaries
                to be close to unity (or identity matrix). Defaults to 100.0.
            jac_cond_weight (float, optional): If not None, this weight is used to add an additional loss using the
                matrix condition number of the jacobian. Defaults to None.
            cov_weight (bool, optional): If true, the jac weight will be used as cov_weight, i.e. loss cov mat will be
                forced to be close to the identity matrix. Note that there will be an additional term forcing the
                inverse of the covariance to be close to the identity as well since the cov is guaranteed to be square.
                This is the same as Luca's regularization term, but without the adaptive weight. Defaults to False.
            n_output (int, optional): Dimensionality of the summary statistic. Defaults to None, which corresponds to
                predictions.shape[-1].
            n_partial (np.ndarray, optional): To train only on a subset of parameters and not all underlying model
                parameter. Defaults to None which means the information inequality is minimized in a normal fashion.
                Note that due to the necessity of some algebraic manipulations n_partial == None and
                n_partial == n_params lead to slightly different behaviour. Defaults to None.
            weights (np.ndarray, optional): An 1d array of length n_points, used as weights in means of the different
                points. Defaults to None.
            no_correlations (bool, optional): Do not consider correlations between the parameter, this means that one
                tries to find an optimal summary (single value) for each underlying model parameter, only possible if
                n_output == n_params. Defaults to False.
            clip_by_value (tf.tensor, optional): Clip the gradients by given 1d array of values into the interval
                [value[0], value[1]]. Defaults to None (no clipping).
            clip_by_norm (tf.tensor, optional): Clip the gradients by norm. Defaults to None (no clipping).
            clip_by_global_norm (tf.tensor, optional): Clip the gradients by global norm. Defaults to None (no
                clipping).
            l2_norm_weight (float, optional): Weight for the L2 norm of the trainable weights. Defaults to None
                (no regularization).
            use_log_det (bool, optional): Use the log of the determinants in the information inequality, should be
                True. If False, the information inequality is not minimized in a proper manner and the training can
                become unstable. Defaults to True.
            tikhonov_regu (bool, optional): Use Tikhonov regularization of matrices e.g. to avoid vanishing
                determinants. This is the recommended regularization method as it allows the usage of some optimized
                routines. Defaults to False.
            eps (float, optional): A small positive value used for regularization of things like logs etc. This should
                only be increased if tikhonov_regu is used and a error is raised. Defaults to 1e-32.
            img_summary (bool, optional): Save image summaries of the Jacobian and the covariance. Defaults to False.

        Returns:
            callable: A callable function that performs one gradient descent step with respect to the delta loss.
        """
        # setup a loss function
        def loss_func(predictions):
            return delta_loss.delta_loss(
                predictions=predictions,
                n_params=n_params,
                n_same=n_same,
                off_sets=off_sets,
                # regularization
                force_params_value=force_params_value,
                force_params_weight=force_params_weight,
                jac_weight=jac_weight,
                jac_cond_weight=jac_cond_weight,
                cov_weight=cov_weight,
                # summary statistic
                n_output=n_output,
                n_partial=n_partial,
                weights=weights,
                no_correlations=no_correlations,
                # numerical stability
                use_log_det=use_log_det,
                tikhonov_regu=tikhonov_regu,
                eps=eps,
                # tf.summary
                training=True,
                summary_writer=self.summary_writer,
                img_summary=img_summary,
            )

        # get the backend float and input shape
        current_float = delta_loss._get_backend_floatx()

        n_batch = n_points * n_same * (2 * n_params + 1)
        if isinstance(self.network, HealpyGCNN):
            in_shape = (n_batch, len(self.network.indices_in), n_channels)
        else:
            if n_channels is None:
                in_shape = (n_batch, n_input)
            else:
                in_shape = (n_batch, n_input, n_channels)

        # tf function with fixed signature
        @tf.function(input_signature=[tf.TensorSpec(shape=in_shape, dtype=current_float)])
        def delta_train_step(input_batch):
            LOGGER.warning(f"Tracing delta_train_step")
            self.base_train_step(
                input_tensor=input_batch,
                loss_function=loss_func,
                input_labels=None,
                clip_by_value=clip_by_value,
                clip_by_norm=clip_by_norm,
                clip_by_global_norm=clip_by_global_norm,
                l2_norm_weight=l2_norm_weight,
            )

        LOGGER.info("Set up the traing step of the delta loss")
        self.delta_train_step = delta_train_step
