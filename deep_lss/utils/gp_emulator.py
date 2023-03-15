# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created March 2023
Author: Arne Thomsen, Janis Fluri

Implements a Gaussian Process regrossor, which is used like an emulator. For details, see Section F in
https://arxiv.org/pdf/2107.09002.pdf

Adapted from
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/gp_emulator.py
by Janis Fluri
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import GPyOpt
import gpflow

from gpflow.optimizers import NaturalGradient
from gpflow.utilities import print_summary

from time import time
from tqdm import tqdm
import os

# tf.config.run_functions_eagerly(True)

class HeteroskedasticGaussian(gpflow.likelihoods.Likelihood):
    """
    Likelihood for varying noise amplitude in the data, see the docs of GPflow
    https://gpflow.github.io/GPflow/2.4.0/notebooks/advanced/heteroskedastic.html#Heteroskedastic-Regression
    """

    def __init__(self, **kwargs):
        # this likelihood expects a single latent function F, and two columns in the data matrix Y:
        super().__init__(latent_dim=1, observation_dim=2, **kwargs)

    def _log_prob(self, F, Y):
        # log_prob is used by the quadrature fallback of variational_expectations and predict_log_density.
        # Because variational_expectations is implemented analytically below, this is not actually needed,
        # but is included for pedagogical purposes.
        # Note that currently relying on the quadrature would fail due to https://github.com/GPflow/GPflow/issues/966
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        print(F.shape)
        return gpflow.logdensities.gaussian(Y, F, NoiseVar)

    # def _variational_expectations(self, Fmu, Fvar, Y):
    def _variational_expectations(self, X, Fmu, Fvar, Y):
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        Fmu = Fmu[:, 0]
        Fvar = Fvar[:, 0]
        return (
            -0.5 * np.log(2 * np.pi) - 0.5 * tf.math.log(NoiseVar) - 0.5 * (tf.math.square(Y - Fmu) + Fvar) / NoiseVar
        )

    # The following two methods are abstract in the base class and therefore need to be implemented even if not used.

    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def _predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError


class VGP_Emu:
    # default types
    default_np_float = gpflow.default_float()
    if gpflow.default_float() is np.float32:
        default_tf_float = tf.float32
    else:
        default_tf_float = tf.float64
    default_np_int = gpflow.default_int()
    if gpflow.default_int() is np.int32:
        default_tf_int = tf.int32
    else:
        default_tf_int = tf.int64

    def __init__(
        self,
        objective=None,
        space=None,
        N_init=20,
        X_init=None,
        Y_init=None,
        normalize_X=True,
        normalize_Y=True,
        mean_only=False,
        alpha=0.01,
        kern="matern52",
        num_restarts=10,
        verbosity=0,
        max_opt_iter=1000,
        full_restart=False,
        ARD=False,
        learning_rate=1e-4,
        parameter_noise_scale=0.1,
        minimum_variance=1e-3,
    ):
        """
        An class that fits a Gaussian process to a given objective function
        :param objective: function used for the fitting (needs to estimate the noise as well!)
        :param space: a GPy space for the prior
        :param N_init: number of initial points
        :param X_init: initial points in space (if set N_init is ignored)
        :param Y_init: optionial with set with X_init the objective is not called on X_init
        :param normalize_X: Normalize the input parameter -> rotate to eigen values, scale and strech according to Raoul
        :param normalize_Y: normalize the Y coordinate to have zero mean and unit variance (standard)
        :param mean_only: normalize Y only such that it has zero mean but leave std as is
        :param alpha: alpha value in the acquisition function of Raul's paper
        :param kern: kernel type, currently only Matern52, Exponential, SquaredeEponential
        :param num_restarts: number of restarts for each GP optimization
        :param verbosity: 0 -> print minimal output, higher value = more output
        :param max_opt_iter: maximum iteration for a single GP optimization
        :param full_restart: ignore the current kernel values for the next optimization
        :param ARD: Auto Relevance Determination, use a lengthscale in the kernel for each dimension of the problem
        :param learning_rate: learning rate for the Adam optimizer used for the optimization
        :param parameter_noise_scale: noise std that is added to the parameter for optimization
        :param minimum_variance: minimum of the allowed variance estimate, choosing this too small leads to numerical
        instabilities.
        """

        # some sanity checks
        if (objective is None or space is None) and (X_init is None and Y_init is None):
            raise ValueError("If there is no initial dataset, one has to provide an objective function and a space!")

        self.objective = objective
        self.verbosity = verbosity
        # how to start
        if X_init is None:
            initial_design = GPyOpt.experiment_design.initial_design("latin", space, N_init)
            initial_Y = objective(initial_design)
        elif Y_init is None:
            initial_design = X_init
            initial_Y = objective(initial_design)
        else:
            initial_design = X_init
            initial_Y = Y_init

        # we need to split off the variance estimates
        initial_Y, initial_var = np.split(initial_Y, axis=1, indices_or_sections=2)

        # tfp prior
        self.space = space
        if self.space is not None:
            a_min = np.asarray(self.space.get_bounds(), dtype=self.default_np_float).T[0]
            a_max = np.asarray(self.space.get_bounds(), dtype=self.default_np_float).T[1]
            self.tfp_prior = tfp.distributions.Uniform(low=a_min, high=a_max)

        # normalize
        if normalize_Y:
            self.Y_mean = np.mean(initial_Y)
            if mean_only:
                self.Y_std = 1.0
            else:
                self.Y_std = np.std(initial_Y)
            self.Y_all = (initial_Y - self.Y_mean) / self.Y_std
        else:
            self.Y_mean = 0.0
            self.Y_std = 1.0
            self.Y_all = initial_Y
        self.normalize_Y = normalize_Y
        self.mean_only = mean_only

        # now we need to take care of the variance estimates
        self.var_estimates = initial_var / self.Y_std**2

        # normalization
        self.normalize_X = normalize_X
        self.params, self.rot_mat, self.rot_mean, self.rot_std = self.normalize_params(
            initial_design, norm=self.normalize_X
        )

        # kernel
        self.dims = int(X_init.shape[-1])
        self.kern_type = kern.lower()
        if ARD:
            lengthscales = [1.0 for _ in range(self.dims)]
        else:
            lengthscales = 1.0
        if self.kern_type == "matern52":
            self.kern = gpflow.kernels.Matern52(lengthscales=lengthscales)
        elif self.kern_type == "exponential":
            self.kern = gpflow.kernels.Exponential(lengthscales=lengthscales)
        elif self.kern_type == "squaredexponential":
            self.kern = gpflow.kernels.SquaredExponential(lengthscales=lengthscales)
        else:
            raise IOError("Unkown kernel")
        self.lengthscale_shape = self.kern.lengthscales.shape

        if num_restarts is None or num_restarts < 1:
            print("Number of restarts is set to 1!")
            self.num_restarts = self.default_np_int(1)
        else:
            self.num_restarts = self.default_np_int(num_restarts)

        # get the likelihood
        # self.likelihood = HeteroskedasticGaussian()
        # TODO
        self.likelihood = HeteroskedasticGaussian(input_dim=X_init.shape[1])

        # model (if you get a matrix inversion error here increase number of initial params)
        self.minimum_variance = minimum_variance
        data = np.concatenate([self.Y_all, np.maximum(self.var_estimates, self.minimum_variance)], axis=1)
        self.model = gpflow.models.VGP(
            (self.params.astype(self.default_np_float), data.astype(self.default_np_float)),
            kernel=self.kern,
            likelihood=self.likelihood,
            num_latent_gps=1,
        )

        # We turn off training for q as it is trained with natgrad
        gpflow.utilities.set_trainable(self.model.q_mu, False)
        gpflow.utilities.set_trainable(self.model.q_sqrt, False)

        # summary
        print_summary(self.model)

        # save params
        self.learning_rate = learning_rate
        self.parameter_noise_scale = parameter_noise_scale
        self.max_opt_iter = max_opt_iter
        self.full_restart = full_restart
        # self.optimize_model()

        # for acquisition
        self.current_transform = lambda x: self.transform_params(x, self.rot_mat, self.rot_mean, self.rot_std)
        self.alpha = alpha

    def train_model(self, n_iters):
        """
        Optimizes the model for n_iters
        :param n_iters: number of iterations for optimization
        :return: a list of losses of each step with length n_iters
        """

        @tf.function
        def objective_closure():
            return self.model.training_loss()

        natgrad = NaturalGradient(gamma=1.0)
        adam = tf.optimizers.Adam(self.learning_rate)

        print("Training the VGP model params...", flush=True)
        losses = []
        with tqdm(range(n_iters), total=n_iters) as pbar:
            for _ in pbar:
                natgrad.minimize(objective_closure, [(self.model.q_mu, self.model.q_sqrt)])
                adam.minimize(objective_closure, self.model.trainable_variables)
                loss = objective_closure().numpy()
                losses.append(loss)
                pbar.set_postfix(loss_val=loss, refresh=False)

        return losses

    def _readout_params(self):
        """
        Reads out the params of the model and returns them as tupel of arrays
        :return: tupel of params
        """

        params = (
            self.model.kernel.variance.numpy(),
            self.model.kernel.lengthscales.numpy(),
            self.model.q_mu.numpy(),
            self.model.q_sqrt.numpy(),
        )

        return params

    def _set_params(self, params):
        """
        Sets the model params to a given tupel of array
        :param params: params (tupel of arrays)
        """
        self.model.kernel.variance.assign(params[0])
        self.model.kernel.lengthscales.assign(params[1])
        self.model.q_mu.assign(params[2])
        self.model.q_sqrt.assign(params[3])

    def optimize_model(self, scale=1.0):
        """
        Optimizes the model for a given number of restarts and chooses the best result
        :param scale: std of the normal distribution used to draw new params
        """

        func_vals = []
        model_params = []

        # read out the original params
        original_params = self._readout_params()

        for i in range(self.num_restarts):
            # we need to create a new optimizer since Adam has params itself
            self.opt = tf.optimizers.Adam(self.learning_rate)

            try:
                # assign new staring vals
                if self.full_restart:
                    # This is used in GPy opt if no prior is specified (see model.randomize() defined in paramz pack)
                    self.model.kernel.variance.assign(
                        tf.maximum(
                            tf.random.normal(
                                shape=(), dtype=self.default_tf_float, mean=scale, stddev=self.parameter_noise_scale
                            ),
                            tf.constant(0.1, dtype=self.default_tf_float),
                        )
                    )
                    self.model.kernel.lengthscales.assign(
                        tf.maximum(
                            tf.random.normal(
                                shape=self.lengthscale_shape,
                                dtype=self.default_tf_float,
                                mean=scale,
                                stddev=self.parameter_noise_scale,
                            ),
                            tf.constant(0.1, dtype=self.default_tf_float),
                        )
                    )
                    self.model.q_mu.assign(tf.zeros_like(self.model.q_mu))
                    self.model.q_sqrt.assign(
                        tf.eye(len(original_params[2]), batch_shape=[1], dtype=self.default_tf_float)
                    )
                else:
                    self.model.kernel.variance.assign(
                        tf.maximum(
                            original_params[0]
                            + tf.random.normal(
                                shape=(), dtype=self.default_tf_float, stddev=self.parameter_noise_scale
                            ),
                            tf.constant(
                                0.1,
                                dtype=self.default_tf_float,
                            ),
                        )
                    )
                    self.model.kernel.lengthscales.assign(
                        tf.maximum(
                            original_params[1]
                            + tf.random.normal(
                                shape=self.lengthscale_shape,
                                dtype=self.default_tf_float,
                                stddev=self.parameter_noise_scale,
                            ),
                            tf.constant(0.1, dtype=self.default_tf_float),
                        )
                    )
                    self.model.q_mu.assign(original_params[2])
                    self.model.q_sqrt.assign(original_params[3])

                # now we optimize
                losses = self.train_model(self.max_opt_iter)

                # we append the final loss value
                func_vals.append(losses[-1])
                model_params.append(self._readout_params())
                if self.verbosity > 0:
                    print("Optimization {}: achieved {} with params {}".format(i, func_vals[-1], model_params[-1]))
            except:
                print("Failed Optimization {}".format(i))

        # set to minimum
        min_index = np.argmin(func_vals)
        self._set_params(model_params[min_index])

    @classmethod
    def normalize_params(self, params, norm=True):
        """
        Normalized the params according to Raoul's prescription
        :param params: 2D array with shape (N, d) of params to normalize
        :param norm: If False, the params won't be transformed and the identity will be returned
        :return: The new params, as well as all parts necessary to perform the transformation
        """
        # make the params linearly uncorrelated
        cov = np.cov(params, rowvar=False)
        # eigenvals and vecs
        w, v = np.linalg.eig(cov)
        # rot mat is v.T
        rot_mat = v.T
        # dot prod
        rot_params = np.einsum("ij,aj->ai", rot_mat, params)
        # mean
        rot_mean = np.mean(rot_params, axis=0, keepdims=True)
        # std (ddof of np.cov for consistency)
        rot_std = np.std(rot_params, axis=0, keepdims=True, ddof=1)
        # normalize
        new_params = (rot_params - rot_mean) / rot_std

        if norm:
            return new_params, rot_mat, rot_mean, rot_std

        else:
            return params, np.eye(rot_mat.shape[0]), np.zeros_like(rot_mean), np.ones_like(rot_std)

    @classmethod
    def transform_params(self, params, rot_mat, rot_mean, rot_std):
        """
        Normalizes params given the rot, shift and scale
        """
        rot_params = np.einsum("ij,aj->ai", rot_mat, params)
        new_params = (rot_params - rot_mean) / rot_std
        return new_params

    @classmethod
    def unnormalize_params(self, params, rot_mat, rot_mean, rot_std):
        """
        Removes normalization
        """
        new_params = params * rot_std + rot_mean
        # inverse rotation
        new_params = np.einsum("ij,aj->ai", rot_mat.T, new_params)
        return new_params

    def save_model(self, save_dir):
        """
        Saves the model as compressed numpy file
        :param save_dir: directory to store the emu
        """

        # save the kernel
        q_mu = self.model.q_mu.numpy()
        q_sqrt = self.model.q_sqrt.numpy()
        # save model params
        save_dict = {
            self.kern_type + "_var": self.model.kernel.variance.numpy(),
            self.kern_type + "_scale": self.model.kernel.lengthscales.numpy(),
            "q_mu": q_mu,
            "q_sqrt": q_sqrt,
            "min_var": np.array([self.minimum_variance], dtype=self.default_np_float),
        }
        # save params
        params = self.unnormalize_params(self.params, self.rot_mat, self.rot_mean, self.rot_std)
        save_dict.update({"norm_params": self.normalize_X, "params": params})

        # save the evals
        Y_all = self.Y_all * self.Y_std + self.Y_mean
        var_estimates = self.var_estimates * self.Y_std**2
        save_dict.update(
            {
                "normalize_Y": self.normalize_Y,
                "mean_only": self.mean_only,
                "Y_all": Y_all,
                "var_estimates": var_estimates,
            }
        )

        # save everyting
        np.savez(os.path.join(save_dir, "gp_emu.npz"), **save_dict)

    def get_noiseless_predictor(self):
        # get the relevant stuff
        Y_std = self.Y_std
        Y_mean = self.Y_mean

        transform = lambda x: self.transform_params(x, self.rot_mat, self.rot_mean, self.rot_std)

        model = self.model

        def noiseless_predictor(X):
            X = transform(X)
            preds = model.predict_f(X)[0].numpy()
            return preds * Y_std + Y_mean

        return noiseless_predictor

    @classmethod
    def restore_noiseless_predictor(self, restore_path, numpy=True):
        """
        Restores the noiseless predictor from a saved model
        :param restore_path: Path to the directory
        :param numpy: If False, return a TF function instead of a normal python function
        :return: the restored noiseless predictor
        """

        # load the data
        data = np.load(os.path.join(restore_path, "gp_emu.npz"))
        # params
        params = data["params"]
        params, rot_mat, rot_mean, rot_std = self.normalize_params(params, norm=np.all(data["norm_params"]))
        transform = lambda x: self.transform_params(x, rot_mat, rot_mean, rot_std)

        # restore kernel
        if "matern52_var" in data.files:
            var = data["matern52_var"]
            scale = data["matern52_scale"]
            print("restoring matern52 kernel with variance {} and lengthscale {}...".format(var, scale))
            kernel = gpflow.kernels.Matern52(variance=var, lengthscales=scale)
        elif "exponential_var" in data.files:
            var = data["exponential_var"]
            scale = data["exponential_scale"]
            print("restoring exponential kernel with variance {} and lengthscale {}...".format(var, scale))
            kernel = gpflow.kernels.Exponential(variance=var, lengthscales=scale)
        elif "squaredexponential_var" in data.files:
            var = data["squaredexponential_var"]
            scale = data["squaredexponential_scale"]
            print("restoring squaredexponential kernel with variance {} and lengthscale {}...".format(var, scale))
            kernel = gpflow.kernels.SquaredExponential(variance=var, lengthscales=scale)

        # restore the other model params
        q_mu = data["q_mu"]
        q_sqrt = data["q_sqrt"]
        var_estimates = data["var_estimates"]
        min_variance = data["min_var"]
        Y_all = data["Y_all"]

        # restore evals
        if np.all(data["normalize_Y"]):
            if np.all(data["mean_only"]):
                Y_mean = np.mean(Y_all)
                Y_std = 1.0
                Y_all = (Y_all - Y_mean) / Y_std
            else:
                Y_mean = np.mean(Y_all)
                Y_std = np.std(Y_all)
                Y_all = (Y_all - Y_mean) / Y_std
        # no norm
        else:
            Y_mean = 0.0
            Y_std = 1.0

        # build the model
        likelihood = HeteroskedasticGaussian()
        data = np.concatenate([Y_all, np.maximum(var_estimates, min_variance)], axis=1)
        model = gpflow.models.VGP(
            (params.astype(self.default_np_float), data.astype(self.default_np_float)),
            kernel=kernel,
            likelihood=likelihood,
            num_latent_gps=1,
        )

        # assign variables
        model.q_mu.assign(q_mu)
        model.q_sqrt.assign(q_sqrt)

        if numpy:

            def noiseless_predictor(X):
                X = transform(X)
                preds = model.predict_f(X)[0].numpy()
                return preds * Y_std + Y_mean

            return noiseless_predictor

        else:
            if self.default_tf_float is tf.float64:
                print("Warning: tf function should be used with float32")

            rot_mat = tf.constant(rot_mat, dtype=self.default_tf_float)
            rot_mean = tf.constant(rot_mean, dtype=self.default_tf_float)
            rot_std = tf.constant(rot_std, dtype=self.default_tf_float)
            Y_mean = tf.constant(Y_mean, dtype=self.default_tf_float)
            Y_std = tf.constant(Y_std, dtype=self.default_tf_float)

            # tf function
            @tf.function(input_signature=[tf.TensorSpec(shape=(None, params.shape[1]), dtype=self.default_tf_float)])
            def noiseless_predictor(X):
                rot_params = tf.einsum("ij,aj->ai", rot_mat, X)
                X = (rot_params - rot_mean) / rot_std
                preds = model.predict_f(X)[0]
                return preds * Y_std + Y_mean

            return noiseless_predictor
