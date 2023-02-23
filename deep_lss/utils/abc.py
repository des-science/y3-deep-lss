# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
This file contains some useful functions concerning likelihoods and so on
"""

from deep_lss.utils import kernels

from multiprocessing import Pool, cpu_count

from scipy.spatial import Delaunay
from scipy.optimize import fsolve

import numpy as np
import mpmath as mp



def _estimate_ABC_posterior(predictions, fisher_matrix, observation, kernel, scale):
    """
    Estimates the ABC posterior and its uncertainty from parameter estimations
    :param predictions: 2D array of predictions with shape [n_preds, n_params]
    :param fisher_matrix: The (estimated) fisher matrix of the fiducial parameter with shape [n_params, n_params],
                          if None, take inv cov of predictions
    :param observation: The estimator of the observation with shape [1, n_params]
    :param kernel: The kernel to use for the ABC estimation, either sigmoid, gauss or logistic
    :param scale: The scale used for the kernel.
    :return: the ABC posterior estimate and its uncertainty
    """

    # check kernel consistency
    if kernel == "sigmoid":
        def kernel(d):
            return kernels.sigmoid_kernel(d, scale=scale, use_mp=True, return_mp=True)
    elif kernel == "gauss":
        def kernel(d):
            return kernels.gaussian_kernel(d, scale=scale, use_mp=True, return_mp=True)
    elif kernel == "logistic":
        def kernel(d):
            return kernels.logistic_kernel(d, scale=scale, use_mp=True, return_mp=True)
    else:
        raise ValueError(f"kernel type not understood {kernel}! kernel has to be either sigmoid, gauss or logistic.")

    with mp.workdps(100):
        # get the fisher matrix
        if fisher_matrix is None:
            fisher_matrix = np.linalg.inv(np.cov(predictions, rowvar=False))
        d = np.einsum("ij,aj->ai", fisher_matrix, predictions - observation)
        d = np.sqrt(np.sum(d * (predictions - observation), axis=1))
        # get the kernal vals
        kernel_vals = kernel(d)
        # mean
        mean = kernels.mp_mean(kernel_vals)
        # variance of the mean
        var = kernels.mp_std(kernel_vals) ** 2 / len(d)
        log_like = np.float(mp.log(mean))
        # we add an eps here
        log_var = np.float(var / (mean ** 2 + 1e-99))

    return log_like, log_var


def estimate_ABC_posterior(predictions, fisher_matrix, observation, kernel="sigmoid", scale=0.5, n_proc=None):
    """
    Estimates the ABC posterior and its uncertainty from parameter estimations
    :param predictions: 3D array of predictions with shape (n_cosmos, n_examples_per_cosmo, n_params)
    :param fisher_matrix: The (estimated) fisher matrix of the fiducial parameter with shape [n_params, n_params],
                          if None, take inv cov of predictions
    :param observation: The estimator of the observation with shape [1, n_params]
    :param kernel: The kernel to use for the ABC estimation, either sigmoid, gauss or logistic
    :param scale: The scale used for the kernel.
    :param n_proc: number of parallel process, defaults to mp.cpu_coun()
    :return: A 2D array of shape [n_cosmos, 2] containing the ABC posterior estimates and their variance
    """

    # number of procs
    if n_proc is None:
        n_proc = cpu_count()

    # make the obs 2d
    observation = np.atleast_2d(observation)

    # create the inputs
    inp = [(pred, fisher_matrix, observation, kernel, scale) for pred in predictions]

    # get the likelihood estimates
    loglike_estimates = []
    variance_estimates = []
    with mp.workdps(100):
        with Pool(processes=n_proc) as pool:
            for log_like, log_var in pool.starmap(_estimate_ABC_posterior, inp):
                loglike_estimates.append(log_like)
                variance_estimates.append(log_var)

    # concat to initial value
    Y_init = np.concatenate([np.asarray(loglike_estimates).reshape((-1, 1)),
                             np.asarray(variance_estimates).reshape((-1, 1))], axis=1)
    return Y_init


