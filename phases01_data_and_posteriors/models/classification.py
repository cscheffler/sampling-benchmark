"""
This module provides functions for sampling from the posteriors of various
classification models. The supported models are specified in the
CLASSIFICATION_MODEL_NAMES constant.
"""

import pymc3 as pm
import numpy as np
import theano
import theano.tensor as tt
from timeit import default_timer as timer

from .nn import build_shallow_nn
from . import MAX_NUM_SAMPLES
from .utils import reduce_data_dimension, subsample
from .sampling import sample_model

# Arguably, build_pm_gp_cov should go in some 3rd file like util
from .regression import build_pm_gp_cov

CLASSIFICATION_MODEL_NAMES = \
    [
     'softmax-linear-class',
     'shallow-nn-class',
     # 'gp-ExpQuad-class', 'gp-Exponential-class', 'gp-Matern32-class', 'gp-Matern52-class',
     # 'gp-RatQuad-class'
     ]


def sample_classification_model(model_name, X, y, num_samples=MAX_NUM_SAMPLES,
                                step=None, num_non_categorical=None,
                                raw_trace=False, random_seed=None):
    """
    Sample from the posteriors of any of the supported models

    Args:
        model_name: to specify which model to sample from
        X: data matrix
        y: targets
        num_samples: number points to sample from the model posterior
        num_non_categorical: number of non-categorical features

    Returns:
        samples

    Raises:
        ValueError: if the specified model name is not supported
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    X, y = subsample(X, y, model_name)
    d = X.shape[1]
    X = reduce_data_dimension(X, model_name)
    reduced_d = X.shape[1]
    if reduced_d < d:
        num_non_categorical = reduced_d

    model_name = model_name.replace('-class', '')
    print('Number of data:', X.shape[0])
    print('Data dimension:', X.shape[1])
    
    model = build_model(model_name, X, y, num_non_categorical)
    if 'nn' in model_name:
        return sample_model(model, step=step, advi=True, raw_trace=raw_trace, random_seed=random_seed)
    else:
        return sample_model(model, step=step, advi=False, raw_trace=raw_trace, random_seed=random_seed)


def build_model(model_name, X, y, num_non_categorical=None):
    """Build model for specified classificaiton model name"""
    if 'softmax-linear' == model_name:
        model = build_softmax_linear(X, y)
    elif 'shallow-nn' == model_name:
        model = build_shallow_nn(X, y, output='classification')
    elif 'gp-ExpQuad' == model_name:
        model = build_gpc(X, y, 'ExpQuad')
    elif 'gp-Exponential' == model_name:
        model = build_gpc(X, y, 'Exponential')
    elif 'gp-Matern32' == model_name:
        model = build_gpc(X, y, 'Matern32')
    elif 'gp-Matern52' == model_name:
        model = build_gpc(X, y, 'Matern52')
    elif 'gp-RatQuad' == model_name:
        model = build_gpc(X, y, 'RatQuad')
    else:
        raise ValueError('Unsupported model: {}\nSupported models: {}'
                         .format(model_name, CLASSIFICATION_MODEL_NAMES))
    return model


def build_softmax_linear(X, y, force_softmax=False):
    """
    Sample from Bayesian Softmax Linear Regression
    """
    num_features = X.shape[1]
    num_classes = len(np.unique(y))
    logistic_regression = num_classes == 2
    Xt = theano.shared(X)
    
    if logistic_regression and not force_softmax:
        print('running logistic regression')
        with pm.Model() as model:
            W = pm.Normal('W', 0, sd=1e6, shape=num_features)
            b = pm.Flat('b')
            logit = Xt.dot(W) + b
            p = tt.nnet.sigmoid(logit)
            observed = pm.Bernoulli('obs', p=p, observed=y)
    else:
        with pm.Model() as model:
            W = pm.Normal('W', 0, sd=1e6, shape=(num_features, num_classes))
            b = pm.Flat('b', shape=num_classes)
            logit = Xt.dot(W) + b
            p = tt.nnet.softmax(logit)
            observed = pm.Categorical('obs', p=p, observed=y)
    return model


def sample_shallow_nn_class(X, y, num_samples=MAX_NUM_SAMPLES):
    """
    Sample from shallow Bayesian neural network, using variational inference.
    Uses Categorical likelihood.
    """
    return sample_shallow_nn(X, y, 'classification')


def build_gpc(X, y, cov_f='ExpQuad'):
    """Sample from Gaussian Process"""
    # TODO also implement version that uses Elliptical slice sampling
    N, D = X.shape

    with pm.Model() as model_gp:
        # uninformative prior on the function variance
        log_s2_f = pm.Uniform('log_s2_f', lower=-10.0, upper=5.0)
        s2_f = pm.Deterministic('s2_f', tt.exp(log_s2_f))

        # covariance functions for the function f and the noise
        cov_func = s2_f * build_pm_gp_cov(D, cov_f)

        # Specify the GP.  The default mean function is `Zero`.
        f = pm.gp.GP('f', cov_func=cov_func, X=X, sigma=1e-6)
        # Smash to a probability
        f_transform = pm.invlogit(f)

        # Add the observations
        pm.Binomial('y', observed=y, n=np.ones(N), p=f_transform, shape=N)

    return model_gp
