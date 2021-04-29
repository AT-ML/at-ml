"""
The :mod:'atml.irt' module contains a set of statistical models based on the Item Response Theory.
"""
# Author: Hao Song (nuacesh@gmail.com)
# License: BSD-3

import numpy

import scipy.stats

import tensorflow as tf

from adhoc_opt.optimiser import parameter_update

tf.compat.v1.enable_eager_execution()

pi = tf.constant(numpy.pi, dtype='float32')

eps = 1e-6


class Beta_3_IRT:
    """

    The class for the Beta-3 IRT model.

    """

    def __init__(self):
        """

        """

        self.irt_type = 'beta3'

        self.N_dataset = None

        self.N_model = None

        self.logit_theta = None

        self.logit_delta = None

        self.log_a = None

        self.parameter = None

    def fit(self, dataset_list, dataset, model_list, model, measure):
        """
        Fit the IRT model to a collection of testing results.

        Parameters
        ----------
        dataset_list: list
            The list contains all the reference names of the datasets.

        dataset: numpy.ndarray
            A (n_experiment, ) numpy array contains the dataset index of each experiment

        model_list: list
            The list contains all the reference names of the models.

        model: numpy.ndarray
            A (n_experiment, ) numpy array contains the model index of each experiment.

        measure: numpy.ndarray
            A (n_experiment, ) numpy array contains the performance measurement of each experiment.

        """

        self.N_dataset = len(dataset_list)

        self.N_model = len(model_list)

        self.logit_theta = numpy.zeros(self.N_model)

        self.logit_delta = numpy.zeros(self.N_dataset)

        self.log_a = numpy.zeros(self.N_dataset)

        parameter = tf.cast(numpy.hstack([self.logit_theta, self.logit_delta, self.log_a]), 'float32')

        measure[measure <= eps] = eps

        measure[measure >= (1.0 - eps)] = 1 - eps

        data = numpy.hstack([model.reshape(-1, 1), dataset.reshape(-1, 1), measure.reshape(-1, 1)])

        extra_args = ('beta3', (self.N_dataset, self.N_model))

        parameter = parameter_update(theta_0=tf.Variable(parameter), data=data, extra_args=extra_args,
                                     obj=get_obj, obj_g=get_obj_g,
                                     lr=1e-3,
                                     batch_size=1,
                                     val_size=None, factr=0.0, tol=8192,
                                     max_batch=int(1e8), plot_loss=False)

        self.parameter = parameter

        self.logit_theta = parameter[:self.N_model]

        self.logit_delta = parameter[self.N_model:self.N_model + self.N_dataset]

        self.log_a = parameter[self.N_model + self.N_dataset:self.N_model + 2 * self.N_dataset]

    def predict(self, dataset, model):
        """
        Predict the expected response for a set of combinations between different models and datasets

        Parameters
        ----------
        dataset: numpy.ndarray
            A (n_test, ) numpy array contains the dataset index of each testing experiment.

        model: numpy.ndarray
            A (n_test, ) numpy array contains the model index of each testing experiment.

        Returns
        ----------
        E: numpy.ndarray
            A (n_test, ) numpy array contains the expected performance measurement of each testing experiment.

        """

        idx = numpy.arange(0, len(dataset), 8)
        
        E = []
        
        for i in range(0, len(idx)):
            E.append(get_E_beta_3(self.parameter, 
                                  model[idx[i]:idx[i]+8], 
                                  dataset[idx[i]:idx[i]+8], 
                                  self.N_dataset, 
                                  self.N_model))

        E = numpy.hstack(E)

        return E

    def curve(self, d_id):
        """
        Generate the item characteristic curve (expectation, 0.75, 0.5, 0.25 percentile) for a given dataset

        Parameters
        ----------
        d_id: int
            The index of the given dataset.

        Returns
        ----------
        E: numpy.ndarray
            The curve values of the expectation.

        E_up: numpy.ndarray
            The curve values of the 0.75 percentile.

        E_mid: numpy.ndarray
            The curve values of the 0.5 percentile (median).

        E_low: numpy.ndarray
            The curve values of the 0.25 percentile.

        """
        E, E_up, E_mid, E_low = get_curve_beta3(self.parameter, d_id, self.N_dataset, self.N_model)

        return E, E_up, E_mid, E_low


class Logistic_IRT:
    """

    The class for the Logistic IRT model.

    """

    def __init__(self):
        """

        """

        self.irt_type = 'logistic'

        self.N_dataset = None

        self.N_model = None

        self.parameter = None

        self.theta = None

        self.delta = None

        self.log_a = None

        self.log_s2 = None

    def fit(self, dataset_list, dataset, model_list, model, measure):
        """
        Fit the IRT model to a collection of testing results.

        Parameters
        ----------
        dataset_list: list
            The list contains all the reference names of the datasets.

        dataset: numpy.ndarray
            A (n_experiment, ) numpy array contains the dataset index of each experiment

        model_list: list
            The list contains all the reference names of the models.

        model: numpy.ndarray
            A (n_experiment, ) numpy array contains the model index of each experiment.

        measure: atml.Measure
            A (n_experiment, ) numpy array contains the performance measurement of each experiment.

        """

        self.N_dataset = len(dataset_list)

        self.N_model = len(model_list)

        self.theta = numpy.random.randn(self.N_model)

        self.delta = numpy.random.randn(self.N_dataset)

        self.log_a = numpy.random.randn(self.N_dataset)

        self.log_s2 = numpy.ones(self.N_dataset) * numpy.log(1e2)

        parameter = tf.cast(numpy.hstack([self.theta, self.delta, self.log_a, self.log_s2]), 'float32')

        measure[measure <= eps] = eps

        measure[measure >= (1.0 - eps)] = 1 - eps

        data = numpy.hstack([model.reshape(-1, 1), dataset.reshape(-1, 1), measure.reshape(-1, 1)])

        extra_args = ('logistic', (self.N_dataset, self.N_model))

        parameter = parameter_update(theta_0=tf.Variable(parameter), data=data, extra_args=extra_args,
                                     obj=get_obj, obj_g=get_obj_g,
                                     lr=1e-3,
                                     batch_size=1, val_size=None, factr=0.0, tol=8192,
                                     max_batch=int(1e8), plot_loss=False)

        self.parameter = parameter

        self.theta = parameter[:self.N_model]

        self.delta = parameter[self.N_model:self.N_model + self.N_dataset]

        self.log_a = parameter[self.N_model + self.N_dataset:self.N_model + 2 * self.N_dataset]

        self.log_s2 = parameter[self.N_model + 2 * self.N_dataset:self.N_model + 3 * self.N_dataset]

    def predict(self, dataset, model):
        """
        Predict the expected response for a set of combinations between different models and datasets

        Parameters
        ----------
        dataset: numpy.ndarray
            A (n_test, ) numpy array contains the dataset index of each testing experiment.

        model: numpy.ndarray
            A (n_test, ) numpy array contains the model index of each testing experiment.

        Returns
        ----------
        E: numpy.ndarray
            A (n_test, ) numpy array contains the expected performance measurement of each testing experiment.

        """
        
        idx = numpy.arange(0, len(dataset), 8)
        
        E = []
        
        for i in range(0, len(idx)):
            E.append(get_E_logistic(self.parameter, 
                                    model[idx[i]:idx[i]+8], 
                                    dataset[idx[i]:idx[i]+8], 
                                    self.N_dataset, 
                                    self.N_model))

        return numpy.hstack(E)

    def curve(self, d_id):
        """
        Generate the item characteristic curve (expectation, 0.75, 0.5, 0.25 percentile) for a given dataset

        Parameters
        ----------
        d_id: int
            The index of the given dataset.

        Returns
        ----------
        E: numpy.ndarray
            The curve values of the expectation.

        E_up: numpy.ndarray
            The curve values of the 0.75 percentile.

        E_mid: numpy.ndarray
            The curve values of the 0.5 percentile (median).

        E_low: numpy.ndarray
            The curve values of the 0.25 percentile.

        """
        E, E_up, E_mid, E_low = get_curve_logistic(self.parameter, d_id, self.N_dataset, self.N_model)
        return E, E_up, E_mid, E_up


def get_curve_logistic(parameter, did, N_data, N_flow):
    """
    Generate the item characteristic curve (expectation, 0.75, 0.5, 0.25 percentile) for a given dataset

    Parameters
    ----------
    parameter: numpy.ndarray
        The estimated parameters for the Logistic IRT model.

    did: int
        The index of the given dataset.

    N_data: int
        The total number of datasets in the IRT model.

    N_flow: int
        The total number of models in the IRT model.

    Returns
    ----------
    E: numpy.ndarray
        The curve values of the expectation.

    E_up: numpy.ndarray
        The curve values of the 0.75 percentile.

    E_mid: numpy.ndarray
        The curve values of the 0.5 percentile (median).

    E_low: numpy.ndarray
        The curve values of the 0.25 percentile.

    """

    theta_edge = numpy.max(numpy.abs(parameter[:N_flow]))

    theta = tf.cast(numpy.linspace(-theta_edge, theta_edge, 128), dtype='float32')

    did = (numpy.ones(128) * did).astype('int')

    delta = tf.gather(parameter[N_flow:N_flow + N_data], did, axis=0)

    a = tf.math.exp(tf.gather(parameter[N_flow + N_data:N_flow + 2 * N_data], did, axis=0))

    log_s2 = tf.gather(parameter[N_flow + 2 * N_data:N_flow + 3 * N_data], did, axis=0)

    s2 = tf.math.exp(log_s2)

    s = tf.sqrt(s2)

    mu = - a * (theta.numpy().ravel() - delta.numpy().ravel())

    E = 1 / (1 + numpy.exp((mu / numpy.sqrt(1 + numpy.pi * numpy.square(s.numpy()) / 8))))

    E_up = 1 / (1 + numpy.exp(scipy.stats.norm.ppf(q=0.75, loc=mu, scale=s.numpy())))

    E_mid = 1 / (1 + numpy.exp(scipy.stats.norm.ppf(q=0.5, loc=mu, scale=s.numpy())))

    E_low = 1 / (1 + numpy.exp(scipy.stats.norm.ppf(q=0.25, loc=mu, scale=s.numpy())))

    return E, E_up, E_mid, E_low


def ml_logistic_obj(parameter, fid, did, measure, N_data, N_flow):
    """
    The log-likelihood objective function of the Logistic IRT model

    Parameters
    ----------
    parameter: tensorflow.Variable
        The parameters for the Logistic IRT model.

    fid: tf.Tensor
        A (n_batch, ) tensorflow array contains the dataset index of each experiment.

    did: tf.Tensor
        A (n_batch, ) tensorflow array contains the model index of each experiment.

    measure: tf.Tensor
        A (n_batch, ) tensorflow array contains the performance measurement of each experiment.

    N_data: int
        The total number of datasets in the IRT model.

    N_flow: int
        The total number of models in the IRT model.

    Returns
    ----------
    L: tf.Tensor
        The averaged log-likelihood of the IRT model.

    """
    
    measure = tf.reshape(measure, -1)

    logit_measure = tf.math.log((1 - measure) / measure)

    theta = tf.gather(parameter[:N_flow], fid, axis=0)

    delta = tf.gather(parameter[N_flow:N_flow + N_data], did, axis=0)

    a = tf.math.exp(tf.gather(parameter[N_flow + N_data:N_flow + 2 * N_data], did, axis=0))

    log_s2 = tf.gather(parameter[N_flow + 2 * N_data:N_flow + 3 * N_data], did, axis=0)

    s2 = tf.math.exp(log_s2) + tf.constant(eps, dtype='float32')

    mu = - a * (theta - delta)
    
    diff = logit_measure - mu
    
    exp = tf.math.exp(- 0.5 * tf.math.square(diff) / s2)
    
    sample_lik = 1 / (tf.math.sqrt(2 * pi * s2)) * exp * (1 / (measure * (1 - measure)))
    
    loglik = tf.math.log(sample_lik + tf.constant(eps, dtype='float32'))

    # loglik = 0.5 * tf.math.log(1 / (2 * pi)) + 0.5 * tf.math.log(1 / s2) \
    #     - 0.5 * tf.math.square(logit_measure - mu) / s2 \
    #     + tf.math.log(1 / (measure * (1 - measure)))

    L = - tf.reduce_mean(loglik)
    
    return L


def get_E_logistic(parameter, fid, did, N_data, N_flow):
    """
    Predict the expected response for a set of combinations between different models and datasets

    Parameters
    ----------
    parameter: numpy.ndarray
        The estimated parameters for the Logistic IRT model.

    fid: numpy.ndarray
        A (n_test, ) numpy array contains the model index of each testing experiment.

    did: numpy.ndarray
        A (n_test, ) numpy array contains the dataset index of each testing experiment.

    N_data: int
        The total number of datasets in the IRT model.

    N_flow: int
        The total number of models in the IRT model.

    Returns
    ----------
    E: numpy.ndarray
        A (n_test, ) numpy array contains the expected performance measurement of each testing experiment.

    """

    theta = tf.gather(parameter[:N_flow], fid, axis=0)

    delta = tf.gather(parameter[N_flow:N_flow + N_data], did, axis=0)

    a = tf.math.exp(tf.gather(parameter[N_flow + N_data:N_flow + 2 * N_data], did, axis=0))

    log_s2 = tf.gather(parameter[N_flow + 2 * N_data:N_flow + 3 * N_data], did, axis=0)
    
    s2 = tf.math.exp(log_s2) + tf.constant(eps, dtype='float32')

    s = tf.math.sqrt(s2)

    mu = - a * (theta - delta)

    samples = scipy.stats.norm.rvs(loc=mu, scale=s, size=[1024, len(mu)])

    E = numpy.mean(1 / (1 + numpy.exp(samples)), axis=0)

    return E


def get_curve_beta3(parameter, did, N_data, N_flow):
    """
    Generate the item characteristic curve (expectation, 0.75, 0.5, 0.25 percentile) for a given dataset

    Parameters
    ----------
    parameter: numpy.ndarray
        The estimated parameters for the Beta-3 IRT model.

    did: int
        The index of the given dataset.

    N_data: int
        The total number of datasets in the IRT model.

    N_flow: int
        The total number of models in the IRT model.

    Returns
    ----------
    E: numpy.ndarray
        The curve values of the expectation.

    E_up: numpy.ndarray
        The curve values of the 0.75 percentile.

    E_mid: numpy.ndarray
        The curve values of the 0.5 percentile (median).

    E_low: numpy.ndarray
        The curve values of the 0.25 percentile.

    """

    theta = tf.cast(numpy.linspace(1e-6, 1.0-1e-6, 128), dtype='float32')

    did = (numpy.ones(128) * did).astype('int')

    logit_delta = tf.cast(tf.gather(parameter[N_flow:N_flow + N_data], did, axis=0), 'float32')

    a = tf.cast(tf.math.exp(tf.gather(parameter[N_flow + N_data:N_flow + 2 * N_data], did, axis=0)), 'float32')

    delta = tf.clip_by_value(1 / (1 + tf.math.exp(logit_delta)),
                             tf.constant(eps, dtype='float32'),
                             tf.constant(1 - eps, dtype='float32'))

    alpha = tf.math.pow(theta / delta, a) + tf.constant(eps, dtype='float32')

    beta = tf.math.pow((1 - theta) / (1 - delta), a) + tf.constant(eps, dtype='float32')

    E = alpha.numpy() / (alpha.numpy() + beta.numpy())

    E_up = scipy.stats.beta.ppf(q=0.75, a=alpha.numpy(), b=beta.numpy())

    E_mid = scipy.stats.beta.ppf(q=0.5, a=alpha.numpy(), b=beta.numpy())

    E_low = scipy.stats.beta.ppf(q=0.25, a=alpha.numpy(), b=beta.numpy())

    return E, E_up, E_mid, E_low


def ml_beta_3_obj(parameter, fid, did, measure, N_data, N_flow):
    """
    The log-likelihood objective function of the Beta-3 IRT model

    Parameters
    ----------
    parameter: tensorflow.Variable
        The parameters for the Beta-3 IRT model.

    fid: tf.Tensor
        A (n_batch, ) tensorflow array contains the dataset index of each experiment.

    did: tf.Tensor
        A (n_batch, ) tensorflow array contains the model index of each experiment.

    measure: tf.Tensor
        A (n_batch, ) tensorflow array contains the performance measurement of each experiment.

    N_data: int
        The total number of datasets in the IRT model.

    N_flow: int
        The total number of models in the IRT model.

    Returns
    ----------
    L: tf.Tensor
        The averaged log-likelihood of the IRT model.

    """

    logit_theta = tf.gather(parameter[:N_flow], fid, axis=0)

    logit_delta = tf.gather(parameter[N_flow:N_flow + N_data], did, axis=0)

    a = tf.math.exp(tf.gather(parameter[N_flow + N_data:N_flow + 2 * N_data], did, axis=0))

    theta = tf.clip_by_value(1 / (1 + tf.math.exp(logit_theta)),
                             tf.constant(eps, dtype='float32'),
                             tf.constant(1 - eps, dtype='float32'))

    delta = tf.clip_by_value(1 / (1 + tf.math.exp(logit_delta)),
                             tf.constant(eps, dtype='float32'),
                             tf.constant(1 - eps, dtype='float32'))

    alpha = tf.math.pow(theta / delta, a) + tf.constant(eps, dtype='float32')

    beta = tf.math.pow((1 - theta) / (1 - delta), a) + tf.constant(eps, dtype='float32')
    
    measure = tf.reshape(measure, -1)

    loglik = (alpha - 1) * tf.math.log(measure) + (beta - 1) * tf.math.log(1 - measure) - \
             (tf.math.lgamma(alpha) + tf.math.lgamma(beta) - tf.math.lgamma(alpha + beta))

    L = - tf.reduce_mean(loglik)
             
    return L


def get_E_beta_3(parameter, fid, did, N_data, N_flow):
    """
    Predict the expected response for a set of combinations between different models and datasets

    Parameters
    ----------
    parameter: numpy.ndarray
        The estimated parameters for the Beta-3 IRT model.

    fid: numpy.ndarray
        A (n_test, ) numpy array contains the model index of each testing experiment.

    did: numpy.ndarray
        A (n_test, ) numpy array contains the dataset index of each testing experiment.

    N_data: int
        The total number of datasets in the IRT model.

    N_flow: int
        The total number of models in the IRT model.

    Returns
    ----------
    E: numpy.ndarray
        A (n_test, ) numpy array contains the expected performance measurement of each testing experiment.

    """

    logit_theta = tf.gather(parameter[:N_flow], fid, axis=0)

    logit_delta = tf.gather(parameter[N_flow:N_flow + N_data], did, axis=0)

    a = tf.math.exp(tf.gather(parameter[N_flow + N_data:N_flow + 2 * N_data], did, axis=0))

    theta = tf.clip_by_value(1 / (1 + tf.math.exp(logit_theta)),
                             tf.constant(eps, dtype='float32'),
                             tf.constant(1 - eps, dtype='float32'))

    delta = tf.clip_by_value(1 / (1 + tf.math.exp(logit_delta)),
                             tf.constant(eps, dtype='float32'),
                             tf.constant(1 - eps, dtype='float32'))

    alpha = tf.math.pow(theta / delta, a) + tf.constant(eps, dtype='float32')

    beta = tf.math.pow((1 - theta) / (1 - delta), a) + tf.constant(eps, dtype='float32')

    E = alpha / (alpha + beta)

    return E


def get_obj(parameter, data, extra_args):
    """
    Wrapper function to get the value of the objective function.

    Parameters
    ----------
    parameter: tensorflow.Variable
        The parameter of the IRT model.

    data: tensorflow.Tensor
        A (n_batch, 3) tensor contains the model index, dataset index, and performance measure of each experiment.

    extra_args: tuple
        A tuple contains extra information for the IRT model.
        (irt type, total number of datasets, total number of models)

    Returns
    ----------
    L: tensorflow.Tensor
        The averaged log-likelihood of the IRT model.

    """
    
    fid = tf.cast(data[:, 0], 'int32')
    
    did = tf.cast(data[:, 1], 'int32')
    
    measure = tf.cast(data[:, 2:], 'float32')

    irt_type, tmp_args = extra_args
    
    if irt_type == 'beta3':

        N_data, N_flow = tmp_args

        L = ml_beta_3_obj(parameter, fid, did, measure, N_data, N_flow)
            
    elif irt_type == 'logistic':

        N_data, N_flow = tmp_args
        
        L = ml_logistic_obj(parameter, fid, did, measure, N_data, N_flow)
            
    else:

        L = None
    
    return L


def get_obj_g(parameter, data, extra_args):
    """
    Wrapper function to get the gradient of the objective function.

    Parameters
    ----------
    parameter: tensorflow.Variable
        The parameter of the IRT model.

    data: tensorflow.Tensor
        A (n_batch, 3) tensor contains the model index, dataset index, and performance measure of each experiment.

    extra_args: tuple
        A tuple contains extra information for the IRT model.
        (irt type, total number of datasets, total number of models)

    Returns
    ----------
    L: tensorflow.Tensor

    g: tensorflow.Tensor

    """
    
    with tf.GradientTape() as gt:
        
        gt.watch(parameter)
        
        L = get_obj(parameter, data, extra_args)
        
        g = gt.gradient(L, parameter)
        
    return L, g
