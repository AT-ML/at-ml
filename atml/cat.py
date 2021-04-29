"""
The :mod:'atml.cat' module contains a set of functions to perform adaptive testing on predictive ML models.
"""
# Author: Hao Song (nuacesh@gmail.com)
# License: BSD-3

import numpy

import scipy.stats

import tensorflow as tf

from exp import get_single_testing

from adhoc_opt.optimiser import parameter_update

import copy

tf.compat.v1.enable_eager_execution()

pi = tf.constant(numpy.pi, dtype='float32')

eps = 1e-6


class Standard_CAT:
    """

    The class for a standard adaptive testing on ML models

    """

    def __init__(self, irt_mdl):
        """

        Parameters
        ----------
        irt_mdl: atml.irt
            A trained IRT model as defined by atml.irt.

        """

        self.N_model = irt_mdl.N_model
        self.N_dataset = irt_mdl.N_dataset
        self.irt_type = irt_mdl.irt_type
        self.N_test = None
        self.logit_theta = tf.zeros(self.N_model, 'float32')
        self.logit_delta = tf.zeros(self.N_dataset, 'float32')
        self.log_a = tf.zeros(self.N_dataset, 'float32')
        self.theta = tf.zeros(self.N_model, 'float32')
        self.delta = tf.zeros(self.N_dataset, 'float32')
        self.log_s2 = tf.zeros(self.N_dataset, 'float32')

        if self.irt_type == 'beta3':
            self.logit_theta = tf.cast(irt_mdl.logit_theta, 'float32')
            self.logit_delta = tf.cast(irt_mdl.logit_delta, 'float32')
            self.log_a = tf.cast(irt_mdl.log_a, 'float32')
        elif self.irt_type == 'logistic':
            self.theta = tf.cast(irt_mdl.theta, 'float32')
            self.delta = tf.cast(irt_mdl.delta, 'float32')
            self.log_a = tf.cast(irt_mdl.log_a, 'float32')
            self.log_s2 = tf.cast(irt_mdl.log_s2, 'float32')

    def testing(self, mdl, measure,
                data_dict, get_data,
                item_info='fisher',
                ability_0=None,
                N_test=None, remove_tested=True,
                sparse=False, cap_size=10000, tes_size=0.5):
        """
        Perform the adaptive testing and record the testing sequence.

        Parameters
        ----------
        mdl: sklearn.predictor
            An instance of the sklearn predictor.
            The model should have a fit(x, y) method for training and predict_proba(x) for testing.

        measure: atml.Measure
            A evaluation measure selected from the atml.measure module.

        data_dict: dict
            A dictionary that defines the index and the reference name of all the datasets.
            Example: data_dict = {0: 'iris', 1: 'digits', 2: 'wine'}

        get_data: Callable
            A function that takes the dataset index and returns the features (x), and target (y) for the specified
            dataset.

        item_info: string
            The selected item information criterion. Options: (1) 'fisher', (2) 'kl', (3) 'random'.
            'fisher': the Fisher item information.
            'kl': the Kullback-Leibler item information.
            'random': random Gaussian item information.

        ability_0: float
            Initial value for the ability parameter.

        N_test: int
            Number of tests to be performed.

        remove_tested: boolean
            Whether to remove tested dataset and no longer test with the same dataset.

        sparse: boolean
            To indicate whether to only use a subset of the dataset to perform the experiments.

        cap_size: int
            In the case sparse=True, cap_size specifies the maximum size of the dataset to run the experiments.

        tes_size: float
            The proportion of the dataset that is used as the testing set (validation set).

        Returns
        ----------

        selected_dataset_index: numpy.ndarray
            The index sequence of selected datasets during the adaptive testing.

        selected_dataset: list
            The reference name sequence of selected dataset during the adaptive testing.

        measurement: numpy.ndarray
            The performance measurements of selected datasets during the adaptive testing.

        ability_seq: numpy.ndarray
            The estimated ability through the adaptive testing sequence.

        """

        if N_test is None:
            self.N_test = self.N_dataset
        else:
            self.N_test = N_test

        if ability_0 is None:
            if self.irt_type == 'beta3':
                ability = tf.cast(numpy.median(self.logit_theta), 'float32')
            elif self.irt_type == 'logistic':
                ability = tf.cast(numpy.median(self.theta), 'float32')
        else:
            ability = tf.cast(ability_0, 'float32')

        selected_dataset = []

        selected_dataset_index = numpy.zeros(self.N_test)

        measurements = numpy.zeros(self.N_test)

        ability_seq = numpy.zeros(self.N_test + 1)

        if self.irt_type == 'beta3':
            ability_seq[0] = 1 / (1 + numpy.exp(ability.numpy()))
        else:
            ability_seq[0] = ability.numpy().copy()

        for i in range(1, self.N_test+1):

            print('======================================')

            print('Test No.' + str(i) + ', mdl: ' + str(mdl) + ', measure: ' + str(measure) + ', info: ' + item_info +
                  ', irt: ' + self.irt_type)

            if item_info == 'fisher':

                v = get_fisher_item_information(ability=ability, logit_delta=self.logit_delta, delta=self.delta,
                                                log_a=self.log_a, log_s2=self.log_s2, irt_type=self.irt_type)

            elif item_info == 'kl':

                v = get_kl_item_information(ability=ability, logit_delta=self.logit_delta, delta=self.delta,
                                            log_a=self.log_a, log_s2=self.log_s2, irt_type=self.irt_type)
                
            elif item_info == 'random':
                
                v = numpy.random.randn(self.N_dataset)

            print('Max Info:' + str(numpy.max(v)))

            print('Min Info:' + str(numpy.min(v)))

            if remove_tested:
                v[selected_dataset_index[:i].astype('int')] = - numpy.inf

            max_idx = numpy.argmax(v)

            selected_dataset_index[i-1] = max_idx
            
            tmp_mdl = copy.deepcopy(mdl)

            tmp_measure = get_single_testing(max_idx, tmp_mdl, data_dict, get_data, measure,
                                             sparse, cap_size, tes_size)

            tmp_measure = measure.transform(tmp_measure)

            if tmp_measure >= (1 - eps):
                tmp_measure = 1 - eps

            if tmp_measure <= eps:
                tmp_measure = eps

            measurements[i-1] = tmp_measure

            print('selected dataset: ' + data_dict[max_idx])

            selected_dataset.append(data_dict[max_idx])

            print('test result is: ' + str(tmp_measure))

            ability = tf.reshape(tf.cast(ability, 'float32'), [-1, 1])
            
            data = numpy.hstack([numpy.array(selected_dataset_index[:i]).reshape(-1, 1),
                                 numpy.array(measurements[:i]).reshape(-1, 1)])

            extra_args = (self.irt_type, self.logit_delta, self.delta, self.log_a, self.log_s2)
            
            ability = parameter_update(theta_0=tf.Variable(ability), data=data, extra_args=extra_args,
                                       obj=get_obj, obj_g=get_obj_g,
                                       lr=1e-3,
                                       batch_size=1, val_size=len(selected_dataset), factr=1e-16,
                                       tol=1024,
                                       max_batch=int(1e8),
                                       plot_loss=False, print_info=False, 
                                       plot_final_loss=False, print_iteration=False).numpy()

            if self.irt_type == 'beta3':
                ability_seq[i] = 1 / (1 + numpy.exp(ability))
            else:
                ability_seq[i] = ability.copy()

            print('current estimated ability is:' + str(ability_seq[i]))

        return selected_dataset_index, selected_dataset, measurements, ability_seq


def get_kl_item_information(ability, logit_delta, delta, log_a, log_s2, d_ability=1e-1, irt_type='beta3',
                            n_sample=65536):
    """
    Compute the KL item information for adaptive testing.

    Parameters
    ----------
    ability: float
        The current estimated ability for the candidate model.

    logit_delta: numpy.ndarray
        The logit of the delta parameter of the IRT model.

    delta: numpy.ndarray
        The delta parameter of the IRT model.

    log_a: numpy.ndarray
        The log_a parameter of the IRT model.

    log_s2: numpy.ndarray
        The log_s2 parameter of the IRT model.

    d_ability: float
        The change of the ability parameter when calculating the KL item information.

    irt_type: string
        The type of the IRT model.

    n_sample: integer
        The number of random samples used when calculating the KL item information.

    Returns
    ----------
    info: numpy.ndarray
        The KL item information for all the datasets with the given ability.

    """

    n_dataset = len(log_a)

    a = numpy.exp(log_a)

    if irt_type == 'beta3':

        theta = tf.clip_by_value(1 / (1 + tf.math.exp(ability)), tf.constant(1e-6, dtype='float32'),
                                 tf.constant(1 - 1e-6, dtype='float32'))

        delta = tf.clip_by_value(1 / (1 + tf.math.exp(logit_delta)), tf.constant(1e-6, dtype='float32'),
                                 tf.constant(1 - 1e-6, dtype='float32'))

        alpha = numpy.repeat((tf.math.pow(theta / delta, a) +
                              tf.constant(1e-6, dtype='float32')).numpy().reshape(1, -1), n_sample, axis=0)

        beta = numpy.repeat((tf.math.pow((1 - theta) / (1 - delta), a) +
                             tf.constant(1e-6, dtype='float32')).numpy().reshape(1, -1), n_sample, axis=0)

        measure_sample = scipy.stats.beta.rvs(a=alpha, b=beta, size=[n_sample, n_dataset])

    elif irt_type == 'logistic':
        
        s2 = numpy.exp(log_s2) + 1e-6

        s = numpy.repeat(numpy.sqrt(s2).reshape(1, -1), n_sample, axis=0) 

        mu = numpy.repeat(tf.reshape(- a * (ability - delta.numpy()), [1, -1]).numpy(), n_sample, axis=0)

        logit_measure_sample = scipy.stats.norm.rvs(loc=mu, scale=s, size=[n_sample, n_dataset])

        measure_sample = 1 / (1 + numpy.exp(logit_measure_sample))

    measure_sample[measure_sample >= (1.0 - eps)] = 1.0 - eps

    measure_sample[measure_sample <= eps] = eps
    
    measure_sample = tf.cast(measure_sample, 'float32')

    sample_ability = tf.convert_to_tensor(numpy.repeat(ability, n_sample * n_dataset), 'float32')

    sample_ability_p = tf.convert_to_tensor(numpy.repeat(ability + d_ability, n_sample * n_dataset), 'float32')

    sample_ability_n = tf.convert_to_tensor(numpy.repeat(ability - d_ability, n_sample * n_dataset), 'float32')

    all_selected_dataset = list(range(0, len(log_a))) * n_sample

    if irt_type == 'beta3':
        L = ml_beta_3_obj(sample_ability, logit_delta, log_a,
                          measure_sample, all_selected_dataset, True).numpy().reshape(n_sample, n_dataset)
        L_p = ml_beta_3_obj(sample_ability_p, logit_delta, log_a,
                            measure_sample, all_selected_dataset, True).numpy().reshape(n_sample, n_dataset)
        L_n = ml_beta_3_obj(sample_ability_n, logit_delta, log_a,
                            measure_sample, all_selected_dataset, True).numpy().reshape(n_sample, n_dataset)
    elif irt_type == 'logistic':
        L = ml_logistic_obj(sample_ability, delta, log_a, log_s2,
                            measure_sample, all_selected_dataset, True).numpy().reshape(n_sample, n_dataset)
        L_p = ml_logistic_obj(sample_ability_p, delta, log_a, log_s2,
                              measure_sample, all_selected_dataset, True).numpy().reshape(n_sample, n_dataset)
        L_n = ml_logistic_obj(sample_ability_n, delta, log_a, log_s2,
                              measure_sample, all_selected_dataset, True).numpy().reshape(n_sample, n_dataset)

    info = (numpy.mean(L_p, axis=0) - numpy.mean(L, axis=0)) + (numpy.mean(L_n, axis=0) - numpy.mean(L, axis=0))

    return info


def get_fisher_item_information(ability, logit_delta, delta, log_a, log_s2, irt_type='beta3', n_sample=65536):
    """
    Compute the Fisher item information for adaptive testing.

    Parameters
    ----------
    ability: float
        The current estimated ability for the candidate model.

    logit_delta: numpy.ndarray
        The logit of the delta parameter of the IRT model.

    delta: numpy.ndarray
        The delta parameter of the IRT model.

    log_a: numpy.ndarray
        The log_a parameter of the IRT model.

    log_s2: numpy.ndarray
        The log_s2 parameter of the IRT model.

    irt_type: string
        The type of the IRT model.

    n_sample: integer
        The number of random samples used when calculating the KL item information.

    Returns
    ----------
    info: numpy.ndarray
        The Fisher item information for all the datasets with the given ability.

    """

    n_dataset = len(log_a)

    measure_sample = numpy.zeros([n_sample, n_dataset])

    all_selected_dataset = list(range(0, len(log_a))) * n_sample

    a = numpy.exp(log_a)

    if irt_type == 'beta3':

        theta = tf.clip_by_value(1 / (1 + tf.math.exp(ability)), tf.constant(1e-6, dtype='float32'),
                                 tf.constant(1 - 1e-6, dtype='float32'))

        delta = tf.clip_by_value(1 / (1 + tf.math.exp(logit_delta)), tf.constant(1e-6, dtype='float32'),
                                 tf.constant(1 - 1e-6, dtype='float32'))

        alpha = numpy.repeat((tf.math.pow(theta / delta, a) + tf.constant(1e-6, dtype='float32')).numpy().reshape(1, -1), n_sample, axis=0)

        beta = numpy.repeat((tf.math.pow((1 - theta) / (1 - delta), a) + tf.constant(1e-6, dtype='float32')).numpy().reshape(1, -1), n_sample, axis=0)

        measure_sample = scipy.stats.beta.rvs(a=alpha, b=beta, size=[n_sample, n_dataset])

    elif irt_type == 'logistic':
        
        s2 = numpy.exp(log_s2) + 1e-6

        s = numpy.repeat(numpy.sqrt(s2).reshape(1, -1), n_sample, axis=0) 

        mu = numpy.repeat((- a * (ability - delta)).numpy().reshape(1, -1), n_sample, axis=0)

        logit_measure_sample = scipy.stats.norm.rvs(loc=mu, scale=s, size=[n_sample, n_dataset])

        measure_sample = 1 / (1 + numpy.exp(logit_measure_sample))

    measure_sample[measure_sample >= (1.0 - eps)] = 1.0 - eps

    measure_sample[measure_sample <= eps] = eps
    
    measure_sample = tf.cast(measure_sample, 'float32')

    sample_ability = tf.cast(numpy.repeat(ability, n_sample * n_dataset), 'float32')

    with tf.GradientTape() as gt:

        gt.watch(sample_ability)

        if irt_type == 'beta3':
            L = ml_beta_3_obj(sample_ability, logit_delta, log_a, measure_sample, all_selected_dataset, True)
        elif irt_type == 'logistic':
            L = ml_logistic_obj(sample_ability, delta, log_a, log_s2, measure_sample, all_selected_dataset, True)

        g = gt.gradient(L, sample_ability).numpy().reshape(n_sample, n_dataset)

    info = numpy.mean(numpy.square(g), axis=0)

    return info


def m_beta_3_irt(logit_theta, logit_delta, log_a):
    """
    Predict the expected response for a set of IRT parameters

    Parameters
    ----------
    logit_theta: numpy.ndarray
        The logit_theta parameter of the Beta-3 IRT model.

    logit_delta: numpy.ndarray
        The logit_delta parameter of the Beta-3 IRT model.

    log_a: numpy.ndarray
        The log_a parameter of the Beta-3 IRT model.

    Returns
    ----------
    E: numpy.ndarray
        The expected performance measurement.

    """

    a = tf.math.exp(log_a)

    theta = tf.clip_by_value(1 / (1 + tf.math.exp(logit_theta)), 
                             tf.constant(1e-6, dtype='float32'), 
                             tf.constant(1-1e-6, dtype='float32'))

    delta = tf.clip_by_value(1 / (1 + tf.math.exp(logit_delta)),
                             tf.constant(1e-6, dtype='float32'),
                             tf.constant(1-1e-6, dtype='float32'))

    alpha = tf.math.pow(theta / delta, a) + tf.constant(1e-6, dtype='float32')

    beta = tf.math.pow((1 - theta) / (1 - delta), a) + tf.constant(1e-6, dtype='float32')

    E = alpha / (alpha + beta)

    return E


def m_logistic_irt(theta, delta, log_a, log_s2):
    """
    Predict the expected response for a set of IRT parameters

    Parameters
    ----------
    theta: numpy.ndarray
        The theta parameter of the Logistic IRT model.

    delta: numpy.ndarray
        The delta parameter of the Logistic IRT model.

    log_a: numpy.ndarray
        The log_a parameter of the Logistic IRT model.

    log_s2: numpy.ndarray
        The log_s2 parameter of the Logistic IRT model.

    Returns
    ----------
    E: numpy.ndarray
        The expected performance measurement.

    """

    a = numpy.exp(log_a)
    
    s2 = numpy.exp(log_s2) + 1e-6

    s = numpy.sqrt(s2)

    mu = - a * (theta - delta)

    samples = scipy.stats.norm.rvs(loc=mu, scale=s, size=[65536, len(log_a)])

    E = numpy.mean(1 / (1 + numpy.exp(samples)), axis=0)

    return E


def ml_logistic_obj(theta, delta, log_a, log_s2, measure, tested_list, using_samples=False):
    """
    The log-likelihood objective function of the Logistic IRT model

    Parameters
    ----------
    theta: tensorflow.Variable
        The current ability parameter of the IRT model.

    delta: tf.Tensor
        The current delta parameter of the IRT model.

    log_a: tf.Tensor
        The current log_a parameter of the IRT model.

    log_s2: tf.Tensor
        The current log_s2 parameter of the IRT model.

    measure: tf.Tensor
        The performance measurements for the selected datasets.

    tested_list: tf.Tensor
        The index of each selected dataset.

    using_samples: boolean
        Whether to return the sample-wise negative log-likelihood.

    Returns
    ----------
    nll: tf.Tensor
        The negative log-likelihood of the IRT model.

    """

    delta = tf.gather(delta, tested_list, axis=0)

    a = tf.math.exp(tf.gather(log_a, tested_list, axis=0))

    log_s2 = tf.gather(log_s2, tested_list, axis=0)

    s2 = tf.math.exp(log_s2) + tf.constant(1e-6, dtype='float32')

    measure = tf.reshape(measure, [-1])

    logit_measure = tf.math.log((1 - measure) / measure)

    mu = - a * (theta - delta)
    
    diff = logit_measure - mu
    
    exp = tf.math.exp(- 0.5 * tf.math.square(diff) / s2)
    
    sample_lik = 1 / (tf.math.sqrt(2 * pi * s2)) * exp * (1 / (measure * (1 - measure)))
    
    loglik = tf.negative(tf.math.log(sample_lik + tf.constant(1e-6, dtype='float32')))

    # loglik = 0.5 * tf.math.log(1 / (2 * pi)) + 0.5 * tf.math.log(1 / s2) \
    #     - 0.5 * tf.math.square(logit_measure - mu) / s2 \
    #     + tf.math.log(1 / (measure * (1 - measure)))

    if using_samples:
        nll = loglik
    else:
        nll = tf.reduce_mean(loglik)

    return nll


def ml_beta_3_obj(logit_theta, logit_delta, log_a, measure, tested_list, using_samples=False):
    """
    The log-likelihood objective function of the Logistic IRT model

    Parameters
    ----------
    logit_theta: tensorflow.Variable
        The current ability parameter of the IRT model.

    logit_delta: tf.Tensor
        The current logit_delta parameter of the IRT model.

    log_a: tf.Tensor
        The current log_a parameter of the IRT model.

    measure: tf.Tensor
        The performance measurements for the selected datasets.

    tested_list: tf.Tensor
        The index of each selected dataset.

    using_samples: boolean
        Whether to return the sample-wise negative log-likelihood.

    Returns
    ----------
    nll: tf.Tensor
        The negative log-likelihood of the IRT model.

    """

    logit_delta = tf.gather(logit_delta, tested_list, axis=0)

    measure = tf.reshape(measure, [-1])

    theta = tf.clip_by_value(1 / (1 + tf.math.exp(logit_theta)), tf.constant(1e-16, dtype='float32'),
                             tf.constant(1-1e-16, dtype='float32'))

    delta = tf.clip_by_value(1 / (1 + tf.math.exp(logit_delta)), tf.constant(1e-16, dtype='float32'),
                             tf.constant(1-1e-16, dtype='float32'))

    a = tf.math.exp(tf.gather(log_a, tested_list, axis=0))

    alpha = tf.math.pow(theta / delta, a) + tf.constant(1e-6, dtype='float32')

    beta = tf.math.pow((1 - theta) / (1 - delta), a) + tf.constant(1e-6, dtype='float32')

    loglik = (alpha - 1) * tf.math.log(measure) + (beta - 1) * tf.math.log(1 - measure) - \
             (tf.math.lgamma(alpha) + tf.math.lgamma(beta) - tf.math.lgamma(alpha + beta))

    if using_samples:
        nll = - loglik
    else:
        nll = - tf.reduce_mean(loglik)

    return nll


def get_obj(parameter, data, extra_args):
    """
    Wrapper function to get the value of the objective function.

    Parameters
    ----------
    parameter: tensorflow.Variable
        The parameter of the IRT model.

    data: tensorflow.Tensor
        A tensor contains the selected dataset index, and performance measure of each experiment.

    extra_args: tuple
        A tuple contains extra parameters of the IRT model.
        (irt type, logit_delta, delta, log_a, log_2)

    Returns
    ----------
    L: tensorflow.Tensor
        The negative log-likelihood.

    """
    
    tested_list = tf.cast(data[:, 0], 'int32')
    
    measure = tf.cast(data[:, 1], 'float32')
    
    irt_type, logit_delta, delta, log_a, log_s2 = extra_args
    
    logit_delta = tf.cast(logit_delta, 'float32')
    
    delta = tf.cast(delta, 'float32')
    
    log_a = tf.cast(log_a, 'float32')
    
    log_s2 = tf.cast(log_s2, 'float32')
    
    if irt_type == 'beta3':
        L = ml_beta_3_obj(parameter, logit_delta, log_a, measure, tested_list)
    elif irt_type == 'logistic':
        L = ml_logistic_obj(parameter, delta, log_a, log_s2, measure, tested_list)
        
    return L


def get_obj_g(parameter, data, extra_args):
    """
    Wrapper function to get the gradient of the objective function.

    Parameters
    ----------
    parameter: tensorflow.Variable
        The parameter of the IRT model.

    data: tensorflow.Tensor
        A tensor contains the selected dataset index, and performance measure of each experiment.

    extra_args: tuple
        A tuple contains extra parameters of the IRT model.
        (irt type, logit_delta, delta, log_a, log_2)

    Returns
    ----------
    L: tensorflow.Tensor
        The negative log-likelihood.

    g: tensorflow.Tensor
        The gradient of the parameters.

    """
    
    with tf.GradientTape() as gt:
        
        gt.watch(parameter)
        
        L = get_obj(parameter, data, extra_args)
        
        g = gt.gradient(L, parameter)
        
    return L, g

