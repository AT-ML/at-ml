import numpy

import scipy.stats

import tensorflow as tf

from tensorflow import keras

from data import dataset_list, load_dataset

from measure import measures_all

from irt import psi

from exp import get_random_split_measurement

from adhoc_opt.optimiser import parameter_update

import matplotlib

import copy

matplotlib.use('Agg')

import matplotlib.pyplot

tf.compat.v1.enable_eager_execution()

# tf.compat.v1.enable_eager_execution()

# lr_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]


class Standard_CAT:

    def __init__(self, irt_type='beta3', N_dataset=None, parameter_list=None):

        self.N_dataset = N_dataset
        self.irt_type = irt_type
        # self.logit_delta = None
        # self.delta = None
        # self.log_a = None
        # self.log_s2 = None
        # self.N_approx = None
        # self.mu_alpha = None
        # self.L_alpha = None
        # self.dataset_log_s2 = None
        # self.e_0 = None

        if irt_type == 'beta3':
            self.logit_delta = tf.cast(parameter_list[0], 'float32')
            self.log_a = tf.cast(parameter_list[1], 'float32')
        elif irt_type == 'logistic':
            self.delta = tf.cast(parameter_list[0], 'float32')
            self.log_a = tf.cast(parameter_list[1], 'float32')
            self.log_s2 = tf.cast(parameter_list[2], 'float32')
        elif irt_type == 'gp':
            self.N_approx = tf.cast(parameter_list[0], 'int32')
            self.mu_alpha = tf.cast(parameter_list[1], 'float32')
            self.L_alpha = tf.cast(parameter_list[2], 'float32')
            self.dataset_log_s2 = tf.cast(parameter_list[3], 'float32')
            self.e_0 = tf.cast(parameter_list[4], 'float32')

    def testing(self, mdl=None, model_theta=None,
                test_all_measure=None,
                test_size=0.5, target_measure=2,
                item_info='fisher', N_test=None, remove_tested=True):

        if N_test is None:
            N_test = len(dataset_list)

        test_measure = numpy.zeros([len(dataset_list), 10])

        if test_all_measure is None:
            test_all_measure = numpy.zeros([10, len(dataset_list), 6])
            for i in range(0, 10):
                for j in range(0, len(dataset_list)):
                    test_all_measure[i, j, :] = get_random_split_measurement('test',
                                                                             dataset_list[j],
                                                                             test_size,
                                                                             copy.deepcopy(mdl))

        eps = 1e-6

        for i in range(0, 10):
            test_all_measure[i, :, 1] = (1 - (test_all_measure[i, :, 1] / 2))
            test_all_measure[i, :, 2] = numpy.exp(- test_all_measure[i, :, 2])
            test_measure[:, i] = test_all_measure[i, :, target_measure - 2]
            test_measure[test_measure[:, i] <= eps, i] = eps
            test_measure[test_measure[:, i] >= (1.0 - eps), i] = 1 - eps
            
        test_measure = tf.cast(test_measure, 'float32')

        # if ability is None:
        #     ability = tf.constant(0.0, dtype='float32')
        # else:
        #     ability = tf.cast(ability, 'float32')

        if model_theta is None:
            if self.irt_type == 'gp':
                model_theta = tf.zeros(2, 'float32')
            else:
                model_theta = tf.zeros(1, 'float32')
        else:
            model_theta = tf.cast(model_theta, 'float32')

        selected_dataset = []

        selected_dataset_index = []

        selected_measure = []

        all_selected_dataset = list(range(0, len(dataset_list)))

        mse = numpy.zeros([len(dataset_list) + 1])

        nll = numpy.zeros([len(dataset_list) + 1])

        for j in range(0, 10):
            if self.irt_type == 'logistic':
                mse[0] = mse[0] + tf.reduce_mean(tf.math.square(test_measure[:, j] -
                                                                m_logistic_irt(model_theta,
                                                                               self.delta,
                                                                               self.log_a,
                                                                               self.log_s2)))
                nll[0] = nll[0] + ml_logistic_obj(model_theta, self.delta, self.log_a,
                                                  self.log_s2, test_measure[:, j],
                                                  all_selected_dataset)
            elif self.irt_type == 'beta3':
                mse[0] = mse[0] + tf.reduce_mean(tf.math.square(test_measure[:, j] -
                                                                m_beta_3_irt(model_theta,
                                                                             self.logit_delta,
                                                                             self.log_a)))
                nll[0] = nll[0] + ml_beta_3_obj(model_theta, self.logit_delta, self.log_a,
                                                test_measure[:, j], all_selected_dataset)
            elif self.irt_type == 'gp':
                mse[0] = mse[0] + tf.reduce_mean(tf.math.square(test_measure[:, j] -
                                                                m_gp_irt(model_theta,
                                                                         self.mu_alpha,
                                                                         self.L_alpha,
                                                                         self.dataset_log_s2,
                                                                         self.e_0,
                                                                         len(dataset_list),
                                                                         self.N_approx)))
                nll[0] = nll[0] + ml_beta_3_obj(ability, self.logit_delta, self.log_a,
                                                test_measure[:, j], all_selected_dataset)

        for i in range(1, N_test+1):

            print('======================================')

            print('Test No.' + str(i) + ', mdl: ' + str(mdl) +  ', measure: ' + str(target_measure) + ', info: ' + item_info + ', irt: ' + self.irt_type)

            if item_info == 'fisher':

                v = get_fisher_item_information(ability=ability, logit_delta=self.logit_delta, delta=self.delta,
                                                log_a=self.log_a, log_s2=self.log_s2, irt_type=self.irt_type)

            elif item_info == 'kl':

                v = get_kl_item_information(ability=ability, logit_delta=self.logit_delta, delta=self.delta,
                                            log_a=self.log_a, log_s2=self.log_s2, irt_type=self.irt_type)
                
            elif item_info == 'random':
                
                v = numpy.random.randn(len(dataset_list))

            print('Max Info:' + str(numpy.max(v)))

            print('Min Info:' + str(numpy.min(v)))

            if remove_tested:
                if len(selected_dataset_index) > 0:
                    v[numpy.array(selected_dataset_index)] = - numpy.inf

            max_idx = numpy.argmax(v)

            selected_dataset_index.append(max_idx)
            
            tmp_mdl = copy.deepcopy(mdl)

            tmp_measure = numpy.array(get_random_split_measurement(model='test', dataset=dataset_list[max_idx], 
                                                                   test_size=test_size, model_instance=tmp_mdl))

            tmp_measure[1] = (1 - (tmp_measure[1] / 2))

            tmp_measure[2] = numpy.exp(- tmp_measure[2])

            tmp_measure = tmp_measure[target_measure - 2]

            if tmp_measure >= (1 - eps):
                tmp_measure = 1 - eps

            if tmp_measure <= eps:
                tmp_measure = eps

            selected_measure.append(tmp_measure)
            
            print('selected dataset: ' + dataset_list[max_idx])

            selected_dataset.append(dataset_list[max_idx])

            print('test result is: ' + str(tmp_measure))

            ability = tf.reshape(tf.cast(ability, 'float32'), [-1, 1])
            
            data = numpy.hstack([numpy.array(selected_dataset_index).reshape(-1, 1), 
                                 numpy.array(selected_measure).reshape(-1, 1)])
            
            extra_args = (self.irt_type, self.logit_delta, self.delta, self.log_a, self.log_s2)
            
            ability = parameter_update(theta_0=tf.Variable(ability), data=data, extra_args=extra_args,
                                       obj=get_obj, obj_g=get_obj_g,
                                       lr=1e-3,
                                       batch_size=1, val_size=len(selected_dataset), factr=1e-16, tol=len(selected_measure) * 8,
                                       max_batch=int(1e8),
                                       plot_loss=False, print_info=False, 
                                       plot_final_loss=False, print_iteration=False).numpy()

            print('current estimated ability is:' + str(ability))

            for j in range(0, 10):
                if self.irt_type == 'logistic':
                    mse[i] = mse[i] + tf.reduce_mean(tf.math.square(test_measure[:, j] -
                                                                    m_logistic_irt(ability,
                                                                                   self.delta, self.log_a, self.log_s2)))
                    nll[i] = nll[i] + ml_logistic_obj(ability, self.delta, self.log_a, self.log_s2,
                                                      test_measure[:, j], all_selected_dataset)
                elif self.irt_type == 'beta3':
                    mse[i] = mse[i] + tf.reduce_mean(tf.math.square(test_measure[:, j] -
                                                                    m_beta_3_irt(ability, self.logit_delta,
                                                                                 self.log_a)))
                    nll[i] = nll[i] + ml_beta_3_obj(ability, self.logit_delta, self.log_a,
                                                    test_measure[:, j], all_selected_dataset)

        return selected_dataset_index, mse * 0.1, nll * 0.1


def get_kl_item_information(ability, logit_delta, delta, log_a, log_s2, d_ability=1e-1, irt_type='beta3',
                            n_sample=65536):

    n_dataset = len(dataset_list)

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

        mu = numpy.repeat(tf.reshape(- a * (ability - delta.numpy()), [1, -1]).numpy(), n_sample, axis=0)

        logit_measure_sample = scipy.stats.norm.rvs(loc=mu, scale=s, size=[n_sample, n_dataset])

        measure_sample = 1 / (1 + numpy.exp(logit_measure_sample))

    eps = 1e-6

    measure_sample[measure_sample >= (1.0 - eps)] = 1.0 - eps

    measure_sample[measure_sample <= eps] = eps
    
    measure_sample = tf.cast(measure_sample, 'float32')

    sample_ability = tf.convert_to_tensor(numpy.repeat(ability, n_sample * n_dataset), 'float32')

    sample_ability_p = tf.convert_to_tensor(numpy.repeat(ability + d_ability, n_sample * n_dataset), 'float32')

    sample_ability_n = tf.convert_to_tensor(numpy.repeat(ability - d_ability, n_sample * n_dataset), 'float32')

    all_selected_dataset = list(range(0, len(dataset_list))) * n_sample

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

    return (numpy.mean(L_p, axis=0) - numpy.mean(L, axis=0)) + (numpy.mean(L_n, axis=0) - numpy.mean(L, axis=0))


def get_fisher_item_information(ability, logit_delta, delta, log_a, log_s2, irt_type='beta3', n_sample=65536):

    n_dataset = len(dataset_list)

    measure_sample = numpy.zeros([n_sample, n_dataset])

    all_selected_dataset = list(range(0, len(dataset_list))) * n_sample

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

    eps = 1e-6

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

    return numpy.mean(numpy.square(g), axis=0)


def m_gp_irt(logit_theta, mu_alpha, L_alpha,
             dataset_log_s2, e_0,
             N_data, N_approx, N_sample=4096):

    eps = tf.constant(1e-6, 'float32')

    logit_theta = tf.cast(logit_theta, 'float32') * tf.ones(N_data, 'float32')

    mu_alpha = tf.reshape(tf.cast(mu_alpha, 'float32'), [N_data, N_approx])

    L_alpha = tf.reshape(tf.cast(L_alpha, 'float32'), [N_data, N_approx, N_approx])

    dataset_log_s2 = tf.cast(dataset_log_s2, 'float32')

    dataset_s2 = tf.square(tf.exp(dataset_log_s2))

    model_log_s2 = tf.cast(model_log_s2, 'float32')

    model_s2 = tf.square(tf.exp(model_log_s2))

    s2 = dataset_s2 + model_s2

    e_0 = tf.cast(e_0, 'float32')

    j_index, i_index = numpy.meshgrid(numpy.arange(0, N_approx) + 1, numpy.arange(0, N_approx) + 1)

    theta = tf.clip_by_value(1 / (1 + tf.math.exp(logit_theta)),
                             tf.constant(eps, dtype='float32'),
                             tf.constant(1 - eps, dtype='float32')) - 0.5

    psi_mat = psi(tf.reshape(theta, [-1, 1, 1]),
                  tf.constant(i_index.reshape(1, N_approx, N_approx), dtype='float32'),
                  tf.constant(j_index.reshape(1, N_approx, N_approx), dtype='float32'))

    raw_alpha_samples = tf.random.normal([N_data, N_sample, N_approx], dtype='float32')

    alpha_samples = tf.matmul(raw_alpha_samples, L_alpha) + tf.reshape(mu_alpha, [-1, 1, N_approx])

    e_samples = tf.reduce_mean(tf.matmul(alpha_samples, psi_mat) * alpha_samples, axis=1) + tf.reshape(e_0, [-1, 1])

    mu = numpy.mean(e_samples.numpy(), axis=1)

    var_e = numpy.var(e_samples.numpy(), axis=1)

    s2_hat = s2.numpy().ravel() + var_e.ravel()

    s_hat = numpy.sqrt(s2_hat)

    E = 1 - 1 / (1 + numpy.exp((mu / numpy.sqrt(1 + numpy.pi * numpy.square(s_hat) / 8))))

    return E


def m_beta_3_irt(logit_theta, logit_delta, log_a):

    a = tf.math.exp(log_a)

    theta = tf.clip_by_value(1 / (1 + tf.math.exp(logit_theta)), 
                             tf.constant(1e-6, dtype='float32'), 
                             tf.constant(1-1e-6, dtype='float32'))

    delta = tf.clip_by_value(1 / (1 + tf.math.exp(logit_delta)),
                             tf.constant(1e-6, dtype='float32'),
                             tf.constant(1-1e-6, dtype='float32'))

    alpha = tf.math.pow(theta / delta, a) + tf.constant(1e-6, dtype='float32')

    beta = tf.math.pow((1 - theta) / (1 - delta), a) + tf.constant(1e-6, dtype='float32')

    return alpha / (alpha + beta)


def m_logistic_irt(theta, delta, log_a, log_s2):

    a = numpy.exp(log_a)
    
    s2 = numpy.exp(log_s2) + 1e-6

    s = numpy.sqrt(s2)

    mu = - a * (theta - delta)

    samples = scipy.stats.norm.rvs(loc=mu, scale=s, size=[65536, len(dataset_list)])

    return numpy.mean(1 / (1 + numpy.exp(samples)), axis=0)


def ml_gp_obj(logit_theta, log_s2,
              mu_alpha, L_alpha,
              dataset_log_s2, e_0,
              measure, tested_list,
              N_data, N_flow, N_approx, N_sample,
              using_samples=False):

    eps = tf.constant(1e-6, 'float32')

    pi = tf.constant(numpy.pi, 'float32')

    logit_theta = tf.cast(logit_theta, 'float32') * tf.ones(len(tested_list), 'float32')

    mu_alpha = tf.gather(tf.reshape(tf.cast(mu_alpha, 'float32'), [N_data, N_approx]), tested_list, axis=0)

    L_alpha = tf.gather(tf.reshape(tf.cast(L_alpha, 'float32'), [N_data, N_approx, N_approx]), tested_list, axis=0)

    dataset_log_s2 = tf.gather(tf.cast(dataset_log_s2, 'float32'), tested_list, axis=0)

    dataset_s2 = tf.square(tf.exp(dataset_log_s2))

    model_log_s2 = tf.cast(log_s2, 'float32') * tf.ones(len(tested_list), 'float32')

    model_s2 = tf.square(tf.exp(model_log_s2))

    s2 = dataset_s2 + model_s2

    e_0 = tf.gather(tf.cast(e_0, 'float32'), tested_list, axis=0)

    j_index, i_index = numpy.meshgrid(numpy.arange(0, N_approx) + 1, numpy.arange(0, N_approx) + 1)

    theta = tf.clip_by_value(1 / (1 + tf.math.exp(logit_theta)),
                             tf.constant(eps, dtype='float32'),
                             tf.constant(1 - eps, dtype='float32')) - 0.5

    psi_mat = psi(tf.reshape(theta, [-1, 1, 1]),
                  tf.constant(i_index.reshape(1, N_approx, N_approx), dtype='float32'),
                  tf.constant(j_index.reshape(1, N_approx, N_approx), dtype='float32'))

    raw_alpha_samples = tf.random.normal([len(tested_list), N_sample, N_approx], dtype='float32')

    logit_measure = tf.math.log(measure / (1 - measure))

    alpha_samples = tf.matmul(raw_alpha_samples, L_alpha) + tf.reshape(mu_alpha, [-1, 1, N_approx])

    e_samples = tf.reduce_mean(tf.matmul(alpha_samples, psi_mat) * alpha_samples, axis=2) + tf.reshape(e_0, [-1, 1])

    diff = logit_measure - e_samples

    exp = tf.math.exp(- 0.5 * tf.math.square(diff) / s2)

    sample_lik = 1 / (tf.math.sqrt(2 * pi * s2)) * exp * (1 / (tf.reshape(measure, [-1, 1]) *
                                                               (1 - tf.reshape(measure, [-1, 1]))))

    link_log_lik = tf.math.log(tf.reduce_mean(sample_lik + tf.constant(1e-16, dtype='float32'), axis=1))

    if using_samples:
        res = - link_log_lik
    else:
        res = - tf.reduce_mean(link_log_lik)

    return res


def ml_logistic_obj(theta, delta, log_a, log_s2, measure, tested_list, using_samples=False):

    pi = tf.constant(numpy.pi, dtype='float32')

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
    
    with tf.GradientTape() as gt:
        
        gt.watch(parameter)
        
        L = get_obj(parameter, data, extra_args)
        
        g = gt.gradient(L, parameter)
        
    return L, g

