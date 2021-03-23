import os

import numpy

import scipy.stats

import tensorflow as tf

from adhoc_opt.optimiser import parameter_update

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot

tf.compat.v1.enable_eager_execution()

pi = tf.constant(numpy.pi, dtype='float32')

eps = 1e-6


class GP_IRT:

    def __init__(self):

        self.N_dataset = None

        self.N_model = None

        self.logit_theta = None

        self.mu_alpha = None

        self.L_alpha = None

        self.dataset_log_s2 = None

        self.model_log_s2 = None

        self.ls = None

        self.std = None

        self.e_0 = None

        self.N_sample = None

        self.N_approx = None

        self.parameter = None

    def fit(self, dataset_list, dataset, model_list, model, all_measure,
            target_measure=2,
            N_sample=1024, N_approx=32):

        self.N_dataset = len(dataset_list)

        self.N_model = len(model_list)

        self.N_sample = N_sample

        self.N_approx = N_approx

        self.logit_theta = numpy.random.randn(self.N_model)

        self.mu_alpha = numpy.zeros(self.N_dataset * self.N_approx)

        self.dataset_log_s2 = numpy.ones(self.N_dataset) * numpy.log(1e0)

        self.model_log_s2 = numpy.ones(self.N_model) * numpy.log(1e0)

        self.ls = numpy.ones(self.N_dataset) * 1e-8

        self.std = numpy.ones(self.N_dataset) * 1e8 

        all_measure[:, 3] = (1 - (all_measure[:, 3] / 2))

        all_measure[:, 4] = numpy.exp(- all_measure[:, 4])

        measure = all_measure[:, target_measure]

        measure[measure <= eps] = eps

        measure[measure >= (1.0 - eps)] = 1 - eps

        data = numpy.hstack([model.reshape(-1, 1), dataset.reshape(-1, 1), measure.reshape(-1, 1)])
        
        logit_measure = numpy.reshape(numpy.log(measure / (1 - measure)), [-1, 1])
        
        self.L_alpha = []
        
        self.e_0 = numpy.ones(self.N_dataset)
        
        for i in range(0, self.N_dataset):
            
            self.e_0[i] = numpy.min(logit_measure[data[:, 1] == i, 0])
            
            self.L_alpha.append(numpy.eye(self.N_approx).ravel() *
                                (numpy.sqrt(numpy.max(logit_measure[data[:, 1] == i, 0]) -
                                            numpy.min(logit_measure[data[:, 1] == i, 0])) + 1e-2))
            
        self.L_alpha = numpy.hstack(self.L_alpha) 
        
        parameter = numpy.hstack([self.logit_theta,
                                  self.mu_alpha, self.L_alpha,
                                  self.dataset_log_s2, self.model_log_s2,
                                  self.ls, self.std, self.e_0])

        extra_args = ('gp', (self.N_dataset, self.N_model, N_approx, N_sample, numpy.shape(data)[0]))

        parameter = parameter_update(theta_0=tf.Variable(tf.cast(parameter, 'float32')),
                                     data=data, extra_args=extra_args,
                                     obj=get_obj, obj_g=get_obj_g,
                                     lr=1e-3,
                                     batch_size=512, val_size=None, factr=0.0, tol=8192,
                                     max_batch=int(1e8))

        self.parameter = parameter

        parameter_idx = [self.N_model,
                         self.N_dataset * N_approx, self.N_dataset * N_approx * N_approx,
                         self.N_dataset, self.N_model,
                         self.N_dataset, self.N_dataset, self.N_dataset]

        self.logit_theta = parameter[:parameter_idx[0]]

        self.mu_alpha = parameter[numpy.sum(parameter_idx[:1]):numpy.sum(parameter_idx[:2])]

        self.L_alpha = parameter[numpy.sum(parameter_idx[:2]):numpy.sum(parameter_idx[:3])]

        self.dataset_log_s2 = parameter[numpy.sum(parameter_idx[:3]):numpy.sum(parameter_idx[:4])]

        self.model_log_s2 = parameter[numpy.sum(parameter_idx[:4]):numpy.sum(parameter_idx[:5])]

        self.ls = parameter[numpy.sum(parameter_idx[:5]):numpy.sum(parameter_idx[:6])]

        self.std = parameter[numpy.sum(parameter_idx[:6]):numpy.sum(parameter_idx[:7])]

        self.e_0 = parameter[numpy.sum(parameter_idx[:7]):numpy.sum(parameter_idx[:8])]

    def predict(self, dataset, model):
        
        idx = numpy.arange(0, len(dataset), 8)
        
        E = []
        
        for i in range(0, len(idx)):
            E.append(get_E_gp(self.parameter, 
                              model[idx[i]:idx[i]+8], 
                              dataset[idx[i]:idx[i]+8], 
                              self.N_dataset, 
                              self.N_model,
                              self.N_approx, 
                              self.N_sample))

        return numpy.hstack(E)

    def curve(self, did):

        return get_curve_gp(tf.cast(self.parameter, dtype='float32'),
                            did, self.N_dataset, self.N_model, self.N_approx, 4096)


def get_curve_gp(parameter, did, N_data, N_flow, N_approx, N_sample):

    parameter_idx = [N_flow, N_data * N_approx, N_data * N_approx * N_approx,
                     N_data, N_flow,
                     N_data, N_data, N_data]

    theta = tf.cast(numpy.linspace(1e-6, 1.0 - 1e-6, 128) - 0.5, dtype='float32')

    did = (numpy.ones(128) * did).astype('int')

    mu_alpha = tf.gather(tf.reshape(parameter[numpy.sum(parameter_idx[:1]):numpy.sum(parameter_idx[:2])],
                                    [N_data, N_approx]), did, axis=0)

    L_alpha = tf.gather(tf.reshape(parameter[numpy.sum(parameter_idx[:2]):numpy.sum(parameter_idx[:3])],
                                   [N_data, N_approx, N_approx]), did, axis=0)

    dataset_log_s2 = tf.gather(parameter[numpy.sum(parameter_idx[:3]):numpy.sum(parameter_idx[:4])], did, axis=0)

    # model_log_s2 = tf.gather(parameter[numpy.sum(parameter_idx[:4]):numpy.sum(parameter_idx[:5])], fid, axis=0)

    s2 = tf.reshape(tf.math.exp(dataset_log_s2), [-1, 1]) # + tf.reshape(tf.math.exp(model_log_s2), [-1, 1])

    e_0 = tf.gather(parameter[numpy.sum(parameter_idx[:7]):numpy.sum(parameter_idx[:8])], did, axis=0)

    j_index, i_index = numpy.meshgrid(numpy.arange(0, N_approx) + 1, numpy.arange(0, N_approx) + 1)

    psi_mat = psi(tf.reshape(theta, [-1, 1, 1]),
                  tf.constant(i_index.reshape(1, N_approx, N_approx), dtype='float32'),
                  tf.constant(j_index.reshape(1, N_approx, N_approx), dtype='float32'))

    raw_alpha_samples = tf.random.normal([len(did), N_sample, N_approx], dtype='float32')

    alpha_samples = tf.matmul(raw_alpha_samples, L_alpha) + tf.reshape(mu_alpha, [-1, 1, N_approx])

    e_samples = tf.reduce_mean(tf.matmul(alpha_samples, psi_mat) * alpha_samples, axis=1) + tf.reshape(e_0, [-1, 1])

    mu = numpy.mean(e_samples.numpy(), axis=1)

    var_e = numpy.var(e_samples.numpy(), axis=1)

    s2_hat = s2.numpy().ravel() + var_e.ravel()

    s_hat = numpy.sqrt(s2_hat)

    E = 1 - 1 / (1 + numpy.exp((mu / numpy.sqrt(1 + numpy.pi * numpy.square(s_hat) / 8))))

    E_up = 1 - 1 / (1 + numpy.exp(scipy.stats.norm.ppf(q=0.75, loc=mu, scale=s_hat)))

    E_mid = 1 - 1 / (1 + numpy.exp(scipy.stats.norm.ppf(q=0.5, loc=mu, scale=s_hat)))

    E_low = 1 - 1 / (1 + numpy.exp(scipy.stats.norm.ppf(q=0.25, loc=mu, scale=s_hat)))

    return E, E_up, E_mid, E_low
    
    
def get_E_gp(parameter, fid, did, N_data, N_flow, N_approx, N_sample):
    
    parameter_idx = [N_flow, N_data * N_approx, N_data * N_approx * N_approx,
                     N_data, N_flow,
                     N_data, N_data, N_data]
    
    logit_theta = tf.gather(parameter[:parameter_idx[0]], fid, axis=0)

    mu_alpha = tf.gather(tf.reshape(parameter[numpy.sum(parameter_idx[:1]):numpy.sum(parameter_idx[:2])],
                                    [N_data, N_approx]), did, axis=0)

    L_alpha = tf.gather(tf.reshape(parameter[numpy.sum(parameter_idx[:2]):numpy.sum(parameter_idx[:3])],
                                   [N_data, N_approx, N_approx]), did, axis=0)

    dataset_log_s2 = tf.gather(parameter[numpy.sum(parameter_idx[:3]):numpy.sum(parameter_idx[:4])], did, axis=0)

    dataset_s2 = tf.square(tf.exp(dataset_log_s2))

    model_log_s2 = tf.gather(parameter[numpy.sum(parameter_idx[:4]):numpy.sum(parameter_idx[:5])], fid, axis=0)

    model_s2 = tf.square(tf.exp(model_log_s2))

    s2 = dataset_s2 + model_s2

    e_0 = tf.gather(parameter[numpy.sum(parameter_idx[:7]):numpy.sum(parameter_idx[:8])], did, axis=0)

    j_index, i_index = numpy.meshgrid(numpy.arange(0, N_approx) + 1, numpy.arange(0, N_approx) + 1)

    theta = tf.clip_by_value(1 / (1 + tf.math.exp(logit_theta)),
                             tf.constant(eps, dtype='float32'),
                             tf.constant(1 - eps, dtype='float32')) - 0.5

    psi_mat = psi(tf.reshape(theta, [-1, 1, 1]),
                  tf.constant(i_index.reshape(1, N_approx, N_approx), dtype='float32'),
                  tf.constant(j_index.reshape(1, N_approx, N_approx), dtype='float32'))

    raw_alpha_samples = tf.random.normal([len(did), N_sample, N_approx], dtype='float32')
    
    alpha_samples = tf.matmul(raw_alpha_samples, L_alpha) + tf.reshape(mu_alpha, [-1, 1, N_approx])
    
    e_samples = tf.reduce_mean(tf.matmul(alpha_samples, psi_mat) * alpha_samples, axis=1) + tf.reshape(e_0, [-1, 1])

    mu = numpy.mean(e_samples.numpy(), axis=1)

    var_e = numpy.var(e_samples.numpy(), axis=1)

    s2_hat = s2.numpy().ravel() + var_e.ravel()

    s_hat = numpy.sqrt(s2_hat)

    E = 1 - 1 / (1 + numpy.exp((mu / numpy.sqrt(1 + numpy.pi * numpy.square(s_hat) / 8))))

    return E
    

def vi_obj(parameter, fid, did, measure, N_data, N_flow, N_approx, N_sample, N_total):

    parameter_idx = [N_flow, N_data * N_approx, N_data * N_approx * N_approx,
                     N_data, N_flow,
                     N_data, N_data, N_data]

    logit_theta = tf.gather(parameter[:parameter_idx[0]], fid, axis=0)

    mu_alpha = tf.gather(tf.reshape(parameter[numpy.sum(parameter_idx[:1]):numpy.sum(parameter_idx[:2])],
                                    [N_data, N_approx]), did, axis=0)

    L_alpha = tf.gather(tf.reshape(parameter[numpy.sum(parameter_idx[:2]):numpy.sum(parameter_idx[:3])],
                                   [N_data, N_approx, N_approx]), did, axis=0)

    dataset_log_s2 = tf.gather(parameter[numpy.sum(parameter_idx[:3]):numpy.sum(parameter_idx[:4])], did, axis=0)

    model_log_s2 = tf.gather(parameter[numpy.sum(parameter_idx[:4]):numpy.sum(parameter_idx[:5])], fid, axis=0)

    s2 = tf.reshape(tf.math.exp(dataset_log_s2), [-1, 1]) + tf.reshape(tf.math.exp(model_log_s2), [-1, 1])

    e_0 = tf.gather(parameter[numpy.sum(parameter_idx[:7]):numpy.sum(parameter_idx[:8])], did, axis=0)

    j_index, i_index = numpy.meshgrid(numpy.arange(0, N_approx) + 1, numpy.arange(0, N_approx) + 1)

    theta = tf.clip_by_value(1 / (1 + tf.math.exp(logit_theta)),
                             tf.constant(eps, dtype='float32'),
                             tf.constant(1 - eps, dtype='float32')) - 0.5

    psi_mat = psi(tf.reshape(theta, [-1, 1, 1]),
                  tf.constant(i_index.reshape(1, N_approx, N_approx), dtype='float32'),
                  tf.constant(j_index.reshape(1, N_approx, N_approx), dtype='float32'))

    raw_alpha_samples = tf.random.normal([len(did), N_sample, N_approx], dtype='float32')

    logit_measure = tf.math.log(measure / (1 - measure))
    
    alpha_samples = tf.matmul(raw_alpha_samples, L_alpha) + tf.reshape(mu_alpha, [-1, 1, N_approx])
    
    e_samples = tf.reduce_mean(tf.matmul(alpha_samples, psi_mat) * alpha_samples, axis=2) + tf.reshape(e_0, [-1, 1])
    
    diff = logit_measure - e_samples
    
    exp = tf.math.exp(- 0.5 * tf.math.square(diff) / s2)
    
    sample_lik = 1 / (tf.math.sqrt(2 * pi * s2)) * exp * (1 / (tf.reshape(measure, [-1, 1]) *
                                                               (1 - tf.reshape(measure, [-1, 1]))))
    
    link_log_lik = tf.math.log(tf.reduce_mean(sample_lik + tf.constant(1e-16, dtype='float32'), axis=1))

    res = - tf.reduce_mean(link_log_lik)

    # d_unique = numpy.unique(did)
    #
    # mu_alpha_unique = tf.gather(tf.reshape(parameter[numpy.sum(parameter_idx[:1]):numpy.sum(parameter_idx[:2])],
    #                                        [N_data, N_approx]), d_unique, axis=0)
    #
    # L_alpha_unique = tf.gather(tf.reshape(parameter[numpy.sum(parameter_idx[:2]):numpy.sum(parameter_idx[:3])],
    #                                       [N_data, N_approx, N_approx]), d_unique, axis=0)
    #
    # ls = tf.gather(parameter[numpy.sum(parameter_idx[:4]):numpy.sum(parameter_idx[:5])], d_unique, axis=0)
    #
    # std = tf.gather(parameter[numpy.sum(parameter_idx[:5]):numpy.sum(parameter_idx[:6])], d_unique, axis=0)
    #
    # kl = []
    #
    # for i in range(0, len(d_unique)):
    #     kl.append(get_normal_kl(mu_alpha_unique[i, :], L_alpha_unique[i, :, :], ls[i], std[i], N_approx))
    #
    # res = res + tf.reduce_sum(kl)

    return res


def get_normal_kl(mu_alpha, L_alpha, ls, std, N_approx, L=0.5):

    V_alpha = tf.linalg.matmul(tf.transpose(L_alpha), L_alpha)

    C = tf.linalg.diag(S_RBF(numpy.sqrt(lambd(numpy.arange(0, N_approx) + 1, L)), ls) * tf.math.square(std) +
                       tf.constant(eps, dtype='float32'))

    kl = -0.5 * tf.linalg.slogdet(V_alpha)[1] + \
        0.5 * tf.linalg.slogdet(C)[1] + \
        0.5 * tf.matmul(tf.reshape(mu_alpha, [1, -1]), tf.matmul(tf.linalg.inv(C), tf.reshape(mu_alpha, [-1, 1]))) + \
        0.5 * tf.linalg.trace(tf.matmul(tf.linalg.inv(C), V_alpha))

    return kl


def S_RBF(omega, ls):
    return 2 * pi * tf.math.square(ls) * tf.math.exp(-2 * tf.math.square(pi) *
                                                     tf.math.square(ls) * tf.math.square(omega))


def lambd(j, L=0.5):
    return j * pi / (2 * L)


def phi(x, j, L=0.5):
    lambd_sqrt = j * pi / (2 * L)
    return tf.math.sqrt(1 / L) * tf.math.sin(lambd_sqrt * (x + L))


def psi(x, i, j, L=0.5):
    lambd_sqrt_neg = i * pi / (2 * L) - j * pi / (2 * L) + tf.cast(i == j, 'float32')
    lambd_sqrt_pos = i * pi / (2 * L) + j * pi / (2 * L)
    return tf.cast(i == j, 'float32') * ((1 / (2 * L)) * (x + L) -
                                         tf.math.sin(lambd_sqrt_pos * (x + L)) / (2 * L * lambd_sqrt_pos)) \
        + tf.cast(i != j, 'float32') * (tf.math.sin(lambd_sqrt_neg * (x + L)) / (2 * L * lambd_sqrt_neg) -
                                        tf.math.sin(lambd_sqrt_pos * (x + L)) / (2 * L * lambd_sqrt_pos))


class Beta_3_IRT:

    def __init__(self):

        self.N_dataset = None

        self.N_model = None

        self.logit_theta = None

        self.logit_delta = None

        self.log_a = None

        self.parameter = None

    def fit(self, dataset_list, dataset, model_list, model, all_measure, target_measure=2):

        self.N_dataset = len(dataset_list)

        self.N_model = len(model_list)

        self.logit_theta = numpy.random.randn(self.N_model)

        self.logit_delta = numpy.random.randn(self.N_dataset)

        self.log_a = numpy.random.randn(self.N_dataset)

        parameter = tf.cast(numpy.hstack([self.logit_theta, self.logit_delta, self.log_a]), 'float32')

        all_measure[:, 3] = (1 - (all_measure[:, 3] / 2))

        all_measure[:, 4] = numpy.exp(- all_measure[:, 4])

        measure = all_measure[:, target_measure]

        measure[measure <= eps] = eps

        measure[measure >= (1.0 - eps)] = 1 - eps

        data = numpy.hstack([model.reshape(-1, 1), dataset.reshape(-1, 1), measure.reshape(-1, 1)])

        extra_args = ('beta3', (self.N_dataset, self.N_model))

        parameter = parameter_update(theta_0=tf.Variable(parameter), data=data, extra_args=extra_args,
                                     obj=get_obj, obj_g=get_obj_g,
                                     lr=1e-3,
                                     batch_size=512, val_size=None, factr=0.0, tol=8192,
                                     max_batch=int(1e8))

        self.parameter = parameter

        self.logit_theta = parameter[:self.N_model]

        self.logit_delta = parameter[self.N_model:self.N_model + self.N_dataset]

        self.log_a = parameter[self.N_model + self.N_dataset:self.N_model + 2 * self.N_dataset]

    def predict(self, dataset, model):

        idx = numpy.arange(0, len(dataset), 8)
        
        E = []
        
        for i in range(0, len(idx)):
            E.append(get_E_beta_3(self.parameter, 
                                  model[idx[i]:idx[i]+8], 
                                  dataset[idx[i]:idx[i]+8], 
                                  self.N_dataset, 
                                  self.N_model))

        return numpy.hstack(E)

    def curve(self, d_id):

        return get_curve_beta3(self.parameter, d_id, self.N_dataset, self.N_model)


class Logistic_IRT:

    def __init__(self):

        self.N_dataset = None

        self.N_model = None

        self.parameter = None

        self.theta = None

        self.delta = None

        self.log_a = None

        self.log_s2 = None

    def fit(self, dataset_list, dataset, model_list, model, all_measure, target_measure=0):

        self.N_dataset = len(dataset_list)

        self.N_model = len(model_list)

        self.theta = numpy.random.randn(self.N_model)

        self.delta = numpy.random.randn(self.N_dataset)

        self.log_a = numpy.random.randn(self.N_dataset)

        self.log_s2 = numpy.random.randn(self.N_dataset)

        parameter = tf.cast(numpy.hstack([self.theta, self.delta, self.log_a, self.log_s2]), 'float32')

        all_measure[:, 3] = (1 - (all_measure[:, 3] / 2))

        all_measure[:, 4] = numpy.exp(- all_measure[:, 4])

        measure = all_measure[:, target_measure]

        measure[measure <= eps] = eps

        measure[measure >= (1.0 - eps)] = 1 - eps

        data = numpy.hstack([model.reshape(-1, 1), dataset.reshape(-1, 1), measure.reshape(-1, 1)])

        extra_args = ('logistic', (self.N_dataset, self.N_model))

        parameter = parameter_update(theta_0=tf.Variable(parameter), data=data, extra_args=extra_args,
                                     obj=get_obj, obj_g=get_obj_g,
                                     lr=1e-3,
                                     batch_size=512, val_size=None, factr=0.0, tol=8192,
                                     max_batch=int(1e8))

        self.parameter = parameter

        self.theta = parameter[:self.N_model]

        self.delta = parameter[self.N_model:self.N_model + self.N_dataset]

        self.log_a = parameter[self.N_model + self.N_dataset:self.N_model + 2 * self.N_dataset]

        self.log_s2 = parameter[self.N_model + 2 * self.N_dataset:self.N_model + 3 * self.N_dataset]

    def predict(self, dataset, model):
        
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

        return get_curve_logistic(self.parameter, d_id, self.N_dataset, self.N_model)


def get_curve_logistic(parameter, did, N_data, N_flow):

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
    
    return - tf.reduce_mean(loglik)


def get_E_logistic(parameter, fid, did, N_data, N_flow):

    theta = tf.gather(parameter[:N_flow], fid, axis=0)

    delta = tf.gather(parameter[N_flow:N_flow + N_data], did, axis=0)

    a = tf.math.exp(tf.gather(parameter[N_flow + N_data:N_flow + 2 * N_data], did, axis=0))

    log_s2 = tf.gather(parameter[N_flow + 2 * N_data:N_flow + 3 * N_data], did, axis=0)
    
    s2 = tf.math.exp(log_s2) + tf.constant(eps, dtype='float32')

    s = tf.math.sqrt(s2)

    mu = - a * (theta - delta)

    samples = scipy.stats.norm.rvs(loc=mu, scale=s, size=[1024, len(mu)])

    return numpy.mean(1 / (1 + numpy.exp(samples)), axis=0)


def get_curve_beta3(parameter, did, N_data, N_flow):

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
             
    return - tf.reduce_mean(loglik)


def get_E_beta_3(parameter, fid, did, N_data, N_flow):

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

    return alpha / (alpha + beta)


def get_obj(parameter, data, extra_args):
    
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
            
    elif irt_type == 'gp':

        N_data, N_flow, N_approx, N_sample, N_total = tmp_args
        
        L = vi_obj(parameter, fid, did, measure, N_data, N_flow, N_approx, N_sample, N_total)

    else:

        L = None
    
    return L


def get_obj_g(parameter, data, extra_args):
    
    with tf.GradientTape() as gt:
        
        gt.watch(parameter)
        
        L = get_obj(parameter, data, extra_args)
        
        g = gt.gradient(L, parameter)
        
    return L, g
