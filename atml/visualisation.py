import numpy
import pandas
import scipy.stats
import scipy.special
import matplotlib.pyplot
import tensorflow as tf
from irt import Beta_3_IRT, Logistic_IRT, GP_IRT

import matplotlib
matplotlib.use('Agg')
tf.compat.v1.enable_eager_execution()

m_list = ['Acc', 'BS', 'LL', 'B-Acc', 'AUC', 'F1']

eps = 1e-6


def get_beta3_curve(data_idx, data_ref, beta_mdl, res, measure):
    tmp_E, E_up, E_mid, E_low = beta_mdl.curve(data_idx)
    matplotlib.pyplot.figure(dpi=256, figsize=(4, 3))
    matplotlib.pyplot.plot(numpy.linspace(-0.5, 0.5, 128) + 0.5, tmp_E, 'b')
    matplotlib.pyplot.plot(numpy.linspace(-0.5, 0.5, 128) + 0.5, E_up, 'b--')
    matplotlib.pyplot.plot(numpy.linspace(-0.5, 0.5, 128) + 0.5, E_mid, 'b--')
    matplotlib.pyplot.plot(numpy.linspace(-0.5, 0.5, 128) + 0.5, E_low, 'b--')
    tmp_idx = (res['data_idx'].to_numpy() == data_idx)
    tmp_theta = 1 / (1 + numpy.exp(beta_mdl.logit_theta.numpy()[res['model_idx'].to_numpy()[tmp_idx].astype('int')]))
    matplotlib.pyplot.plot(tmp_theta, measure.transform(res[measure.name].to_numpy())[tmp_idx], 'ko', alpha=0.5)
    matplotlib.pyplot.xlim([-0.05, 1.05])
    matplotlib.pyplot.xlabel('model ability')
    matplotlib.pyplot.title('beta:  ' + data_ref)
    matplotlib.pyplot.ylim([-0.05, 1.05])
    matplotlib.pyplot.ylabel('response:  ' + measure.name)
    matplotlib.pyplot.grid()


def get_beta3_figures(target_measure=2):

    res = pandas.read_csv('./res/res_sparse.csv', header=None).to_numpy()

    tiny = numpy.finfo('float64').tiny
    res[res[:, 2] >= 1, 2] = 1
    res[res[:, 2] <= 0, 2] = 0
    res[:, 3] = (1 - (res[:, 3] / 2))
    res[:, 4] = (- res[:, 4] - numpy.log(tiny)) / (-numpy.log(tiny))
    res[res[:, 5] >= 1, 5] = 1
    res[res[:, 5] <= 0, 5] = 0
    res[res[:, 6] >= 1, 6] = 1
    res[res[:, 6] <= 0, 6] = 0
    res[res[:, 7] >= 1, 7] = 1
    res[res[:, 7] <= 0, 7] = 0

    logit_theta = numpy.loadtxt(fname='./irt/' + str(target_measure) + '_logit_theta_beta3.csv', delimiter=',')
    logit_delta = numpy.loadtxt(fname='./irt/' + str(target_measure) + '_logit_delta_beta3.csv', delimiter=',')
    log_a = numpy.loadtxt(fname='./irt/' + str(target_measure) + '_log_a_beta3.csv', delimiter=',')

    theta = 1 / (1 + numpy.exp(logit_theta))
    delta = 1 / (1 + numpy.exp(logit_delta))
    a = numpy.exp(log_a)

    matplotlib.pyplot.figure(figsize=(16, 9))
    matplotlib.pyplot.bar(numpy.arange(0, len(theta)), theta)
    matplotlib.pyplot.xticks(numpy.arange(0, len(model_list)), model_list, rotation=90);
    matplotlib.pyplot.xlim([-1, len(model_list)])
    matplotlib.pyplot.grid()
    matplotlib.pyplot.title(m_list[target_measure] + ':      ability')
    matplotlib.pyplot.savefig('./figures/irt/' + str(target_measure) + '_' + 'ability_beta3.png')
    matplotlib.pyplot.close()

    matplotlib.pyplot.figure(figsize=(32, 9))
    matplotlib.pyplot.bar(numpy.arange(0, len(delta)), delta)
    matplotlib.pyplot.xticks(numpy.arange(0, len(dataset_list)), dataset_list, rotation=90);
    matplotlib.pyplot.grid()
    matplotlib.pyplot.title(m_list[target_measure] + ':      difficulty')
    matplotlib.pyplot.savefig('./figures/irt/' + str(target_measure) + '_' + 'difficulty_beta3.png')
    matplotlib.pyplot.close()

    matplotlib.pyplot.figure(figsize=(32, 9))
    matplotlib.pyplot.bar(numpy.arange(0, len(a)), a)
    matplotlib.pyplot.xticks(numpy.arange(0, len(dataset_list)), dataset_list, rotation=90);
    matplotlib.pyplot.grid()
    matplotlib.pyplot.title(m_list[target_measure] + ':      discrimination')
    matplotlib.pyplot.savefig('./figures/irt/' + str(target_measure) + '_' + 'discrimination_beta3.png')
    matplotlib.pyplot.close()

    v_theta = numpy.linspace(1e-2, 1 - 1e-2, 1024).reshape(1, -1)
    v_r = numpy.linspace(1e-2, 1 - 1e-2, 1024).reshape(1, -1)
    v_delta = delta.reshape(-1, 1)
    v_a = a.reshape(-1, 1)
    v_delta[v_delta == 1] = 1 - 1e-2
    v_delta[v_delta == 0] = 1e-2

    vv_theta, vv_r = numpy.meshgrid(v_theta, v_r)

    for i in range(0, len(dataset_list)):
        matplotlib.pyplot.figure(figsize=(10, 10))
        alpha = numpy.power(vv_theta.ravel() / v_delta[i], v_a[i]) + 1e-6
        beta = numpy.power((1 - vv_theta.ravel()) / (1 - v_delta[i]), v_a[i]) + 1e-6
        loglik = (alpha - 1) * tf.math.log(vv_r.ravel()) + (beta - 1) * tf.math.log(1 - vv_r.ravel()) - \
                 (tf.math.lgamma(alpha) + tf.math.lgamma(beta) - tf.math.lgamma(alpha + beta))
        lik = numpy.exp(loglik.numpy().reshape(1024, 1024))
        matplotlib.pyplot.imshow(lik, origin='lower', cmap='copper', vmin=0, vmax=8.0, extent=[0, 1, 0, 1], aspect=1)
        idx = numpy.argwhere(res[:, 1] == i)
        matplotlib.pyplot.plot(v_theta.ravel(), numpy.mean((alpha / (alpha + beta)).reshape(1024, 1024), axis=0), 'y--',
                               linewidth=5, alpha=0.85)
        matplotlib.pyplot.plot(theta[res[idx, 0].astype('int')], res[idx, target_measure], 'yo', alpha=0.5)
        for j in range(0, 9):
            tmp_idx = numpy.argwhere((res[:, 0] >= j * 8) & (res[:, 0] <= (j * 8 + 7)) & (res[:, 1] == i))
            mu_r = numpy.mean(res[tmp_idx, target_measure])
            matplotlib.pyplot.text(numpy.mean(theta[j * 8:(j * 8 + 7)]), mu_r, model_list[j * 8].split('-')[0], c='w',
                                   fontsize=15,
                                   fontweight='extra bold')
        matplotlib.pyplot.title(m_list[target_measure] + '      ' +
                                dataset_list[i] + ', dif: ' + str(numpy.around(delta[i], 2)) +
                                ', dis:' + str(numpy.around(v_a[i][0], 2)))
        matplotlib.pyplot.xlabel('ability')
        matplotlib.pyplot.ylabel('response')
        matplotlib.pyplot.savefig('./figures/rc/' + str(target_measure) + '_' + dataset_list[i] + '_beta3.png')
        matplotlib.pyplot.close()


def get_logistic_curve(data_idx, data_ref, logistic_mdl, res, measure):
    theta_edge = numpy.max(numpy.abs(logistic_mdl.theta)) * 2
    tmp_E, E_up, E_mid, E_low = logistic_mdl.curve(data_idx)
    matplotlib.pyplot.figure(dpi=256, figsize=(4, 3))
    matplotlib.pyplot.plot(numpy.linspace(-theta_edge, theta_edge, 128), tmp_E, 'b')
    matplotlib.pyplot.plot(numpy.linspace(-theta_edge, theta_edge, 128), E_up, 'b--')
    matplotlib.pyplot.plot(numpy.linspace(-theta_edge, theta_edge, 128), E_mid, 'b--')
    matplotlib.pyplot.plot(numpy.linspace(-theta_edge, theta_edge, 128), E_low, 'b--')
    tmp_idx = (res['data_idx'].to_numpy() == data_idx)
    tmp_theta = logistic_mdl.theta.numpy()[res['model_idx'].to_numpy()[tmp_idx].astype('int')]
    matplotlib.pyplot.plot(tmp_theta, measure.transform(res[measure.name].to_numpy())[tmp_idx], 'ko', alpha=0.5)
    matplotlib.pyplot.xlim([-theta_edge, theta_edge])
    matplotlib.pyplot.xlabel('model ability')
    matplotlib.pyplot.title('logistic:  ' + data_ref)
    matplotlib.pyplot.ylim([-0.05, 1.05])
    matplotlib.pyplot.ylabel('response:  ' + measure.name)
    matplotlib.pyplot.grid()


def get_logistic_figures(target_measure=2):

    res = pandas.read_csv('./res/res_sparse.csv', header=None).to_numpy()

    tiny = numpy.finfo('float64').tiny
    res[res[:, 2] >= 1, 2] = 1
    res[res[:, 2] <= 0, 2] = 0
    res[:, 3] = (1 - (res[:, 3] / 2))
    res[:, 4] = (- res[:, 4] - numpy.log(tiny)) / (-numpy.log(tiny))
    res[res[:, 5] >= 1, 5] = 1
    res[res[:, 5] <= 0, 5] = 0
    res[res[:, 6] >= 1, 6] = 1
    res[res[:, 6] <= 0, 6] = 0
    res[res[:, 7] >= 1, 7] = 1
    res[res[:, 7] <= 0, 7] = 0

    theta = numpy.loadtxt(fname='./irt/' + str(target_measure) + '_theta_logistic.csv', delimiter=',')
    delta = numpy.loadtxt(fname='./irt/' + str(target_measure) + '_delta_logistic.csv', delimiter=',')
    log_a = numpy.loadtxt(fname='./irt/' + str(target_measure) + '_log_a_logistic.csv', delimiter=',')
    log_s2 = numpy.loadtxt(fname='./irt/' + str(target_measure) + '_log_s2_logistic.csv', delimiter=',')

    matplotlib.pyplot.figure(figsize=(16, 9))
    matplotlib.pyplot.bar(numpy.arange(0, len(theta)), theta)
    matplotlib.pyplot.xticks(numpy.arange(0, len(model_list)), model_list, rotation=90);
    matplotlib.pyplot.xlim([-1, len(model_list)])
    matplotlib.pyplot.grid()
    matplotlib.pyplot.title(m_list[target_measure] + ':      ability')
    matplotlib.pyplot.savefig('./figures/irt/' + str(target_measure) + '_' + 'ability_logistic.png')
    matplotlib.pyplot.close()

    matplotlib.pyplot.figure(figsize=(32, 9))
    matplotlib.pyplot.bar(numpy.arange(0, len(delta)), delta)
    matplotlib.pyplot.xticks(numpy.arange(0, len(dataset_list)), dataset_list, rotation=90);
    matplotlib.pyplot.grid()
    matplotlib.pyplot.title(m_list[target_measure] + ':      difficulty')
    matplotlib.pyplot.savefig('./figures/irt/' + str(target_measure) + '_' + 'difficulty_logistic.png')
    matplotlib.pyplot.close()

    matplotlib.pyplot.figure(figsize=(32, 9))
    matplotlib.pyplot.bar(numpy.arange(0, len(log_a)), numpy.exp(log_a))
    matplotlib.pyplot.xticks(numpy.arange(0, len(dataset_list)), dataset_list, rotation=90);
    matplotlib.pyplot.grid()
    matplotlib.pyplot.title(m_list[target_measure] + ':      discrimination')
    matplotlib.pyplot.savefig('./figures/irt/' + str(target_measure) + '_' + 'discrimination_logistic.png')
    matplotlib.pyplot.close()

    matplotlib.pyplot.figure(figsize=(32, 9))
    matplotlib.pyplot.bar(numpy.arange(0, len(log_s2)), numpy.sqrt(numpy.exp(log_s2) + 1e-2))
    matplotlib.pyplot.xticks(numpy.arange(0, len(dataset_list)), dataset_list, rotation=90);
    matplotlib.pyplot.grid()
    matplotlib.pyplot.title(m_list[target_measure] + ':      std')
    matplotlib.pyplot.savefig('./figures/irt/' + str(target_measure) + '_' + 'std_logistic.png')
    matplotlib.pyplot.close()

    d_theta = (numpy.max(theta) - numpy.min(theta)) * 0.25
    v_theta = numpy.linspace(numpy.min(theta) - d_theta, numpy.max(theta) + d_theta, 1024).reshape(1, -1)
    v_delta = delta.reshape(-1, 1)
    v_a = numpy.exp(log_a).reshape(-1, 1)
    v_s = numpy.sqrt(numpy.exp(log_s2) + 1e-2).reshape(-1, 1)
    v_r = numpy.linspace(1e-8, 1 - 1e-8, 1024).reshape(1, -1)

    vv_theta, vv_r = numpy.meshgrid(v_theta, v_r)
    vv_logit_r = tf.math.log((1 - vv_r) / vv_r).numpy()
    pi = tf.constant(numpy.pi, dtype='float64')

    for i in range(0, len(dataset_list)):
        matplotlib.pyplot.figure(figsize=(10, 10))
        mu = - v_a[i] * (vv_theta.ravel() - v_delta[i])
        s2 = tf.math.square(v_s[i])
        diff = vv_logit_r.ravel() - mu
        exp = tf.math.exp(- 0.5 * tf.math.square(diff) / s2)
        sample_lik = 1 / (tf.math.sqrt(2 * pi * s2)) * exp * (1 / (vv_r.ravel() * (1 - vv_r.ravel())))
        loglik = tf.math.log(sample_lik + tf.constant(1e-6, dtype='float64'))
        # loglik = 0.5 * tf.math.log(1 / (2 * pi)) + 0.5 * tf.math.log(1 / s2) \
        #          - 0.5 * tf.math.square(vv_logit_r.ravel() - mu) / s2 \
        #          + tf.math.log(1 / (vv_r.ravel() * (1 - vv_r.ravel())))
        lik = numpy.exp(loglik.numpy().reshape(1024, 1024))
        matplotlib.pyplot.imshow(lik, origin='lower', cmap='copper', vmin=0, vmax=8.0,
                                 extent=[numpy.min(theta) - d_theta,
                                         numpy.max(theta) + d_theta,
                                         0, 1],
                                 aspect=(d_theta*6.0))
        idx = (res[:, 1] == i)
        samples = scipy.stats.norm.rvs(loc=numpy.mean(mu.reshape(1024, 1024), axis=0),
                                       scale=numpy.mean(numpy.sqrt(s2).repeat(1024 ** 2).reshape(1024, 1024), axis=0),
                                       size=[2 ** 16, 1024])
        logistic_samples = numpy.mean(1 / (1 + numpy.exp(samples)), axis=0)
        matplotlib.pyplot.plot(v_theta.ravel(), logistic_samples, 'y--',
                               linewidth=5, alpha=0.85)
        matplotlib.pyplot.plot(theta[res[idx, 0].astype('int')], res[idx, target_measure], 'yo', alpha=0.5)
        for j in range(0, 9):
            tmp_idx = numpy.argwhere((res[:, 0] >= j * 8) & (res[:, 0] <= (j * 8 + 7)) & (res[:, 1] == i))
            mu_r = numpy.mean(res[tmp_idx, target_measure])
            matplotlib.pyplot.text(numpy.mean(theta[j * 8:(j * 8 + 7)]), mu_r, model_list[j * 8].split('-')[0], c='w',
                                   fontsize=15,
                                   fontweight='extra bold')
        matplotlib.pyplot.title(m_list[target_measure] + '      ' + dataset_list[i] + ', dif: '
                                + str(numpy.around(delta[i], 2)) + ', dis:' + str(numpy.around(v_a[i][0], 2)))
        matplotlib.pyplot.xlabel('ability')
        matplotlib.pyplot.ylabel('response')
        matplotlib.pyplot.savefig('./figures/rc/' + str(target_measure) + '_' + dataset_list[i] + '_logistic.png')
        matplotlib.pyplot.close()


def get_gp_curve(target_measure=2):

    res = pandas.read_csv('./res/res_sparse.csv', header=None).to_numpy()

    res[:, 3] = (1 - (res[:, 3] / 2))

    res[:, 4] = numpy.exp(- res[:, 4])

    measure = res[:, target_measure]

    measure[measure <= eps] = eps

    measure[measure >= (1.0 - eps)] = 1 - eps

    gp_mdl = GP_IRT()
    gp_mdl.N_dataset = len(dataset_list)
    gp_mdl.N_model = len(model_list)
    gp_mdl.N_sample = 1024
    gp_mdl.N_approx = 32

    gp_mdl.logit_theta = numpy.loadtxt('./irt/' + str(target_measure) + '_logit_theta_gp.csv', delimiter=',')
    gp_mdl.mu_alpha = numpy.loadtxt('./irt/' + str(target_measure) + '_mu_alpha_gp.csv', delimiter=',')
    gp_mdl.L_alpha = numpy.loadtxt('./irt/' + str(target_measure) + '_L_alpha_gp.csv', delimiter=',')
    gp_mdl.dataset_log_s2 = numpy.loadtxt('./irt/' + str(target_measure) + '_dataset_log_s2_gp.csv', delimiter=',')
    gp_mdl.model_log_s2 = numpy.loadtxt('./irt/' + str(target_measure) + '_model_log_s2_gp.csv', delimiter=',')
    gp_mdl.ls = numpy.loadtxt('./irt/' + str(target_measure) + '_ls_gp.csv', delimiter=',')
    gp_mdl.std = numpy.loadtxt('./irt/' + str(target_measure) + '_std_gp.csv', delimiter=',')
    gp_mdl.e_0 = numpy.loadtxt('./irt/' + str(target_measure) + '_e_0_gp.csv', delimiter=',')

    parameter = numpy.hstack([gp_mdl.logit_theta.ravel(),
                              gp_mdl.mu_alpha.ravel(),
                              gp_mdl.L_alpha.ravel(),
                              gp_mdl.dataset_log_s2.ravel(),
                              gp_mdl.model_log_s2.ravel(),
                              gp_mdl.ls.ravel(),
                              gp_mdl.std.ravel(),
                              gp_mdl.e_0.ravel()])

    gp_mdl.parameter = parameter

    for d_id in range(0, len(dataset_list)):
        tmp_E, E_up, E_mid, E_low = gp_mdl.curve(d_id)
        matplotlib.pyplot.figure(dpi=256, figsize=(4, 3))
        matplotlib.pyplot.plot(numpy.linspace(-0.5, 0.5, 128) + 0.5,
                               tmp_E, 'b')
        matplotlib.pyplot.plot(numpy.linspace(-0.5, 0.5, 128) + 0.5,
                               E_up, 'b--')
        matplotlib.pyplot.plot(numpy.linspace(-0.5, 0.5, 128) + 0.5,
                               E_mid, 'b--')
        matplotlib.pyplot.plot(numpy.linspace(-0.5, 0.5, 128) + 0.5,
                               E_low, 'b--')
        tmp_idx = (res[:, 1] == d_id)
        tmp_theta = 1 / (1 + numpy.exp(gp_mdl.logit_theta[res[tmp_idx, 0].astype('int')]))
        matplotlib.pyplot.plot(tmp_theta, measure[tmp_idx], 'ko', alpha=0.05)
        tmp_theta_0 = 1 / (1 + numpy.exp(gp_mdl.logit_theta[0]))
        matplotlib.pyplot.plot([tmp_theta_0, tmp_theta_0], [-0.05, 1.05], 'r-x')
        tmp_theta_1 = 1 / (1 + numpy.exp(gp_mdl.logit_theta[-9]))
        matplotlib.pyplot.plot([tmp_theta_1, tmp_theta_1], [-0.05, 1.05], 'g-x')
        matplotlib.pyplot.xlim([-0.05, 1.05])
        matplotlib.pyplot.xlabel('model ability')
        matplotlib.pyplot.title('mgp:  ' + dataset_list[d_id])
        matplotlib.pyplot.ylim([-0.05, 1.05])
        matplotlib.pyplot.ylabel('response:  ' + m_list[target_measure - 2])
        matplotlib.pyplot.xticks([tmp_theta_0, tmp_theta_1], ['MLP', 'NB'], rotation=-90)
        matplotlib.pyplot.grid()
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.savefig('./figures/rc/' + str(target_measure) + '_' + str(d_id) + '_gp.png')
        matplotlib.pyplot.close()


def get_cat_figures(target_measure, test_mdl_class):

    cat_list = ['logistic_fisher_norep', 'logistic_fisher_rep', 'logistic_kl_norep', 'logistic_kl_rep',
                'beta3_fisher_norep', 'beta3_fisher_rep', 'beta3_kl_norep', 'beta3_kl_rep']

    idx = numpy.zeros([8, len(dataset_list)])
    mse = numpy.zeros([8, len(dataset_list) + 1])
    nll = numpy.zeros([8, len(dataset_list) + 1])

    idx[0, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'logistic' + '_' + 'fisher' +
                           '_' + str(True) + '_idx.npy')
    idx[1, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'logistic' + '_' + 'fisher' +
                           '_' + str(False) + '_idx.npy')
    idx[2, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'logistic' + '_' + 'kl' + '_'
                           + str(True) + '_idx.npy')
    idx[3, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'logistic' + '_' + 'kl' + '_'
                           + str(False) + '_idx.npy')

    idx[4, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'beta3' + '_' + 'fisher' + '_'
                           + str(True) + '_idx.npy')
    idx[5, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'beta3' + '_' + 'fisher' + '_'
                           + str(False) + '_idx.npy')
    idx[6, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'beta3' + '_' + 'kl' + '_' +
                           str(True) + '_idx.npy')
    idx[7, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'beta3' + '_' + 'kl' + '_' +
                           str(False) + '_idx.npy')

    idx = numpy.hstack([numpy.zeros((8, 1)) - 1, idx])

    mse[0, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'logistic' + '_' + 'fisher' +
                           '_' + str(True) + '_mse.npy')
    mse[1, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'logistic' + '_' + 'fisher' +
                           '_' + str(False) + '_mse.npy')
    mse[2, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'logistic' + '_' + 'kl' + '_'
                           + str(True) + '_mse.npy')
    mse[3, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'logistic' + '_' + 'kl' + '_'
                           + str(False) + '_mse.npy')

    mse[4, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'beta3' + '_' + 'fisher' + '_'
                           + str(True) + '_mse.npy')
    mse[5, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'beta3' + '_' + 'fisher' + '_'
                           + str(False) + '_mse.npy')
    mse[6, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'beta3' + '_' + 'kl' + '_'
                           + str(True) + '_mse.npy')
    mse[7, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'beta3' + '_' + 'kl' + '_'
                           + str(False) + '_mse.npy')

    nll[0, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'logistic' + '_' + 'fisher' +
                           '_' + str(True) + '_nll.npy')
    nll[1, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'logistic' + '_' + 'fisher' +
                           '_' + str(False) + '_nll.npy')
    nll[2, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'logistic' + '_' + 'kl' + '_'
                           + str(True) + '_nll.npy')
    nll[3, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'logistic' + '_' + 'kl' + '_'
                           + str(False) + '_nll.npy')

    nll[4, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'beta3' + '_' + 'fisher' + '_'
                           + str(True) + '_nll.npy')
    nll[5, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'beta3' + '_' + 'fisher' + '_'
                           + str(False) + '_nll.npy')
    nll[6, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'beta3' + '_' + 'kl' + '_'
                           + str(True) + '_nll.npy')
    nll[7, :] = numpy.load('./cat/' + str(target_measure) + '_' + test_mdl_class + '_' + 'beta3' + '_' + 'kl' + '_'
                           + str(False) + '_nll.npy')

    for i in range(0, 8):
        matplotlib.pyplot.figure(figsize=(8, 6))
        tmp_mat = numpy.zeros((len(dataset_list), len(dataset_list)))
        tmp_mat[idx[i, 1:].astype('int'), numpy.arange(len(dataset_list))] = 1.0
        matplotlib.pyplot.imshow(tmp_mat, aspect=0.5)
        matplotlib.pyplot.xlabel('test idx')
        matplotlib.pyplot.ylabel('selected dataset')
        matplotlib.pyplot.title(m_list[target_measure] + '      ' + cat_list[i])
        matplotlib.pyplot.savefig('./figures/test_idx/' + str(target_measure) + '_' + test_mdl_class + '_' + cat_list[i]
                                  + '_idx.png')
        matplotlib.pyplot.close()

    tau = numpy.zeros([8, 8])
    for i in range(0, 8):
        for j in range(0, 8):
            tau[i, j] = scipy.stats.kendalltau(idx[i, :], idx[j, :])[0]

    matplotlib.pyplot.figure(figsize=(16, 9))
    tau[numpy.diag_indices(8)] = numpy.nan
    matplotlib.pyplot.imshow(tau, origin='lower')
    matplotlib.pyplot.xticks(numpy.arange(8), cat_list, rotation=90)
    matplotlib.pyplot.yticks(numpy.arange(8), cat_list)
    matplotlib.pyplot.title(m_list[target_measure])
    matplotlib.pyplot.savefig('./figures/test_idx/' + str(target_measure) + '_' + test_mdl_class + '_tau.png')
    matplotlib.pyplot.close()

    matplotlib.pyplot.figure(figsize=(16, 9))
    matplotlib.pyplot.plot(numpy.sqrt(mse.transpose()), linewidth=5, alpha=0.8)
    matplotlib.pyplot.ylim(
        [numpy.min(numpy.sqrt(mse.transpose())) - 0.005, numpy.max(numpy.sqrt(mse.transpose())[2:, :])])
    matplotlib.pyplot.legend(cat_list)
    matplotlib.pyplot.xticks(numpy.arange(0, len(dataset_list), 5));
    matplotlib.pyplot.grid()
    matplotlib.pyplot.title(m_list[target_measure])
    matplotlib.pyplot.savefig('./figures/mse/' + str(target_measure) + '_' + test_mdl_class + '_mse.png')
    matplotlib.pyplot.close()

    matplotlib.pyplot.figure(figsize=(16, 9))
    matplotlib.pyplot.plot(nll.transpose(), linewidth=5, alpha=0.8)
    matplotlib.pyplot.ylim([numpy.min(nll.transpose()) - 5, numpy.max(nll.transpose()[2:, :])])
    matplotlib.pyplot.legend(cat_list)
    matplotlib.pyplot.xticks(numpy.arange(0, len(dataset_list), 5));
    matplotlib.pyplot.grid()
    matplotlib.pyplot.title(m_list[target_measure])
    matplotlib.pyplot.savefig('./figures/nll/' + str(target_measure) + '_' + test_mdl_class + '_nll.png')
    matplotlib.pyplot.close()


# if __name__ == '__main__':
#     base_list = ['GBC']
#     for measure in range(2, 8):
#         get_beta3_figures(measure)
#         get_logistic_figures(measure)
#         for base in base_list:
#             get_cat_figures(measure, base)


