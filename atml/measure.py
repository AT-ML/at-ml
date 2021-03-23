import numpy
import scipy.integrate


def measures_all(s, y):
    n_class = numpy.shape(s)[1]
    acc = empirical_accuracy(s, y)
    bs = empirical_brier_score(s, y)
    ll = empirical_log_loss(s, y)
    bin_acc = numpy.zeros(n_class)
    auc = numpy.zeros(n_class)
    f1 = numpy.zeros(n_class)
    for j in range(0, n_class):
        bin_acc[j] = empirical_binary_accuracy(s, y, j)
        auc[j] = empirical_auc(s, y, j)
        f1 = empirical_f_score(s, y, j)
    return acc, bs, ll, bin_acc, auc, f1 


def empirical_auc(s, y, target_positive=0):
    bin_edges = numpy.unique(1-s[:, target_positive])
    count_pos = numpy.histogram(1-s[y[:, target_positive] == 1, target_positive], bins=bin_edges, range=(0.0, 1.0))[0]
    count_neg = numpy.histogram(1-s[y[:, target_positive] != 1, target_positive], bins=bin_edges, range=(0.0, 1.0))[0]
    if numpy.sum(count_pos) == 0:
        count_pos[:] = 1.0
    if numpy.sum(count_neg) == 0:
        count_neg[:] = 1.0
    cdf_pos = numpy.hstack([0.0, 
                            numpy.cumsum(count_pos) / numpy.sum(count_pos), 
                            1.0])
    cdf_neg = numpy.hstack([0.0, 
                            numpy.cumsum(count_neg) / numpy.sum(count_neg), 
                            1.0])
    auc = scipy.integrate.trapz(cdf_pos, cdf_neg)
    return auc


def empirical_brier_score(s, y):
    return numpy.mean(numpy.sum((s - y) ** 2, axis=1))


def empirical_log_loss(s, y):
    eps = numpy.finfo('float64').tiny
    s[s <= eps] = eps
    return numpy.mean(numpy.sum(-numpy.log(s) * y, axis=1))


def empirical_accuracy(s, y):
    return numpy.mean(numpy.argmax(s, axis=1) == numpy.argmax(y, axis=1))


def empirical_binary_accuracy(s, y, target_positive=0):
    n = numpy.shape(s)[0]
    s_bin = numpy.zeros((n, 2))
    y_bin = numpy.zeros((n, 2))
    s_bin[:, 0] = s[:, target_positive]
    s_bin[:, 1] = 1.0 - s_bin[:, 0]
    y_bin[:, 0] = y[:, target_positive]
    y_bin[:, 1] = 1.0 - y_bin[:, 0]
    return numpy.mean(numpy.argmax(s_bin, axis=1) == numpy.argmax(y_bin, axis=1))


def empirical_f_score(s, y, target_class=0):
    y_hat = numpy.argmax(s, axis=1)
    y_label = numpy.argmax(y, axis=1)
    TP = numpy.sum((y_hat == target_class) * (y_label == target_class))
    FP = numpy.sum((y_hat == target_class) * (y_label != target_class))
    FN = numpy.sum((y_hat != target_class) * (y_label == target_class))
    if (TP + FP + FN) == 0:
        TP = 1
        FP = 1
        FN = 1
    return 2*TP / (2*TP + FP + FN)