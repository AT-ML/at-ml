"""
The :mod:'atml.measure' module contains a set of common evaluation measures for predictive machine learning tasks.
"""
# Author: Hao Song (nuacesh@gmail.com)
# License: BSD-3

import numpy
import scipy.integrate

tiny = numpy.finfo('float64').tiny


class Measure:
    """
    The base measure class, and specifies the corresponding task type for the measure. (e.g. classification)
    """

    def __init__(self, task):
        """

        Parameters
        ----------
        task: string

        """
        self.task = task
        
        
class AUC(Measure):
    """
    Area under the ROC curve

    """

    def __init__(self, target_positive=0):
        """

        Parameters
        ----------
        target_positive: int

        """

        super().__init__(task='classification')
        self.target_positive = target_positive
        self.name = 'area under the curve (class ' + str(self.target_positive+1) + 'vs rest)'
        
    def get_measure(self, s, y):
        """

        Parameters
        ----------
        s: numpy.ndarray

        y: numpy.ndarray

        Returns
        ----------
        auc: float

        """

        bin_edges = numpy.unique(1-s[:, self.target_positive])
        count_pos = numpy.histogram(1-s[y[:, self.target_positive] == 1,
                                        self.target_positive], bins=bin_edges, range=(0.0, 1.0))[0]
        count_neg = numpy.histogram(1-s[y[:, self.target_positive] != 1, self.target_positive],
                                    bins=bin_edges, range=(0.0, 1.0))[0]
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

    @staticmethod
    def transform(m):
        """

        Parameters
        ----------
        m: float

        Returns
        ----------
        m_hat: float

        """
        m_hat = m
        return m_hat


class BAcc(Measure):
    """
    Binary accuracy

    """

    def __init__(self, target_positive=0):
        """

        Parameters
        ----------
        target_positive: int

        """

        super().__init__(task='classification', target_positive=target_positive)
        self.name = 'accuracy (class ' + str(self.target_positive+1) + 'vs rest)'

    def get_measure(self, s, y):
        """

        Parameters
        ----------
        s: numpy.ndarray

        y: numpy.ndarray

        Returns
        ----------
        bacc: float

        """

        n = numpy.shape(s)[0]
        s_bin = numpy.zeros((n, 2))
        y_bin = numpy.zeros((n, 2))
        s_bin[:, 0] = s[:, self.target_positive]
        s_bin[:, 1] = 1.0 - s_bin[:, 0]
        y_bin[:, 0] = y[:, self.target_positive]
        y_bin[:, 1] = 1.0 - y_bin[:, 0]
        bacc = numpy.mean(numpy.argmax(s_bin, axis=1) == numpy.argmax(y_bin, axis=1))
        return bacc


class F1(Measure):
    """

    F1 score

    """

    def __init__(self, target_positive=0):
        """

        Parameters
        ----------
        target_positive: int

        """

        super().__init__(task='classification', target_positive=target_positive)
        self.name = 'F1 score (class ' + str(self.target_positive+1) + 'vs rest)'

    def get_measure(self, s, y):
        """

        Parameters
        ----------
        s: numpy.ndarray

        y: numpy.ndarray

        Returns
        ----------
        f1: float

        """

        y_hat = numpy.argmax(s, axis=1)
        y_label = numpy.argmax(y, axis=1)
        TP = numpy.sum((y_hat == self.target_positive) * (y_label == self.target_positive))
        FP = numpy.sum((y_hat == self.target_positive) * (y_label != self.target_positive))
        FN = numpy.sum((y_hat != self.target_positive) * (y_label == self.target_positive))
        if (TP + FP + FN) == 0:
            TP = 1
            FP = 1
            FN = 1
        f1 = 2 * TP / (2 * TP + FP + FN)
        return f1


class Acc(Measure):
    """

    multi-class accuracy

    """

    def __init__(self):
        """

        """

        super().__init__(task='classification')
        self.name = 'accuracy'

    @staticmethod
    def get_measure(s, y):
        """

        Parameters
        ----------
        s: numpy.ndarray

        y: numpy.ndarray

        Returns
        ----------
        acc: float

        """

        acc = numpy.mean(numpy.argmax(s, axis=1) == numpy.argmax(y, axis=1))

        return acc


class BS(Measure):
    """

    Brier score

    """

    def __init__(self):
        """

        """

        super().__init__(task='classification')
        self.name = 'Brier score'

    @staticmethod
    def get_measure(s, y):
        """

        Parameters
        ----------
        s: numpy.ndarray

        y: numpy.ndarray

        Returns
        ----------
        bs: float

        """
        bs = numpy.mean(numpy.sum((s - y) ** 2, axis=1))
        return bs

    @staticmethod
    def transform(m):
        """

        Parameters
        ----------
        m: float

        Returns
        ----------
        m_hat: float

        """
        m_hat = 1 - (m/2)
        return m_hat


class LL(Measure):
    """

    Logarithm loss (cross entropy)

    """

    def __init__(self):
        """

        """

        super().__init__(task='classification')
        self.name = 'Log loss (cross entropy)'

    @staticmethod
    def get_measure(s, y):
        """

        Parameters
        ----------
        s: numpy.ndarray

        y: numpy.ndarray

        Returns
        ----------
        ll: float

        """
        s[s <= tiny] = tiny
        ll = numpy.mean(numpy.sum(-numpy.log(s) * y, axis=1))
        return ll

