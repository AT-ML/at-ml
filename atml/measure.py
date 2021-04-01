import numpy
import scipy.integrate

tiny = numpy.finfo('float64').tiny


class Measure:
    def __init__(self, task):
        self.task = task
        
        
class AUC(Measure):
    def __init__(self, target_positive=0):
        super().__init__(task='classification')
        self.target_positive = target_positive
        self.name = 'area under the curve (class ' + str(self.target_positive+1) + 'vs rest)'
        
    def get_measure(self, s, y):
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


class BAcc(Measure):
    def __init__(self):
        super().__init__(task='classification', target_positive=0)
        self.name = 'accuracy (class ' + str(self.target_positive+1) + 'vs rest)'

    def get_measure(self, s, y):
        n = numpy.shape(s)[0]
        s_bin = numpy.zeros((n, 2))
        y_bin = numpy.zeros((n, 2))
        s_bin[:, 0] = s[:, self.target_positive]
        s_bin[:, 1] = 1.0 - s_bin[:, 0]
        y_bin[:, 0] = y[:, self.target_positive]
        y_bin[:, 1] = 1.0 - y_bin[:, 0]
        return numpy.mean(numpy.argmax(s_bin, axis=1) == numpy.argmax(y_bin, axis=1))


class F1(Measure):
    def __init__(self):
        super().__init__(task='classification', target_positive=0)
        self.name = 'F score (class ' + str(self.target_positive+1) + 'vs rest)'

    def get_measure(self, s, y):
        y_hat = numpy.argmax(s, axis=1)
        y_label = numpy.argmax(y, axis=1)
        TP = numpy.sum((y_hat == self.target_positive) * (y_label == self.target_positive))
        FP = numpy.sum((y_hat == self.target_positive) * (y_label != self.target_positive))
        FN = numpy.sum((y_hat != self.target_positive) * (y_label == self.target_positive))
        if (TP + FP + FN) == 0:
            TP = 1
            FP = 1
            FN = 1
        return 2 * TP / (2 * TP + FP + FN)


class Acc(Measure):
    def __init__(self):
        super().__init__(task='classification')
        self.name = 'accuracy'

    @staticmethod
    def get_measure(s, y):
        return numpy.mean(numpy.argmax(s, axis=1) == numpy.argmax(y, axis=1))


class BS(Measure):
    def __init__(self):
        super().__init__(task='classification')
        self.name = 'Brier score'

    @staticmethod
    def get_measure(s, y):
        return numpy.mean(numpy.sum((s - y) ** 2, axis=1))

    @staticmethod
    def transform(m):
        return 1 - (m/2)


class LL(Measure):
    def __init__(self):
        super().__init__(task='classification')
        self.name = 'Log loss (cross entropy)'

    @staticmethod
    def get_measure(s, y):
        s[s <= tiny] = tiny
        return numpy.mean(numpy.sum(-numpy.log(s) * y, axis=1))

