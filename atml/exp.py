"""
The :mod:'atml.exp' module holds a set of functions to perform machine learning experiments and gather the corresponding
performances metrics.
"""
# Author: Hao Song (nuacesh@gmail.com)
# License: BSD-3

import numpy
import pandas
import openml


def get_random_split_measurement(model_instance, x, y, measure, sparse=False, cap_size=10000, test_size=0.5):
    """
    Perform a random split validation experiment for a given combination of model, dataset, and evaluation measure.

    Parameters
    ----------
    model_instance: sklearn.predictor
        A model instance with the sklearn.predictor template, it should have a fit() method for model training, and
        a predict_proba() method to predict probability vectors on test data.

    x: numpy.ndarray
        The data matrix with shape (n samples, d dimensions)

    y: numpy.ndarray
        The label vector with shape (n samples, 1)

    measure: atml.Measure
        A evaluation measure selected from the atml.measure module.

    sparse: boolean
        To indicate whether to only use a subset of the dataset to perform the experiments.

    cap_size: integer
        In the case sparse=True, cap_size specifies the maximum size of the dataset to run the experiments.

    test_size: float
        The proportion of the dataset that is used as the testing set (validation set).

    Returns
    ----------

    measurement: float
        The performance measurement on the testing set (validation set).

    """
    # x : shape(n, m)
    # y : shape(n, 1)
    # y_vector: shape(n, k)

    n = numpy.shape(x)[0]
    _, y = numpy.unique(y, return_inverse=True)
    k = len(numpy.unique(y))

    shuffle_idx = numpy.random.permutation(numpy.arange(0, n))
    x = x[shuffle_idx, :]
    y = y[shuffle_idx]
    y_vector = numpy.zeros((n, k))
    for i in range(0, k):
        y_vector[:, i] = (y == i)

    if sparse & (n > cap_size):
        class_count = numpy.ceil(numpy.mean(y_vector, axis=0) * cap_size).astype('int')
        tmp_x = []
        tmp_y = []
        tmp_y_vector = []
        for i in range(0, k):
            idx_list = numpy.argwhere(y == i).ravel()
            tmp_idx = numpy.random.choice(idx_list, class_count[i], replace=False)
            tmp_x.append(x[tmp_idx, :])
            tmp_y.append(y[tmp_idx])
            tmp_y_vector.append(y_vector[tmp_idx, :])
        x = numpy.vstack(tmp_x)
        y = numpy.hstack(tmp_y)
        y_vector = numpy.vstack(tmp_y_vector)

    n_train = numpy.ceil(n * (1 - test_size)).astype('int')

    selected_idx = numpy.zeros(n)

    class_count = numpy.ceil(numpy.mean(y_vector, axis=0) * n_train).astype('int')

    x_train = []

    y_train = []

    for j in range(0, k):
        idx_list = numpy.argwhere(y == j).ravel()
        tmp_idx = numpy.random.choice(idx_list, class_count[j], replace=False)
        x_train.append(x[tmp_idx, :])
        y_train.append(y[tmp_idx])
        selected_idx[tmp_idx] = 1.0

    x_train = numpy.vstack(x_train)

    y_train = numpy.hstack(y_train)

    x_test = x[selected_idx == 0, :]

    y_test = y[selected_idx == 0]

    y_train_vector = numpy.zeros((len(y_train), k))

    y_test_vector = numpy.zeros((len(y_test), k))

    for k in range(0, k):
        y_train_vector[:, k] = (y_train == k)
        y_test_vector[:, k] = (y_test == k)

    mdl = model_instance
    mdl.fit(x_train, y_train)

    s_test = mdl.predict_proba(x_test)

    s_test[~numpy.isfinite(numpy.sum(s_test, axis=1)), :] = \
        numpy.repeat(numpy.mean(y_train_vector, axis=0).reshape(1, -1),
                     numpy.sum(~numpy.isfinite(numpy.sum(s_test, axis=1))), axis=0)

    measurement = measure.get_measure(s_test, y_test_vector)

    return measurement


def get_openml_testing(openml_dict, flow_dict, measure, max_n_exp=10):
    """
    Gather machine learning experiment results from OpenML

    Parameters
    ----------

    openml_dict: dict
        A dictionary that defines the (1) user defined dataset index, (2) name of the dataset, (3) OpenML defined
        dataset ID, (3) OpenML defined task ID. Example: {0, ('adult', 1590, 7592)}

    flow_dict: dict
        A dictionary that define the (1) OpenMl defined flow (model) ID, (2) user defined flow (model) index.
        Example: {1172: 0}

    measure: str
        The selected evaluation measure as defined by OpenML. Example: 'predictive_accuracy'
        See: https://www.openml.org/search?type=measure

    max_n_exp: int
        The maximum number of results collected for each dataset and task combination.

    Returns
    ----------

    res: pandas.DataFrame
        A table that contains the collected experiment results.

    """
    data = []
    for ii in range(0, len(openml_dict)):
        for jj in range(0, len(flow_dict)):
            did = openml_dict[ii][1]
            tid = openml_dict[ii][2]
            run_list = list(openml.evaluations.list_evaluations(function=measure,
                                                                tasks=[tid],
                                                                flows=[list(flow_dict.keys())[jj]]).items())
            print([tid, did, list(flow_dict.keys())[jj], measure, len(run_list)])
            for j in range(0, min(max_n_exp, len(run_list))):
                try:
                    tmp_run = openml.runs.get_run(run_id=run_list[j][0])
                    data.append(
                        [ii, openml_dict[ii][0], flow_dict[tmp_run.flow_id], tmp_run.flow_name, run_list[j][1].value])
                except ValueError:
                    pass
    res = pandas.DataFrame(data, columns=['data_idx', 'data_ref', 'model_idx', 'model_ref', measure])
    return res


def get_exhaustive_testing(data_dict, get_data, model_dict, get_model, measure,
                           sparse=False, cap_size=10000, test_size=0.5):
    """
    Perform testing experiments on all the possible combinations between different models and datasets.

    Parameters
    ----------

    data_dict: dict
        A dictionary that defines the index and the reference name of all the datasets.
        Example: data_dict = {0: 'iris', 1: 'digits', 2: 'wine'}

    get_data: Callable
        A function that takes the dataset index and returns the features (x), and target (y) for the specified dataset.

    model_dict: dict
        A dictionary that defines the index and the reference name of all the models.
        Example: model_dict = {0: 'logistic regression', 1: 'random forest', 2: 'naive bayes'}

    get_model: Callable
        A function that takes the model index and returns the instance from a model class with a sklearn template.
        The model should have a fit(x, y) method for training and predict_proba(x) for testing.

    measure: atml.Measure
        A evaluation measure selected from the atml.measure module.

    sparse: boolean
        To indicate whether to only use a subset of the dataset to perform the experiments.

    cap_size: integer
        In the case sparse=True, cap_size specifies the maximum size of the dataset to run the experiments.

    test_size: float
        The proportion of the dataset that is used as the testing set (validation set).

    Returns
    ----------

    res: pandas.DataFrame
        A table that contains the

    """
    n_data = len(data_dict)
    n_model = len(model_dict)

    res = pandas.DataFrame(columns=['data_idx', 'data_ref', 'model_idx', 'model_ref',
                                    measure.name])

    idx = 0

    for i in range(0, n_data):
        for j in range(0, n_model):
            mdl = get_model(model_dict[j])
            x, y = get_data(data_dict[i])
            tmp_m = get_random_split_measurement(mdl, x, y, measure,
                                                 sparse=sparse, cap_size=cap_size, test_size=test_size)
            res.loc[idx] = [i, data_dict[i], j, model_dict[j], tmp_m]

            idx = idx + 1

    return res


def get_single_testing(data_idx, mdl, data_dict, get_data, measure, sparse=False, cap_size=10000, test_size=0.5):
    """
    Perform a single testing experiment on a specified dataset with the given model.

    Parameters
    ----------

    data_idx: int
        The index of the selected dataset, as defined with data_dict.

    mdl: sklearn.predictor
        An instance of the sklearn predictor.
        The model should have a fit(x, y) method for training and predict_proba(x) for testing.

    data_dict: dict
        A dictionary that defines the index and the reference name of all the datasets.
        Example: data_dict = {0: 'iris', 1: 'digits', 2: 'wine'}

    get_data: Callable
        A function that takes the dataset index and returns the features (x), and target (y) for the specified dataset.

    measure: atml.Measure
        A evaluation measure selected from the atml.measure module.

    sparse: boolean
        To indicate whether to only use a subset of the dataset to perform the experiments.

    cap_size: int
        In the case sparse=True, cap_size specifies the maximum size of the dataset to run the experiments.

    test_size: float
        The proportion of the dataset that is used as the testing set (validation set).

    Returns
    ----------

        tmp_m: float
            The performance measurement on the testing set (validation set).

    """

    x, y = get_data(data_dict[data_idx])

    tmp_m = get_random_split_measurement(mdl, x, y, measure,
                                         sparse=sparse, cap_size=cap_size, test_size=test_size)

    return tmp_m
