import numpy
import pandas


def get_random_split_measurement(model_instance, x, y, measure, sparse=False, cap_size=10000, test_size=0.5):
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

    return measure.get_measure(s_test, y_test_vector)


def get_exhaustive_testing(data_dict, get_data, model_dict, get_model, measure,
                           sparse=False, cap_size=10000, test_size=0.5):
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
