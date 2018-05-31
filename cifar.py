import cPickle

import numpy as np


def count_dist(y, K):
    dist = np.zeros(K, np.int)
    for yi in y:
        dist[yi] += 1
    return dist

def unpickle(filename):
    with open(filename, 'rb') as f:
        data = cPickle.load(f)
    return data

def load_cifar(path, grayscale=True, p_valid=0.2, unbalanced=False):
    # loading and collecting training set ...
    for i in xrange(5):
        batch_data = unpickle(path + '/cifar-10-batches-py/data_batch_' + str(i + 1))
        batch_X = batch_data['data']
        batch_y = batch_data['labels']
        if i == 0:
            X_train = batch_X
            y_train = batch_y
        else:
            X_train = np.append(X_train, batch_X, axis=0)
            y_train = np.append(y_train, batch_y)

    X_train = np.array(X_train) / 255.
    y_train = np.array(y_train)

    # loading testing set ...
    test_data = unpickle(path + '/cifar-10-batches-py/test_batch')
    X_test = np.array(test_data['data']) / 255.
    y_test = np.array(test_data['labels'])

    # shuffle the training set ...
    zipper = list(zip(X_train, y_train))
    np.random.shuffle(zipper)
    X_train, y_train = zip(*zipper)
    X_train, y_train = np.array(X_train), np.array(y_train)

    if grayscale:
        green_slice = X_train.shape[1] / 3
        X_train = (X_train[:, :green_slice] + X_train[:, green_slice: green_slice * 2] + X_train[:, green_slice * 2:]) / 3
        X_test = (X_test[:, :green_slice] + X_test[:, green_slice: green_slice * 2] + X_test[:, green_slice * 2:]) / 3

    if unbalanced:

        def make_unbalanced(X, y, class_to_drop, n_keep):
            index_list = []
            for i in xrange(10):
                index_list.append([])
            for i in xrange(X.shape[0]):
                index_list[y[i]].append(i)
            for k in class_to_drop:
                index_list[k] = index_list[k][: n_keep]
            for i in xrange(10):
                if i == 0:
                    X_new = X[index_list[0]]
                    y_new = y[index_list[0]]
                else:
                    X_new = np.append(X_new, X[index_list[i]], axis=0)
                    y_new = np.append(y_new, y[index_list[i]])
            X, y = [X_new, y_new]
            zipper = list(zip(X, y))
            np.random.shuffle(zipper)
            X, y = zip(*zipper)
            X, y = np.array(X), np.array(y)
            return [X, y]

        K_drop = 4
        r_drop = 0.7
        class_to_drop = np.random.choice(a=10, size=K_drop, replace=False)

        n_keep = int(X_train.shape[0] / 10 * (1 - r_drop))
        X_train, y_train = make_unbalanced(X_train, y_train, class_to_drop, n_keep)

        n_keep = int(X_test.shape[0] / 10 * (1 - r_drop))
        X_test, y_test = make_unbalanced(X_test, y_test, class_to_drop, n_keep)

    # split some samples from training set to validation set
    n_valid = int(p_valid * X_train.shape[0])
    valid_ind = np.random.choice(a=X_train.shape[0], size=n_valid, replace=False)
    train_ind = np.delete(np.arange(X_train.shape[0]), valid_ind)
    X_valid = X_train[valid_ind]
    y_valid = y_train[valid_ind]
    X_train = X_train[train_ind]
    y_train = y_train[train_ind]

    return [[X_train, y_train], [X_valid, y_valid], [X_test, y_test]]

def main():
    data = load_cifar(
        path='/home/to/your/data',
        grayscale=True,
        p_valid=0.2,
        unbalanced=True
    )


if __name__ == '__main__':
    main()
