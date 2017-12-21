import numpy as np
import scipy.io as sio


def count_dist(y, K):
    dist = np.zeros(K, np.int)
    for yi in y:
        dist[yi] += 1
    return dist

def load_SVHN(path, grayscale=True, p_valid=0.2, unbalanced=False):
    train_data = sio.loadmat(path + '/SVHN/train_32x32.mat')
    # print a.keys() # ['y', 'X', '__version__', '__header__', '__globals__']
    X = train_data['X']
    y = train_data['y']
    # print type(X), type(y) # <type 'numpy.ndarray'> <type 'numpy.ndarray'>
    # print X.shape, y.shape # (32, 32, 3, 73257) (73257, 1)
    # print np.min(X), np.max(X) # 0 255
    # print np.min(y), np.max(y) # 1 10
    y.shape = (y.shape[0],)
    y -= 1  # make [1, 2, ..., 10] become [0, 1, ..., 9]

    test_data = sio.loadmat(path + '/SVHN/test_32x32.mat')
    X_test = test_data['X']
    y_test = test_data['y']   
    # print X_test.shape, y_test.shape # (32, 32, 3, 26032) (26032, 1)
    # print np.min(X_test), np.max(X_test) # 0 255
    # print np.min(y_test), np.max(y_test) # 1 10
    y_test.shape = (y_test.shape[0],)
    y_test -= 1

    if grayscale:
        X = np.mean(X, axis=2)
        # print X.shape # (32, 32, 73257)
        # print np.min(X[:, :, 0]), np.max(X[:, :, 0]) # 15.0 100.666666667
        X_test = np.mean(X_test, axis=2)

    X_new = np.zeros((X.shape[2], X.shape[0] * X.shape[1]))
    for i in xrange(X.shape[2]):
        X_new[i, :] = X[:, :, i].reshape(X.shape[0] * X.shape[1])
    # print X_new.shape # (73257, 1024)
    # print np.min(X_new[0]), np.max(X_new[0]) # 15.0 100.666666667
    X = X_new / 255

    X_test_new = np.zeros((X_test.shape[2], X_test.shape[0] * X_test.shape[1]))
    for i in xrange(X_test.shape[2]):
        X_test_new[i, :] = X_test[:, :, i].reshape(X_test.shape[0] * X_test.shape[1])
    # print X_test_new.shape # (26032, 1024)
    # print np.min(X_test_new[0]), np.max(X_test_new[0]) # 25.3333333333 89.0
    X_test = X_test_new / 255
    
    # print count_dist(y, 10) # [13861 10585  8497  7458  6882  5727  5595  5045  4659  4948]
    # print count_dist(y_test, 10) # [5099 4149 2882 2523 2384 1977 2019 1660 1595 1744]

    # shuffle the training set ...
    zipper = list(zip(X, y))
    np.random.shuffle(zipper)
    X, y = zip(*zipper)
    X, y = np.array(X), np.array(y)

    # shuffle the testing set ...
    zipper = list(zip(X_test, y_test))
    np.random.shuffle(zipper)
    X_test, y_test = zip(*zipper)
    X_test, y_test = np.array(X_test), np.array(y_test)

    if unbalanced:

        def make_unbalanced(X, y, class_to_drop, r_drop):
            index_list = []
            for i in xrange(10):
                index_list.append([])
            for i in xrange(X.shape[0]):
                index_list[y[i]].append(i)
            for k in class_to_drop:
                index_list[k] = index_list[k][: int((1 - r_drop) * len(index_list[k]))]
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

        K_drop = 4      # TODO: number of classes to be dropped
        r_drop = 0.9    # TODO: rate of samples to be dropped (class-wise)
        class_to_drop = np.random.choice(a=10, size=K_drop, replace=False)
        # print class_to_drop

        # TODO: Be cautious! svhn is unbalanced byself!
        # n_keep = int(X.shape[0] / 10 * (1 - r_drop))
        X, y = make_unbalanced(X, y, class_to_drop, r_drop)

        # n_keep = int(X_test.shape[0] / 10 * (1 - r_drop))
        X_test, y_test = make_unbalanced(X_test, y_test, class_to_drop, r_drop)

    # split some samples from training set to validation set
    n_valid = int(p_valid * X.shape[0])
    valid_ind = np.random.choice(a=X.shape[0], size=n_valid, replace=False)
    train_ind = np.delete(np.arange(X.shape[0]), valid_ind)
    X_valid = X[valid_ind]
    y_valid = y[valid_ind]
    X_train = X[train_ind]
    y_train = y[train_ind]

    # print count_dist(y_train, 10) # [11096  8484  6723  5932  5590  4586  4514  4067  3664  3950]
    # print count_dist(y_valid, 10) # [2765 2101 1774 1526 1292 1141 1081  978  995  998]
    # print count_dist(y_test, 10)  # [5099 4149 2882 2523 2384 1977 2019 1660 1595 1744]

    # print count_dist(y_train, 10)
    # print count_dist(y_valid, 10)
    # print count_dist(y_test, 10)

    return [[X_train, y_train], [X_valid, y_valid], [X_test, y_test]]

def main():
    datasets = load_SVHN(
        path='/project/andyyuan/Datasets',
        # path='/Users/Andy/Desktop',
        grayscale=True,
        p_valid=0.2,
        unbalanced=True
    )

    train_set, valid_set, test_set = datasets
    print train_set[0].shape
    print valid_set[0].shape
    print test_set[0].shape


if __name__ == '__main__':
    main()
