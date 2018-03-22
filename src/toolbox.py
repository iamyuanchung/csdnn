""" The toolbox module provides serveral useful functions. """

import cPickle
import gzip

import numpy as np

import theano
# from sklearn.preprocessing import LabelBinarizer


class CostMatrixGenerator(object):

    def __init__(self, y, n_class):
        """ Create a CostMatrixGenerator object

        :type y: 1-dim numpy.ndarray
        :param y:labels for training examples

        :type n_class: int
        :param n_class: number of classes

        """
        # lb = LabelBinarizer()
        # lb.fit(y)

        # to make sure the classes are: 0, 1, 2, ..., n_class - 1
        # assert len(lb.classes_) == n_class
        # assert False not in (np.arange(n_class) == lb.classes_)

        self.n_class = n_class

        # compute the class distribution
        self.class_dist = np.zeros(self.n_class)
        for i in xrange(len(y)):
            self.class_dist[y[i]] += 1

    def general(self, scale=10.):
        """ Create and return the cost matrix for general case (from Hsuan-Tien, Lin)

        :type scale: float
        :param scale: will be used to scale up or down the cost

        """
        cost_mat = np.zeros((self.n_class, self.n_class))
        for i in xrange(self.n_class):
            for j in xrange(self.n_class):
                cost_mat[i][j] = 0. if i == j else np.random.random() * self.class_dist[j] / self.class_dist[i]
        return scale * cost_mat

    def naive(self, scale=1.):
        """ Create and return the cost matrix for naive case (cost-insensitive)

        :type scale: float
        :param scale: will be used to scale up or down the cost

        """
        cost_mat = np.ones((self.n_class, self.n_class))
        for i in xrange(self.n_class):
            cost_mat[i][i] = 0.
        return scale * cost_mat

    def absolute(self, scale=1.):
        """ Create and return the absolute cost matrix

        :type scale: float
        :param scale: will be used to scale up or down the cost

        """
        cost_mat = np.zeros((self.n_class, self.n_class))
        for i in xrange(self.n_class):
            for j in xrange(self.n_class):
                cost_mat[i][j] = np.abs(i - j)
        return scale * cost_mat

    def square(self, scale=1.):
        """ Create and return the square cost matrix

        :type scale: float
        :param scale: will be used to scale up or down the cost

        """
        return self.absolute(scale) ** 2


class MNISTLoader(object):

    def __init__(self, datasets_path):
        """ Initialize a MNISTLoader object """
        self.datasets_path = datasets_path + '/mnist_variations'

    def load(self, file_name):
        """ Parsing & loading """
        with open(file_name, 'rb') as f:
            data = f.read().split('\n')
        data = data[:-1]
        x = np.zeros((len(data), 784))
        y = np.zeros(len(data))
        for i in xrange(len(data)):
            xi = data[i].split(' ')
            for j in xrange(784):
                x[i][j] = float(xi[j])
            y[i] = float(xi[784])
        return [x, y]

    def mnist(self):
        """ MNIST digits """
        dataset = self.datasets_path + '/mnist.pkl.gz'
        with gzip.open(dataset) as f:
            train_set, valid_set, test_set = cPickle.load(f)
        train_set_x, train_set_y = train_set
        valid_set_x, valid_set_y = valid_set
        train_set_x = np.append(train_set_x, valid_set_x, axis=0)
        train_set_y = np.append(train_set_y, valid_set_y)
        train_set = [train_set_x, train_set_y]
        return train_set, test_set

    def basic(self):
        """ Subset of MNIST digits """
        path = self.datasets_path + '/mnist_basic'
        train_data = path + '/mnist_train.amat'
        test_data = path + '/mnist_test.amat'
        return [self.load(train_data), self.load(test_data)]

    def rot(self):
        """ MNIST digits with added random rotation """
        path = self.datasets_path + '/mnist_rotation'
        train_data = path + '/mnist_all_rotation_normalized_float_train_valid.amat'
        test_data = path + '/mnist_all_rotation_normalized_float_test.amat'
        return [self.load(train_data), self.load(test_data)]

    def bg_rand(self):
        """ MNIST digits with random noise background """
        path = self.datasets_path + '/mnist_background_random'
        train_data = path + '/mnist_background_random_train.amat'
        test_data = path + '/mnist_background_random_test.amat'
        return [self.load(train_data), self.load(test_data)]

    def bg_img(self):
        """ MNIST digits with random image background """
        path = self.datasets_path + '/mnist_background_images'
        train_data = path + '/mnist_background_images_train.amat'
        test_data = path + '/mnist_background_images_test.amat'
        return [self.load(train_data), self.load(test_data)]

    def bg_img_rot(self):
        """ MNIST digits with rotation and image background """
        path = self.datasets_path + '/mnist_rotation_back_image'
        train_data = path + '/mnist_all_background_images_rotation_normalized_train_valid.amat'
        test_data = path + '/mnist_all_background_images_rotation_normalized_test.amat'
        return [self.load(train_data), self.load(test_data)]

    def rect(self):
        path = self.datasets_path + '/./../rect'
        train_data = path + '/rectangles_train.amat'
        test_data = path + '/rectangles_test.amat'
        return [self.load(train_data), self.load(test_data)]

    def rect_img(self):
        path = self.datasets_path + '/./../rect_img'
        train_data = path + '/rectangles_im_train.amat'
        test_data = path + '/rectangles_im_test.amat'
        return [self.load(train_data), self.load(test_data)]

    def convex(self):
        path = self.datasets_path + '/./../convex'
        train_data = path + '/convex_train.amat'
        test_data = path + '/convex_test.amat'
        return [self.load(train_data), self.load(test_data)]


def class_to_example(y, cost_mat):
    """ Transform the cost from class-class to example-class format """
    yc = np.zeros((len(y), cost_mat.shape[0]))
    for i in xrange(len(y)):
        yc[i][:] = cost_mat[y[i]][:]
    return yc

def make_shared_data(data):
    """ Make data become shared variable """
    shared_data = theano.shared(
        value=np.asarray(
            a=data,
            dtype=theano.config.floatX
        ),
        borrow=True
    )
    return shared_data

def naive_prediction_cost(model_name, test_data, cost_mat):
    """ Compute the cost made by naive prediction """
    with open(model_name, 'r') as f:
        classifier = cPickle.load(f)
    test_x, test_y = test_data
    test_x = make_shared_data(test_x)
    predict_model = theano.function(
        inputs=[],
        # outputs=classifier.y_pred,    # TODO: LogisticRegression
        # outputs=classifier.logRegressionLayer.y_pred, # TODO: MLP
        outputs=classifier.logLayer.y_pred,   # TODO: SdA
        givens={
            # classifier.input: test_x    # TODO: LogisticRegression & MLP
            classifier.x: test_x    # TODO: SdA
        }
    )
    y_pred = predict_model()
    cost = 0.
    for i in xrange(test_x.get_value().shape[0]):
        cost += cost_mat[test_y[i]][y_pred[i]]
    # return cost
    return cost / test_x.get_value().shape[0]

def bayes_optimal_cost(model_name, test_data, cost_mat):
    """ Compute the cost made by Bayes-optimal decision """
    with open(model_name, 'r') as f:
        classifier = cPickle.load(f)
    test_x, test_y = test_data
    test_x = make_shared_data(test_x)
    predict_prob_model = theano.function(
        inputs=[],
        # outputs=classifier.p_y_given_x,   # TODO: LogisticRegression
        # outputs=classifier.logRegressionLayer.p_y_given_x,  # TODO: MLP
        outputs=classifier.logLayer.p_y_given_x,    # TODO: SdA
        givens={
            # classifier.input: test_x    # TODO: LogisticRegression & MLP
            classifier.x: test_x    # TODO: SdA
        }
    )
    y_prob = predict_prob_model()
    y_pred = np.argmin(np.dot(y_prob, cost_mat), axis=1)
    cost = 0.
    for i in xrange(test_x.get_value().shape[0]):
        cost += cost_mat[test_y[i]][y_pred[i]]
    # return cost
    return cost / test_x.get_value().shape[0]


def main():

    def count_dist(y, K):
        dist = np.zeros(K)
        for yi in y:
            dist[yi] += 1
        return dist

    loader = MNISTLoader('/home/chungyua/research/datasets')

    print 'loading mnist ...'
    train_set, test_set = loader.mnist()
    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set
    print 'train dist' + str(count_dist(train_set_y, 10))
    print 'train dist' + str(count_dist(test_set_y, 10))
    # print train_set_x.shape
    # print train_set_y.shape

    print 'loading basic ...'
    train_set, test_set =  loader.basic()
    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set
    print 'train dist' + str(count_dist(train_set_y, 10))
    print 'train dist' + str(count_dist(test_set_y, 10))
    # print train_set_x.shape
    # print train_set_y.shape

    print 'loading rot ...'
    train_set, test_set = loader.rot()
    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set
    print 'train dist' + str(count_dist(train_set_y, 10))
    print 'train dist' + str(count_dist(test_set_y, 10))
    # print train_set_x.shape
    # print train_set_y.shape

    print 'loading bg_rand ...'
    train_set, test_set = loader.bg_rand()
    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set
    print 'train dist' + str(count_dist(train_set_y, 10))
    print 'train dist' + str(count_dist(test_set_y, 10))
    # print train_set_x.shape
    # print train_set_y.shape

    print 'loading bg_img ...'
    train_set, test_set = loader.bg_img()
    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set
    print 'train dist' + str(count_dist(train_set_y, 10))
    print 'train dist' + str(count_dist(test_set_y, 10))
    # print train_set_x.shape
    # print train_set_y.shape

    print 'loading bg_img_rot ...'
    train_set, test_set = loader.bg_img_rot()
    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set
    print 'train dist' + str(count_dist(train_set_y, 10))
    print 'train dist' + str(count_dist(test_set_y, 10))
    # print train_set_x.shape
    # print train_set_y.shape

    print 'loading rect ...'
    train_set, test_set = loader.rect()
    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set
    print 'train dist' + str(count_dist(train_set_y, 2))
    print 'train dist' + str(count_dist(test_set_y, 2))
    # print train_set_x.shape
    # print train_set_y.shape

    print 'loading rect_img ...'
    train_set, test_set = loader.rect_img()
    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set
    print 'train dist' + str(count_dist(train_set_y, 2))
    print 'train dist' + str(count_dist(test_set_y, 2))
    # print train_set_x.shape
    # print train_set_y.shape

    print 'loading convex ...'
    train_set, test_set = loader.convex()
    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set
    print 'train dist' + str(count_dist(train_set_y, 2))
    print 'train dist' + str(count_dist(test_set_y, 2))
    # print train_set_x.shape
    # print train_set_y.shape


if __name__ == '__main__':
    main()
