import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from osR import OneSidedCostRegressor


class OneSidedMLP(object):

    def __init__(self, rng, input, n_in, hidden_layer_sizes, n_out):

        self.sigmoid_layers = []
        self.params = []
        self.n_layers = len(hidden_layer_sizes)

        assert self.n_layers > 0

        for i in xrange(self.n_layers):

            if i == 0:
                input_size = n_in
            else:
                input_size = hidden_layer_sizes[i - 1]

            if i == 0:
                layer_input = input
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layer_sizes[i],
                                        activation=T.tanh)

            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

        self.logRegressionLayer = OneSidedCostRegressor(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layer_sizes[-1],
            n_out=n_out
        )

        self.params.extend(self.logRegressionLayer.params)

        self.input = input

    def sgd_optimize(self, train_set, test_set, n_epochs, learning_rate, batch_size):

        train_set_x, train_set_y, train_set_c = train_set
        test_set_x, test_set_y, test_set_c = test_set

        train_set_z = np.zeros(train_set_c.shape) - 1
        for i in xrange(train_set_z.shape[0]):
            train_set_z[i][train_set_y[i]] = 1

        from toolbox import make_shared_data

        train_set_x = make_shared_data(train_set_x)
        train_set_c = make_shared_data(train_set_c)
        train_set_z = make_shared_data(train_set_z)
        train_set_y = T.cast(make_shared_data(train_set_y), 'int32')

        test_set_x = make_shared_data(test_set_x)
        test_set_c = make_shared_data(test_set_c)
        test_set_y = T.cast(make_shared_data(test_set_y), 'int32')

        print '... building the model'

        index = T.lscalar()

        cost = self.logRegressionLayer.one_sided_regression_loss

        gparams = [T.grad(cost, param) for param in self.params]

        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=[
                (param, param - learning_rate * gparam)
                for param, gparam in zip(self.params, gparams)
            ],
            givens={
                self.input: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.logRegressionLayer.cost_vector: train_set_c[index * batch_size: (index + 1) * batch_size],
                self.logRegressionLayer.Z_nk: train_set_z[index * batch_size: (index + 1) * batch_size]
            },
            name='train_model'
        )

        in_sample_result = theano.function(
            inputs=[],
            outputs=[
                self.logRegressionLayer.error,
                self.logRegressionLayer.future_cost
            ],
            givens={
                self.input: train_set_x,
                self.logRegressionLayer.y: train_set_y,
                self.logRegressionLayer.cost_vector: train_set_c
            },
            name='in_sample_result'
        )

        out_sample_result = theano.function(
            inputs=[],
            outputs=[
                self.logRegressionLayer.error,
                self.logRegressionLayer.future_cost
            ],
            givens={
                self.input: test_set_x,
                self.logRegressionLayer.y: test_set_y,
                self.logRegressionLayer.cost_vector: test_set_c
            },
            name='out_sample_result'
        )

        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

        print '... training the model'

        best_Cout = np.inf
        corresponding_Eout = np.inf
        for epoch in xrange(n_epochs):
            print 'epoch #%d' % (epoch + 1)
            for batch_index in xrange(n_train_batches):
                batch_cost = train_model(batch_index)
            Ein, Cin = in_sample_result()
            Eout, Cout = out_sample_result()
            if Cout < best_Cout:
                best_Cout = Cout
                corresponding_Eout = Eout
                print '    better performance achieved ... best_Cout = %f' % best_Cout

        print 'after training %d epochs, best_Cout = %f, and corresponding_Eout = %f'   \
               % (n_epochs, best_Cout, corresponding_Eout)


def main():
    print '... loading dataset'

    from toolbox import CostMatrixGenerator, MNISTLoader, class_to_example

    loader = MNISTLoader('/home/chungyua/research/datasets')

    train_set, test_set = loader.mnist()
    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set

    cmg = CostMatrixGenerator(train_set_y, 10)
    cost_mat = cmg.absolute()

    train_set_c = class_to_example(train_set_y, cost_mat)
    test_set_c = class_to_example(test_set_y, cost_mat)

    rng = np.random.RandomState(np.random.randint(100000))

    reg = OneSidedMLP(
        rng=rng,
        input=T.matrix('input'),
        n_in=28 * 28,
        hidden_layer_sizes=[1000, 1000, 1000],
        n_out=10
    )

    reg.sgd_optimize(
        train_set=[train_set_x, train_set_y, train_set_c],
        test_set=[test_set_x, test_set_y, test_set_c],
        n_epochs=100,
        learning_rate=0.01,
        batch_size=100
    )


if __name__ == '__main__':
    main()
