import cPickle  # for dumping models

import numpy as np

import theano
import theano.tensor as T


class LinearCostRegressor(object):

    def __init__(self, input, n_in, n_out):

        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.cost_predicted_given_x = T.dot(input, self.W) + self.b

        self.y_pred = T.argmin(self.cost_predicted_given_x, axis=1)

        self.params = [self.W, self.b]

        self.input = input
        self.y = T.ivector('y')
        self.cost_vector = T.matrix('cost_vector')

        self.MSE = T.mean((self.cost_predicted_given_x - self.cost_vector) ** 2)

        self.error = T.mean(T.neq(self.y_pred, self.y))

        self.future_cost = T.sum(self.cost_vector[T.arange(self.y_pred.shape[0]), self.y_pred])

    def sgd_optimize(self, train_set, test_set, n_epochs, learning_rate, batch_size):
        """ Optimizing model parameters by stochastic gradient descent """

        train_set_x, train_set_y, train_set_c = train_set
        assert train_set_x.shape == (60000, 784)
        assert train_set_y.shape == (60000,)
        assert train_set_c.shape == (60000, 10)

        test_set_x, test_set_y, test_set_c = test_set
        assert test_set_x.shape == (10000, 784)
        assert test_set_y.shape == (10000,)
        assert test_set_c.shape == (10000, 10)

        from toolbox import make_shared_data

        train_set_x = make_shared_data(train_set_x)
        train_set_c = make_shared_data(train_set_c)
        train_set_y = T.cast(make_shared_data(train_set_y), 'int32')

        test_set_x = make_shared_data(test_set_x)
        test_set_c = make_shared_data(test_set_c)
        test_set_y = T.cast(make_shared_data(test_set_y), 'int32')

        print '... building the model'

        index = T.lscalar()

        cost = self.MSE

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
                self.cost_vector: train_set_c[index * batch_size: (index + 1) * batch_size]
            },
            name='train_model'
        )

        in_sample_result = theano.function(
            inputs=[],
            outputs=[self.error, self.future_cost],
            givens={
                self.input: train_set_x,
                self.y: train_set_y,
                self.cost_vector: train_set_c
            },
            name='in_sample_result'
        )

        out_sample_result = theano.function(
            inputs=[],
            outputs=[self.error, self.future_cost],
            givens={
                self.input: test_set_x,
                self.y: test_set_y,
                self.cost_vector: test_set_c
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
    """ Demo """
    print '... loading dataset'

    from toolbox import CostMatrixGenerator, load_mnist, class_to_example

    train_set, test_set = load_mnist()
    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set

    cmg = CostMatrixGenerator(train_set_y, 10)
    cost_mat = cmg.absolute()

    train_set_c = class_to_example(train_set_y, cost_mat)
    test_set_c = class_to_example(test_set_y, cost_mat)

    reg = LinearCostRegressor(
        input=T.matrix('input'),
        n_in=28 * 28,
        n_out=10
    )

    reg.sgd_optimize(
        train_set=[train_set_x, train_set_y, train_set_c],
        test_set=[test_set_x, test_set_y, test_set_c],
        n_epochs=100,
        learning_rate=0.13,
        batch_size=600
    )


if __name__ == '__main__':
    main()
