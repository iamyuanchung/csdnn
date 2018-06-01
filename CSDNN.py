import cPickle

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from mlp import HiddenLayer
from cA import cA
from osR import OneSidedCostRegressor
from toolbox import make_shared_data


class CSDNN(object):
    def __init__(
        self, numpy_rng,
        n_in, hidden_layer_sizes, n_out
    ):
        self.sigmoid_layers = []
        self.cA_layers = []
        self.params = []
        self.n_layers = len(hidden_layer_sizes)

        assert self.n_layers > 0

        self.input = T.matrix('input')

        for i in xrange(self.n_layers):

            if i == 0:
                input_size = n_in
            else:
                input_size = hidden_layer_sizes[i - 1]

            if i == 0:
                layer_input = self.input
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(
                rng=numpy_rng,
                input=layer_input,
                n_in=input_size,
                n_out=hidden_layer_sizes[i],
                activation=T.nnet.sigmoid
            )

            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            cA_layer = cA(
                numpy_rng=numpy_rng,
                input=layer_input,
                W=sigmoid_layer.W,
                b=sigmoid_layer.b,
                n_visible=input_size,
                n_hidden=hidden_layer_sizes[i],
                n_class=n_out
            )

            self.cA_layers.append(cA_layer)

        self.logLayer = OneSidedCostRegressor(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layer_sizes[-1],
            n_out=n_out
        )

        self.params.extend(self.logLayer.params)

    def pretrain(self, train_set, n_epochs, learning_rate, batch_size, corruption_levels, balance_coefs):
        for i in xrange(self.n_layers):
            print '    pretraining layer #%d' % i
            train_set = self.cA_layers[i].learning_feature(
                train_set=train_set,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                corruption_level=corruption_levels[i],
                balance_coef=balance_coefs[i]
            )

    def finetune(self, train_set, test_set, n_epochs, learning_rate, batch_size):

        train_set_x, train_set_y, train_set_c = train_set
        test_set_x, test_set_y, test_set_c = test_set

        train_set_z = np.zeros(train_set_c.shape) - 1
        for i in xrange(train_set_z.shape[0]):
            train_set_z[i][train_set_y[i]] = 1

        train_set_x = make_shared_data(train_set_x)
        train_set_c = make_shared_data(train_set_c)
        train_set_z = make_shared_data(train_set_z)
        train_set_y = T.cast(make_shared_data(train_set_y), 'int32')

        test_set_x = make_shared_data(test_set_x)
        test_set_c = make_shared_data(test_set_c)
        test_set_y = T.cast(make_shared_data(test_set_y), 'int32')

        index = T.lscalar()  # symbolic variable for index to a mini-batch

        cost = self.logLayer.one_sided_regression_loss

        gparams = T.grad(cost, self.params)

        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=[
                (param, param - learning_rate * gparam)
                for param, gparam in zip(self.params, gparams)
            ],
            givens={
                self.input: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.logLayer.cost_vector: train_set_c[index * batch_size: (index + 1) * batch_size],
                self.logLayer.Z_nk: train_set_z[index * batch_size: (index + 1) * batch_size]
            },
            name='train_model'
        )

        in_sample_result = theano.function(
            inputs=[],
            outputs=[self.logLayer.error, self.logLayer.future_cost],
            givens={
                self.input: train_set_x,
                self.logLayer.y: train_set_y,
                self.logLayer.cost_vector: train_set_c
            },
            name='in_sample_result'
        )

        out_sample_result = theano.function(
            inputs=[],
            outputs=[self.logLayer.error, self.logLayer.future_cost],
            givens={
                self.input: test_set_x,
                self.logLayer.y: test_set_y,
                self.logLayer.cost_vector: test_set_c
            },
            name='out_sample_result'
        )

        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

        best_Cout = np.inf
        corresponding_epoch = None
        corresponding_Eout = None
        for epoch in xrange(n_epochs):
            current_batch_cost = 0.
            for batch_index in xrange(n_train_batches):
                current_batch_cost += train_model(batch_index)
            print '    epoch #%d, loss = %f' % (epoch + 1, current_batch_cost / n_train_batches)
            Ein, Cin = in_sample_result()
            Eout, Cout = out_sample_result()
            if Cout < best_Cout:
                best_Cout = Cout
                corresponding_Eout = Eout
                corresponding_epoch = epoch + 1
                print '        better performance achieved ... best_Cout = %f' % best_Cout

        print 'after training %d epochs, best_Cout = %f, occured in epoch #%d, and corresponding_Eout = %f'   \
               % (n_epochs, best_Cout, corresponding_epoch, corresponding_Eout)
