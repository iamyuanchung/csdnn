import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from toolbox import make_shared_data


class cA(object):

    def __init__(
        self,
        numpy_rng,
        input,
        W, b,
        n_visible, n_hidden,
        n_class
    ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.n_class = n_class

        # create a Theano random generator that gives symbolic random values
        self.theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # self.theano_rng will later be used to generate corrupting inputs

        bvis = theano.shared(
            value=np.zeros(
                n_visible,
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        self.W = W
        self.b = b

        self.Wu = self.W.T  # tied weight
        self.bu = theano.shared(
            value=np.zeros(
                n_visible,
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        self.Ws = theano.shared(
            value=np.asarray(
                a=numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_class)),
                    high=4 * np.sqrt(6. / (n_hidden + n_class)),
                    size=((n_hidden, n_class))
                ),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        self.bs = theano.shared(
            value=np.zeros(
                n_class,
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        self.x = input

        self.params = [self.W, self.b, self.bu, self.Ws, self.bs]

    def get_corrupted_input(self, input, corruption_level):
        # when corruption_level = 0., there's no denoising for the input, this will be
        # one of the benchmark to see whether "denoising" is necessary in cA or not
        return input *  \
               self.theano_rng.binomial(
                   size=input.shape, n=1,
                   p=1 - corruption_level,
                   dtype=theano.config.floatX
               )

    def learning_feature(
        self,
        train_set,
        n_epochs, learning_rate, batch_size,
        corruption_level, balance_coef
    ):
        # perform `denoising`
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        # map the corrupted input to hidden layer
        y = T.nnet.sigmoid(T.dot(tilde_x, self.W) + self.b)
        # maps back hidden representation to unsupervised reconstruction
        z1 = T.nnet.sigmoid(T.dot(y, self.Wu) + self.bu)
        L1 = T.mean(-T.sum(self.x * T.log(z1) + (1 - self.x) * T.log(1 - z1), axis=1))

        # perform one-sided regression to fit the cost !
        z2 = T.dot(y, self.Ws) + self.bs
        # cost_vector = T.matrix('cost_vector')
        # Z_nk = T.matrix('Z_nk')
        # xi = T.maximum((Z_nk * (z2 - cost_vector)), 0.) # xi is a matrix
        # L2 = T.sum(xi)
        # TODO: smooth logistic loss function (upper bound)
        delta = T.log(1 + T.exp(Z_nk * (z2 - cost_vector)))
        L2 = T.sum(delta)

        # symbolic variable for balance_coef
        bc = T.scalar('bc')

        cost = L1 + bc * L2

        gparams = T.grad(cost, self.params)

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        batch_index = T.lscalar('batch_index')

        train_set_x, train_set_y, train_set_c = train_set

        train_set_z = np.zeros(train_set_c.shape) - 1
        for i in xrange(train_set_z.shape[0]):
            train_set_z[i][train_set_y[i]] = 1

        train_set_x = make_shared_data(train_set_x)
        train_set_c = make_shared_data(train_set_c)
        train_set_z = make_shared_data(train_set_z)

        pretrain_model = theano.function(
            inputs=[batch_index, bc],
            outputs=[cost, L1, L2], # TODO: debug
            updates=updates,
            givens={
                self.x: train_set_x[batch_index * batch_size: (batch_index + 1) * batch_size],
                cost_vector: train_set_c[batch_index * batch_size: (batch_index + 1) * batch_size],
                Z_nk: train_set_z[batch_index * batch_size: (batch_index + 1) * batch_size]
            },
            name='pretrain_model'
        )

        n_batches = train_set_x.get_value().shape[0] / batch_size
        for epoch in xrange(n_epochs):
            epoch_cost = 0.
            L1_cost = 0.
            L2_cost = 0.
            for batch in xrange(n_batches):
                batch_cost = pretrain_model(batch, balance_coef)
                epoch_cost += batch_cost[0]
                L1_cost += batch_cost[1]
                L2_cost += batch_cost[2]
            epoch_cost /= n_batches
            L1_cost /= n_batches
            L2_cost /= n_batches
            print '        epoch #%d, loss = (%f, %f, %f)' % (epoch + 1, epoch_cost, L1_cost, L2_cost)

        y_new = T.nnet.sigmoid(T.dot(self.x, self.W) + self.b)

        transform_data = theano.function(
            inputs=[],
            outputs=y_new,
            givens={
                self.x: train_set_x
            },
            name='trainform_data'
        )

        return [transform_data(), train_set_y, train_set_c.get_value()]
