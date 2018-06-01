import numpy as np
import theano
import theano.tensor as T


class OneSidedCostRegressor(object):

    def __init__(self, input, n_in, n_out):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # keep track of model input
        self.input = input
        # symbolic variable of cost vector in terms of a single example;
        # for a batch of examples, it's a matrix, so don't get confused
        # with the variable name `cost_vector`
        self.cost_vector = T.matrix('cost_vector')
        # symbolic variable of Z_{n,k}
        self.Z_nk = T.matrix('Z_nk')

        self.cost_predicted_given_x = T.dot(self.input, self.W) + self.b

        # elementwise comparison with 0
        self.xi = T.maximum((self.Z_nk * (self.cost_predicted_given_x - self.cost_vector)), 0.)

        # define the linear one-sisded regression loss
        self.one_sided_regression_loss = T.sum(self.xi)

        # symbolic description of how to compute prediction as class whose
        # cost is minimum
        self.y_pred = T.argmin(self.cost_predicted_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # symbolic variable of labels, will only be used for computing 0/1 errors
        self.y = T.ivector('y')
        # compute the 0/1 loss
        self.error = T.mean(T.neq(self.y_pred, self.y))

        # when a new example comes in, the model first computes (predicts) its
        # cost on classifying into each class (a vector) by
        # `self.cost_predicted_given_x`; then the model will predict this new
        # example as label with the smallest cost;
        # self.future_cost = T.sum(self.cost_vector[T.arange(self.y_pred.shape[0]), self.y_pred])
        self.future_cost = T.mean(self.cost_vector[T.arange(self.y_pred.shape[0]), self.y_pred])

    def sgd_optimize(self, train_set, test_set, n_epochs, learning_rate, batch_size):
        """ Optimizing model parameters by stochastic gradient descent """

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

        index = T.lscalar()  # symbolic variable for index to a mini-batch

        cost = self.one_sided_regression_loss

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
                self.cost_vector: train_set_c[index * batch_size: (index + 1) * batch_size],
                self.Z_nk: train_set_z[index * batch_size: (index + 1) * batch_size]
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
