import cPickle
import os
import sys
import timeit

import copy_reg
import types

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from dA import dA
from cifar import load_cifar
from svhn import load_SVHN


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)


class SdA(object):

    def __init__(
        self,
        numpy_rng,
        theano_rng,
        n_ins,
        hidden_layers_sizes,
        n_outs,
        corruption_levels
    ):
        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.matrix('x')
        self.y = T.ivector('y')

        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)

        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)

        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size):
        index = T.lscalar('index')
        corruption_level = T.scalar('corruption')
        learning_rate = T.scalar('lr')
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')

        gparams = T.grad(self.finetune_cost, self.params)

        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score


def test_SdA(finetune_lr=0.1, pretraining_epochs=15,
             pretrain_lr=0.001, training_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=1):

    # datasets = load_data(dataset)
    # datasets = load_cifar('/home/syang100/datasets')
    # datasets = load_cifar(
    #     path='/project/andyyuan/Datasets',
    #     grayscale=True,
    #     p_valid=0.2,
    #     unbalanced=True
    # )

    datasets = load_SVHN(
        path='/project/andyyuan/Datasets',
        grayscale=True,
        p_valid=0.2,
        unbalanced=True
    )

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset([train_set_x, train_set_y])
    valid_set_x, valid_set_y = shared_dataset([valid_set_x, valid_set_y])
    test_set_x, test_set_y = shared_dataset([test_set_x, test_set_y])

    datasets = [[train_set_x, train_set_y],
                [valid_set_x, valid_set_y],
                [test_set_x, test_set_y]]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # rng_num = np.random.randint(89677)
    rng_num = 89677
    numpy_rng = np.random.RandomState(rng_num)

    print '... building the model'

    hidden_layer_sizes=[1250, 1250, 1250]  # TODO: to change
    sda = SdA(
        numpy_rng=numpy_rng,
        theano_rng=None,
        n_ins=32 * 32,
        hidden_layers_sizes=hidden_layer_sizes,
        n_outs=10,
        corruption_levels=None
    )

    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print '... pre-training the model'
    start_time = timeit.default_timer()

    corruption_levels = [.25, .25, .25]    # TODO: to change
    for i in xrange(sda.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print np.mean(c)

    end_time = timeit.default_timer()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    # TODO: saving pre-trained model
    with open('pretrained_SdA_not_yet_finetune_' + str(rng_num) + '.pkl', 'w') as f:
        cPickle.dump(sda, f, protocol=cPickle.HIGHEST_PROTOCOL)

    # with open('pretrained_SdA_not_yet_finetune_89677.pkl', 'r') as f:
    #     sda = cPickle.load(f)

    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print '... finetunning the model'
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss:

                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = test_model()
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

                    # TODO: save this model
                    with open('best_SdA_model_pretrained_' + str(rng_num) + '.pkl', 'w') as f:
                        cPickle.dump(sda, f, protocol=cPickle.HIGHEST_PROTOCOL)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    # TODO: print information
    print 'hidden_layer_sizes: ' + str(hidden_layer_sizes)
    print 'corruption_levels: ' + str(corruption_levels)
    print 'rng_num: ' + str(rng_num)

def predict():
    """ An example of how to load a trained model and use it to predict labels. """

    # from toolbox import load_mnist
    # train_set, test_set = load_mnist()
    # teX, teY = test_set

    # datasets = load_cifar(
    #     path='/project/andyyuan/Datasets',
    #     grayscale=True,
    #     p_valid=0.2,
    #     unbalanced=True
    # )

    datasets = load_SVHN(
        path='/project/andyyuan/Datasets',
        grayscale=True,
        p_valid=0.2,
        unbalanced=True
    )

    train_set, valid_set, test_set = datasets
    teX, teY = test_set

    def count_dist(y, K):
        dist = np.zeros(K, np.int)
        for yi in y:
            dist[yi] += 1
        print dist

    # count_dist(teY, 10)

    from toolbox import CostMatrixGenerator
    cmg = CostMatrixGenerator(train_set[1], 10)
    cost_mat = cmg.general(scale=10)
    # cost_mat = cmg.absolute()

    from toolbox import naive_prediction_cost, bayes_optimal_cost

    print naive_prediction_cost(
        model_name='best_SdA_model_pretrained_' + sys.argv[1] + '.pkl',    # TODO: to change
        test_data=[teX, teY],
        cost_mat=cost_mat
    )

    print bayes_optimal_cost(
        model_name='best_SdA_model_pretrained_' + sys.argv[1] + '.pkl',    # TODO: to change
        test_data=[teX, teY],
        cost_mat=cost_mat
    )


if __name__ == '__main__':
    # test_SdA()
    predict()
