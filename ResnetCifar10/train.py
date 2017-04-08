#! /usr/bin/python
# -*- encoding: utf-8 -*-

"""Resnet on CIFAR-10 dataset.

Theano code, with user defined optimizers.
"""

from __future__ import print_function

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import InputLayer
from lasagne.layers import LocalResponseNormalization2DLayer, MaxPool2DLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import PadLayer
from lasagne.layers import batch_norm
from lasagne.layers.helper import get_all_param_values, set_all_param_values
from lasagne.nonlinearities import softmax, rectify

from config import C
from data import prepare_CIFAR10_data, pre_process_CIFAR10_data
from optimizers import Optimizers

__author__ = 'fyabc'


class ResnetModel(object):
    def __init__(self, options=None):
        self.O = C if options is None else options

        # Prepare Theano variables for inputs and targets
        self.input_var = T.tensor4('inputs')
        self.target_var = T.ivector('targets')

        self.learning_rate = T.scalar('lr')

        self.network = self.build_model()

        self.build_train_function()

    def build_model(self, input_var=None, n=None):
        n = self.O['n'] if n is None else n
        input_var = self.input_var if input_var is None else input_var

        # create a residual learning building block with two stacked 3x3 conv-layers as in paper
        def residual_block(layer_, increase_dim=False, projection=False):
            input_num_filters = layer_.output_shape[1]
            if increase_dim:
                first_stride = (2, 2)
                out_num_filters = input_num_filters * 2
            else:
                first_stride = (1, 1)
                out_num_filters = input_num_filters

            stack_1 = batch_norm(
                ConvLayer(layer_, num_filters=out_num_filters, filter_size=(3, 3), stride=first_stride,
                          nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'),
                          flip_filters=False))
            stack_2 = batch_norm(
                ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3, 3), stride=(1, 1),
                          nonlinearity=None,
                          pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

            # add shortcut connections
            if increase_dim:
                if projection:
                    # projection shortcut, as option B in paper
                    projection = batch_norm(
                        ConvLayer(layer_, num_filters=out_num_filters, filter_size=(1, 1), stride=(2, 2),
                                  nonlinearity=None, pad='same', b=None, flip_filters=False))
                    block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]), nonlinearity=rectify)
                else:
                    # identity shortcut, as option A in paper
                    identity = ExpressionLayer(layer_, lambda X: X[:, :, ::2, ::2],
                                               lambda s: (s[0], s[1], s[2] // 2, s[3] // 2))
                    padding = PadLayer(identity, [out_num_filters // 4, 0, 0], batch_ndim=1)
                    block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]), nonlinearity=rectify)
            else:
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, layer_]), nonlinearity=rectify)

            return block

        # Building the network
        layer_in = InputLayer(shape=(None, 3, 32, 32), input_var=input_var)

        # first layer, output is 16 x 32 x 32
        layer = batch_norm(ConvLayer(layer_in, num_filters=16, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify,
                                     pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        # first_layer_output = lasagne.layers.get_output(layer, inputs=input_var)
        # self.f_first_layer_output = theano.function(
        #     inputs=[input_var],
        #     outputs=first_layer_output
        # )

        # first stack of residual blocks, output is 16 x 32 x 32
        for _ in range(n):
            layer = residual_block(layer)

        # second stack of residual blocks, output is 32 x 16 x 16
        layer = residual_block(layer, increase_dim=True)
        for _ in range(1, n):
            layer = residual_block(layer)

        # third stack of residual blocks, output is 64 x 8 x 8
        layer = residual_block(layer, increase_dim=True)
        for _ in range(1, n):
            layer = residual_block(layer)

        # average pooling
        layer = GlobalPoolLayer(layer)

        # fully connected layer
        return DenseLayer(
            layer, num_units=10,
            W=lasagne.init.HeNormal(),
            nonlinearity=softmax)

    def build_train_function(self):
        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        probs = lasagne.layers.get_output(self.network)
        self.f_probs = theano.function(
            inputs=[self.input_var],
            outputs=probs
        )

        loss = lasagne.objectives.categorical_crossentropy(probs, self.target_var)

        self.f_cost_list_without_decay = theano.function([self.input_var, self.target_var], loss)

        loss = loss.mean()

        self.f_cost_without_decay = theano.function([self.input_var, self.target_var], loss)

        # add weight decay
        all_layers = lasagne.layers.get_all_layers(self.network)
        l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * \
            self.O['l2_penalty_factor']
        loss += l2_penalty

        self.f_cost = theano.function([self.input_var, self.target_var], loss)

        # Create update expressions for training
        # Stochastic Gradient Descent (SGD) with momentum
        params = lasagne.layers.get_all_params(self.network, trainable=True)

        grads = T.grad(loss, wrt=params)

        self.f_grad_shared, self.f_update = Optimizers[self.O['optimizer']](
            self.learning_rate, params, grads, [self.input_var], loss)


def main(options):
    C.update(options)

    model = ResnetModel()

    x_train, y_train, x_validate, y_validate, x_test, y_test, train_size, validate_size, test_size = \
        pre_process_CIFAR10_data()

    for epoch in range(C['num_epoch']):
        pass
    

if __name__ == '__main__':
    main({

    })
