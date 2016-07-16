#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import numpy as np
import theano
import theano.tensor as T

__author__ = 'fyabc'

RNG = np.random


def main():
    N = 400         # training sample size
    feats = 784     # number of input variables

    # generate a dataset: D = (input_values, target_class)
    # [NOTE]: randint is [low, high)
    D = (RNG.randn(N, feats), RNG.randint(size=N, low=0, high=2))
    trainingSteps = 10000

    # Declare Theano symbolic variables
    x = T.dmatrix('x')
    y = T.dvector('y')

    # initialize the weight vector w randomly
    #
    # this and the following bias variable b
    # are shared so they keep their values
    # between training iterations (updates)
    w = theano.shared(RNG.randn(feats), name='w')

    # initialize the bias term
    b = theano.shared(0., name='b')

    print('Initial model:')
    print(w.get_value())
    print(b.get_value())

    # Construct Theano expression graph
    p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))             # Probability that target = 1
    prediction = p_1 > 0.5                              # The prediction thresholded
    xCE = -y * T.log(p_1) - (1 - y) * T.log(1 - p_1)    # Cross-entropy loss function
    cost = xCE.mean() + 0.01 * (w ** 2).sum()           # The cost to minimize
    gw, gb = T.grad(cost, [w, b])                       # Compute the gradient of the cost
    # w.r.t weight vector w and
    # bias term b
    # (we shall return to this in a
    # following section of this tutorial)

    # Compile
    train = theano.function(
        inputs=[x, y],
        outputs=[prediction, xCE],
        updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
    predict = theano.function(inputs=[x], outputs=prediction)

    # Train
    for i in range(trainingSteps):
        pred, err = train(D[0], D[1])
        # print('Step %d: pred = %s, err = %s' % (i, str(pred), str(err)))
        print('Step %d' % (i,))

    print("Final model:")
    print(w.get_value())
    print(b.get_value())
    print("target values for D:")
    print(D[1])
    print("prediction on D:")
    print(predict(D[0]))
    print("Error:")
    print(D[1] - predict(D[0]))


if __name__ == '__main__':
    main()
