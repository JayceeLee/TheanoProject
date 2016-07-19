#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import numpy as np
from theano import function
from theano import tensor as T
from theano import pp, scan


__author__ = 'fyabc'


def derivative():
    x = T.dscalar('x')
    y = x ** 2
    gy = T.grad(y, x)
    print(pp(gy))

    f = function([x], gy)
    print(f(4))
    print(np.allclose(f(94.2), 94.2 * 2))
    print(pp(f.maker.fgraph.outputs[0]))


def gradSigmoid():
    x = T.dmatrix('x')
    s = T.sum(1 / (1 + T.exp(-x)))
    gs = T.grad(s, x)
    ds = function([x], gs)

    result = ds([
        [0, 1],
        [-1, -2]
    ])
    print(result)


def getJacobian():
    x = T.dvector('x')
    y = x ** 2
    J, updates = scan(lambda i, y, x: T.grad(y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y, x])
    f = function([x], J, updates=updates)
    print(f([4, 7]))


def getHessian():
    x = T.dvector('x')
    y = x ** 2
    cost = y.sum()
    gy = T.grad(cost, x)
    H, updates = scan(lambda i, gy, x: T.grad(gy[i], x), sequences=T.arange(gy.shape[0]), non_sequences=[gy, x])
    f = function([x], H, updates=updates)
    print(f([4, 7]))


def main():
    # derivative()
    # gradSigmoid()
    # getJacobian()
    getHessian()


if __name__ == '__main__':
    main()
