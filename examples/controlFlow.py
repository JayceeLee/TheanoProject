#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from theano import function
from theano import tensor as T
from theano.ifelse import ifelse
import theano
import numpy
import time

__author__ = 'fyabc'


def conditions():
    a, b = T.scalars('a', 'b')
    x, y = T.matrices('x', 'y')

    z_switch = T.switch(T.lt(a, b), T.mean(x), T.mean(y))
    z_lazy = ifelse(T.lt(a, b), T.mean(x), T.mean(y))

    f_switch = function([a, b, x, y], z_switch, mode=theano.Mode(linker='vm'), allow_input_downcast=True)
    f_lazyIfElse = function([a, b, x, y], z_lazy, mode=theano.Mode(linker='vm'), allow_input_downcast=True)

    val1, val2 = 0., 1.
    bigMat1 = numpy.ones((10000, 2000))
    bigMat2 = numpy.ones((10000, 2000))

    n_times = 10

    tic = time.clock()
    for i in range(n_times):
        f_switch(val1, val2, bigMat1, bigMat2)
    print('time spent evaluating both values %f sec' % (time.clock() - tic))

    tic = time.clock()
    for i in range(n_times):
        f_lazyIfElse(val1, val2, bigMat1, bigMat2)
    print('time spent evaluating one value %f sec' % (time.clock() - tic))


if __name__ == '__main__':
    conditions()
