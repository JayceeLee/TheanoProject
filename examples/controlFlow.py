#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from theano import function
from theano import tensor as T
from theano.ifelse import ifelse
from theano import scan
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


def myPower():
    # Calculate A ** k.

    k = T.iscalar('k')
    A = T.ivector('A')

    # Symbolic description of the result
    #
    # The parameter order of fn:
    #   the output of the prior call to fn (or the initial value, initially),
    #   followed by all non_sequences.
    #
    # outputs_info is the initial value.
    #
    # scan returns the result and a dictionary of updates.
    #   the result is a 3D tensor which contains the value in each step.
    result, updates = scan(fn=lambda priorResult, A: priorResult * A,
                           outputs_info=T.ones_like(A),
                           non_sequences=A,
                           n_steps=k)

    # We only care about A**k, but scan has provided us with A**1 through A**k.
    # Discard the values that we don't care about. Scan is smart enough to
    # notice this and not waste memory saving them.
    finalResult = result[-1]

    power = function(inputs=[A, k], outputs=finalResult, updates=updates)

    print(power(range(10), 2))
    print(power(range(10), 4))
    print(power([1, 2, 3], 10))


if __name__ == '__main__':
    # conditions()
    myPower()
