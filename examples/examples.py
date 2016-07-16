#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import theano.tensor as T
from theano import function, In, shared

__author__ = 'fyabc'


def getLogistic():
    x = T.dmatrix('x')
    s = 1 / (1 + T.exp(-x))
    # s = (1 + T.tanh(x / 2)) / 2
    return function([x], s)


def multiReturn():
    a, b = T.dmatrices('a', 'b')
    diff = a - b
    absDiff = abs(diff)
    diffSquared = diff ** 2
    return function([a, b], [diff, absDiff, diffSquared])


def defaultValue():
    x, y, z = T.dscalars('x', 'y', 'z')
    return function([x, In(y, value=1), In(z, value=2, name='namedZ')], (x + y) * z)


def sharedVar():
    state = shared(0)
    inc = T.iscalar('inc')
    acc = function([inc], state, updates=[(state, state + inc)])

    print(state.get_value())
    acc(3)
    print(state.get_value())
    acc(16)
    print(state.get_value())
    state.set_value(-1)
    acc(5)
    print(state.get_value())

    dec = function([inc], state, updates=[(state, state - inc)])
    dec(2)
    print(state.get_value())


def copyFunc():
    state = shared(0)
    inc = T.iscalar('inc')
    acc = function([inc], state, updates=[(state, state + inc)])

    print(acc(5))
    print(state.get_value())

    newState = shared(0)
    newAcc = acc.copy(swap={state: newState})

    print(newAcc(100))
    print(newState.get_value())
    print(state.get_value())

    nullAcc = acc.copy(delete_updates=True)

    print(nullAcc(1000))
    print(state.get_value())


def randomState():
    pass


def main():
    # logistic = getLogistic()
    # print(logistic([[0, 1], [-1, -2]]))
    #
    # mulRet = multiReturn()
    # print(mulRet([[1, 1], [1, 1]], [[0, 1], [2, 3]]))
    #
    # defaultVal = defaultValue()
    # print(defaultVal(33, namedZ=14, y=-32))
    #
    sharedVar()
    #
    # copyFunc()

if __name__ == '__main__':
    main()
