#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

# import numpy as np
import theano.tensor as T
from theano import function, pp

__author__ = 'fyabc'


def main():
    x = T.dscalar('x')
    y = T.dscalar('y')
    z = x + y
    f = function([x, y], z)

    xm = T.dmatrix('x')
    ym = T.dmatrix('y')
    fm = function([xm, ym], xm * ym)

    print(pp(xm * ym + 4 / ym))
    print(f(2, 3), fm([[1, 2], [3, 4]], [[5, 6], [7, 8]]))

    xv = T.vector()
    yv = T.vector()
    fv = function([xv, yv], xv ** 2 + yv ** 2 + 2 * xv * yv)

    print(fv([1, 2], [3, 4]))


if __name__ == '__main__':
    main()
