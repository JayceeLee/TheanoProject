#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import theano
import theano.tensor as T
import numpy as np

__author__ = 'fyabc'

fX = 'float32'


def test_scan():
    v = T.vector('v', dtype=fX)

    x_init = T.alloc(np.float32(1.), 1)

    a = theano.shared(np.float32(5.))

    def step(v_i, x_i, a_):
        return v_i * x_i + a_

    outputs, updates = theano.scan(
        step,
        sequences=v,
        outputs_info=x_init,
        non_sequences=a,
        n_steps=v.shape[0],
    )

    f = theano.function([v], outputs)

    print(f([2, 4, 3, 0, -2]))


if __name__ == '__main__':
    test_scan()
