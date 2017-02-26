#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import numpy as np

__author__ = 'fyabc'


# The float type of Theano. Default to 'float32'.
fX = 'float32'


def floatX(value):
    return np.asarray(value, dtype=fX)


NMTConfig = {

}
