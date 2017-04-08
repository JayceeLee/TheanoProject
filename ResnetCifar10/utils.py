# -*- coding: utf-8 -*-

from __future__ import print_function

import cPickle as pkl
import gzip

import numpy as np

__author__ = 'fyabc'


fX = 'float32'


def floatX(value):
    return np.asarray(value, dtype=fX)


def f_open(filename, mode='rb', unpickle=True):
    if filename.endswith('.gz'):
        _open = gzip.open
    else:
        _open = open

    if unpickle:
        with _open(filename, 'rb') as f:
            return pkl.load(f)
    else:
        return open(filename, mode)


def itemlist(tparams):
    """Get the list of parameters: Note that tparams must be OrderedDict"""
    return [vv for kk, vv in tparams.iteritems()]


__all__ = [
    'fX',
    'floatX',
    'f_open',
    'itemlist',
]
