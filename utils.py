#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import gzip

import numpy as np
import theano as tn
import theano.tensor as T

from sys import version_info
if version_info.major < 3:
    import cPickle as pkl
else:
    import pickle as pkl

__author__ = 'fyabc'


def getMNIST():
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    trainSet, validSet, testSet = pkl.load(f)   # , encoding='latin1')
    f.close()
    return trainSet, validSet, testSet


def sharedDataset(dataXY):
    dataX, dataY = dataXY
    sharedX = tn.shared(np.asarray(dataX, dtype=tn.config.floatX))
    sharedY = tn.shared(np.asarray(dataY, dtype=tn.config.floatX))

    return sharedX, T.cast(sharedY, 'int32')
