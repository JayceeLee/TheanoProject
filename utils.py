#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import gzip
import os

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
    data_path = 'data/mnist.pkl.gz'

    if not os.path.exists(data_path):
        print('MNIST dataset not found, please download it.')
        exit(1)

    f = gzip.open(data_path, 'rb')
    trainSet, validSet, testSet = pkl.load(f)   # , encoding='latin1')
    f.close()
    return trainSet, validSet, testSet


def sharedDataset(dataXY):
    dataX, dataY = dataXY
    sharedX = tn.shared(np.asarray(dataX, dtype=tn.config.floatX))
    sharedY = tn.shared(np.asarray(dataY, dtype=tn.config.floatX))

    return sharedX, T.cast(sharedY, 'int32')
