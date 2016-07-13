#! /usr/bin/python3
# -*- encoding: utf-8 -*-

__author__ = 'fyabc'

import theano
import theano.tensor as T
import numpy as np

from utils import getMNIST, sharedDataset


def main():
    trainSet, validationSet, testSet = getMNIST()
    trainSetX, trainSetY = sharedDataset(trainSet)
    validationSetX, validationSetY = sharedDataset(validationSet)
    testSetX, testSetY = sharedDataset(testSet)

    print(testSetX, testSetY)


if __name__ == '__main__':
    main()
