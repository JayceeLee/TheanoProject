# -*- coding: utf-8 -*-

from __future__ import print_function

import os

import numpy as np

from config import C
from utils import floatX, f_open, fX

__author__ = 'fyabc'


def load_cifar10_data(data_dir=None, one_file=None):
    """

    Args:
        data_dir: directory of the CIFAR-10 data.
        one_file: is the data in a single file?

    Returns:
        A dict, which contains train data and test data.
        Shapes of data:
            x_train:    (100000, 3, 32, 32)
            y_train:    (100000,)
            x_test:     (10000, 3, 32, 32)
            y_test:     (10000,)
    """

    def process(x):
        x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
        x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0, 3, 1, 2)

        # subtract per-pixel mean
        pixel_mean = np.mean(x[0:train_size], axis=0)
        # pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
        x -= pixel_mean

        return x

    data_dir = data_dir or C['data_dir']
    one_file = one_file or C['one_file']

    if not os.path.exists(data_dir):
        raise Exception("CIFAR-10 dataset can not be found. Please download the dataset from "
                        "'https://www.cs.toronto.edu/~kriz/cifar.html'.")

    # train_size = ParamConfig['train_size']
    train_size = 50000

    if one_file:
        train, test = f_open(data_dir)
        x_train, y_train = train
        x_test, y_test = test

        x_train = process(x_train)
        x_test = process(x_test)
    else:
        xs = []
        ys = []
        for j in range(5):
            d = f_open(data_dir + '/data_batch_%d' % (j + 1))
            xs.append(d['data'])
            ys.append(d['labels'])

        d = f_open(data_dir + '/test_batch')
        xs.append(d['data'])
        ys.append(d['labels'])

        x = np.concatenate(xs) / np.float32(255)
        y = np.concatenate(ys)

        x = process(x)

        x_train = x[0:train_size, :, :, :]
        y_train = y[0:train_size]

        x_test = x[train_size:, :, :, :]
        y_test = y[train_size:]

    # create mirrored images
    x_train_flip = x_train[:, :, :, ::-1]
    y_train_flip = y_train
    x_train = np.concatenate((x_train, x_train_flip), axis=0)
    y_train = np.concatenate((y_train, y_train_flip), axis=0)

    return {
        'x_train': floatX(x_train),
        'y_train': y_train.astype('int32'),
        'x_test': floatX(x_test),
        'y_test': y_test.astype('int32')
    }


def split_cifar10_data(data):
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    # One: validate is not part of train
    x_validate = x_test[:C['validation_size']]
    y_validate = y_test[:C['validation_size']]
    x_test = x_test[-C['test_size']:]
    y_test = y_test[-C['test_size']:]

    # # Another: validate is part of train
    # x_validate = x_train[:C['validation_size']]
    # y_validate = y_train[:C['validation_size']]

    return x_train, y_train, x_validate, y_validate, x_test, y_test


def prepare_CIFAR10_data(inputs, targets):
    batch_size = len(targets)
    padded = np.pad(inputs, ((0, 0), (0, 0), (4, 4), (4, 4)), mode=str('constant'))
    random_cropped = np.zeros_like(inputs, dtype=fX)
    crops = np.random.random_integers(0, high=8, size=(batch_size, 2))

    for r in range(batch_size):
        random_cropped[r, :, :, :] = padded[r, :, crops[r, 0]:(crops[r, 0] + 32), crops[r, 1]:(crops[r, 1] + 32)]

    return random_cropped, targets


def pre_process_CIFAR10_data():
    # Load the dataset
    x_train, y_train, x_validate, y_validate, x_test, y_test = split_cifar10_data(load_cifar10_data())

    train_size, validate_size, test_size = y_train.shape[0], y_validate.shape[0], y_test.shape[0]

    print('Training data size:', train_size)
    print('Validation data size:', validate_size)
    print('Test data size:', test_size)

    return x_train, y_train, x_validate, y_validate, x_test, y_test, train_size, validate_size, test_size
