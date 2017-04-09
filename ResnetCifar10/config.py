# -*- coding: utf-8 -*-

from __future__ import print_function

__author__ = 'fyabc'


Config = {
    'data_dir': './data/cifar10',
    'one_file': False,
    'validation_size': 10000,
    'test_size': 10000,

    # Number of residual blocks
    'n': 5,

    "l2_penalty_factor": 0.0001,

    'batch_size': 128,
    'valid_batch_size': 500,
    'learning_rate': 0.1,
    "optimizer": 'sgd',

    'num_epoch': 82,
}

C = Config


__all__ = [
    'Config',
    'C',
]
