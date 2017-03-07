#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import json
from copy import copy

import numpy as np

from .utils.utils import fX

__author__ = 'fyabc'


MaxArgNum = 3
ArgDepth = 10   # 0 ~ 9 digit, one-hot.

# Program symbols.
PG_Continue = 0
PG_Return = 1


class IntegerArguments(object):
    depth = ArgDepth
    max_arg_num = MaxArgNum
    size_if_arguments = depth * max_arg_num

    def __init__(self, args, values):
        if values is not None:
            self.values = values.reshape((self.max_arg_num, self.depth))
        else:
            self.values = np.zeros((self.max_arg_num, self.depth), dtype=fX)

        if args:
            for i, arg in enumerate(args):
                self.update_to(i, arg)

    def copy(self):
        result = self.__class__.__new__(self.__class__)
        result.values = np.copy(self.values)

        return result

    def decode_all(self):
        return [self.decode_at(i) for i in xrange(len(self.values))]

    def decode_at(self, index):
        return self.values[index].argmax()

    def update_to(self, index, integer):
        self.values[index] = 0
        self.values[index, int(np.clip(integer, 0, self.depth - 1))] = 1

    def __str__(self):
        return '<IA: {}>'.format(self.decode_all())

    __repr__ = __str__


class Program(object):
    output_to_env = False

    def __init__(self, name, *args):
        self.name = name
        self.args = args
        self.program_id = None

    def description_with_args(self, args):
        return '{}({})'.format(self.name, ', '.join(str(x) for x in args.decode_all()))

    def to_one_hot(self, size, dtype=fX):
        result = np.zeros((size,), dtype=dtype)
        result[self.program_id] = 1
        return result

    def do(self, env, args):
        raise NotImplementedError()

    def __str__(self):
        return '<Program: name={}>'.format(self.name)

    __repr__ = __str__
