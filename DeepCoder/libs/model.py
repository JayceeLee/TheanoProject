# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

from collections import OrderedDict

__author__ = 'fyabc'


class Model(object):
    def __init__(self):
        # Parameters (Theano shared variables)
        self.parameters = OrderedDict()

    def build_model(self):
        pass

    def init_np_parameters(self):
        pass

    def init_parameters(self):
        pass
