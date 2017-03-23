# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import time
from functools import wraps

__author__ = 'fyabc'


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()

        print('Time of function {}: {:.6f}s'.format(func.__name__, end_time - start_time))

        return result

    return wrapper
