# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

__author__ = 'fyabc'


# Type ID constants
NONE = -1           # Data type: None
INT = 0             # Data type: int
LIST = 1            # Data type: [int]
INT2INT = 2         # Lambda type: int -> int
INT2BOOL = 3        # Lambda type: int -> bool
INT2INT2INT = 4     # Lambda type: int -> int -> int

Type2String = {
    INT: 'int',
    LIST: '[int]',
    INT2INT: 'int -> int',
    INT2BOOL: 'int -> bool',
    INT2INT2INT: 'int -> int -> int',
}

IsLambda = {
    INT: False,
    LIST: False,
    INT2INT: True,
    INT2BOOL: True,
    INT2INT2INT: True,
}


def get_type_id(raw_data):
    """Get type id of the raw data."""
    if isinstance(raw_data, (list, tuple)):
        return LIST
    if isinstance(raw_data, int):
        return INT
    if isinstance(raw_data, type(None)):
        return NONE
    raise TypeError('Incorrect data type')
