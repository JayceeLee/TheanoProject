# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

from libs.dsl_types import *
from libs.dsl import Program

__author__ = 'fyabc'


def test():
    program = Program(
        input_info=[INT, LIST],
        statements="""\
c = Sort i1
d = Take i0 c
e = Sum d
""",
    )

    print(program)

    print(program([2, [3, 5, 4, 7, 5]]))
    print(program([3, [5, 4, 3, 10, 8, 11]]))
    print(program([3, [1, 2, 2, 9, 19]]))
    print(program([6, [1, 7]]))


if __name__ == '__main__':
    test()
