# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

from libs.dfs import dfs_program

__author__ = 'fyabc'


def test():
    result = dfs_program(
        io_pair_list=[
            ([2, [3, 5, 4, 7, 5]], 7),
            ([3, [5, 4, 3, 10, 8, 11]], 12),
            ([3, [1, 2, 2, 9, 19]], 5),
            ([6, [1, 7]], 8),
        ],
        max_depth=3,
    )

    print(result)


if __name__ == '__main__':
    test()
