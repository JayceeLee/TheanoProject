#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from collections import defaultdict
from types import StringTypes

__author__ = 'fyabc'


# Type ID constants
INT = 0             # Data type: int
LIST = 1            # Data type: [int]
INT2INT = 2         # Lambda type: int -> int
INT2BOOL = 3        # Lambda type: int -> bool
INT2INT2INT = 4     # Lambda type: int -> int -> int


class MapToId(object):
    """Base class to map all instances to unique ids."""

    _AllInstances = []

    def __init__(self):
        self._AllInstances.append(self)

    @classmethod
    def get(cls, i):
        return cls._AllInstances[i]


class Function(MapToId):
    """The class of function."""

    _AllInstances = []

    def __init__(self, name, func, arg_types, ret_type):
        super(Function, self).__init__()
        self.name = name
        self.func = func
        self.arg_types = arg_types
        self.ret_type = ret_type

    def run(self, args):
        return Data(self.func(*(arg.value for arg in args)))

    __call__ = run

    def __str__(self):
        return self.name

    __repr__ = __str__

    def repr_call(self, args):
        return '{}({})'.format(self, ', '.join(str(arg) for arg in args))


class Data(object):
    """The class of data."""

    def __init__(self, value):
        self.value = value

        if isinstance(value, (list, tuple)):
            self.type = LIST
        else:
            self.type = INT

    def __str__(self):
        return 'Data({})'.format(self.value)

    __repr__ = __str__


class Lambda(MapToId):
    """The class of lambda."""

    _AllInstances = []

    def __init__(self, name, func, type_):
        super(Lambda, self).__init__()
        self.name = name
        self.value = func
        self.type = type_

    def __str__(self):
        return self.name

    __repr__ = __str__


class DSL(object):
    """The class of the DSL."""

    # List of all function ids.
    functions = []

    # Table of all lambdas.
    # Key: lambda type id
    # Value: list of lambda ids
    lambdas = defaultdict(list)

    @classmethod
    def set_functions_lambdas(cls):
        """Set all functions and lambdas of the DSL."""

        # First-order functions
        head = Function(
            name='Head',
            func=lambda xs: xs[0] if len(xs) > 0 else None,
            arg_types=[LIST],
            ret_type=INT,
        )

        last = Function(
            name='Last',
            func=lambda xs: xs[-1] if len(xs) > 0 else None,
            arg_types=[LIST],
            ret_type=INT,
        )

        take = Function(
            name='Last',
            func=lambda n, xs: xs[:n],
            arg_types=[INT, LIST],
            ret_type=LIST,
        )

        drop = Function(
            name='Drop',
            func=lambda n, xs: xs[n:],
            arg_types=[INT, LIST],
            ret_type=LIST,
        )

        # High-order functions


DSL.set_functions_lambdas()


class Statement(object):
    """The class of statement."""

    def __init__(self, function, arg_list, ret):
        """

        :param function: int
            Function id.
        :param arg_list: list
            Arguments.
            Argument may be int (lambda id) or string (variable name)
        :param ret: str
            Return variable name.
        """
        self.ret = ret
        self.function = function
        self.arg_list = arg_list

    def run(self, variables):
        """Run the statement, add the result to `variables`

        :param variables: Variable table
        """

        variables[self.ret] = Function.get(self.function)([
            variables[arg] if isinstance(arg, StringTypes) else Lambda.get(arg)
            for arg in self.arg_list
        ])


class Program(object):
    """The class of program.

    todo
    """

    def __init__(self, input_info, statements):
        """

        :param input_info: list
            Input information.
            A list of input type ids.
        :param statements: list
            Statement information.
            A list of statements.
        """

        # Table of variables.
        # Key: variable names
        # Value: variable value or None (None means variables have not assigned yet)
        #
        # [NOTE] Reserve variables:
        # i0, i1, i2, ...: input variables
        # o: output variable
        self.variables = {
            'i{}'.format(i): None
            for i in range(len(input_info))
        }

        self.input_info = input_info
        self.statements = statements

    def run(self, input_list):
        """Run the program with given input, return the output.

        :param input_list: list
            Input list.
            Each element is a `Data` object or raw data
        :return: Data
            The result.
        """

        # Set input variables
        for i, input_data in enumerate(input_list):
            if not isinstance(input_data, Data):
                input_data = Data(input_data)
            self.variables['i{}'.format(i)] = input_data

        for statement in self.statements:
            statement.run(self.variables)

        return self.variables['o']

    __call__ = run


def dfs_program(io_pair_list):
    pass


def test():
    prog = Program(
        input_info=[INT, LIST],
        statements=[
            Statement(
                function=2,
                arg_list=['i0', 'i1'],
                ret='o',
            ),
        ],
    )

    print(prog([2, [2, 3, 4]]))


if __name__ == '__main__':
    test()

