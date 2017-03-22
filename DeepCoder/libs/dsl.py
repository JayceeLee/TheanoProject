#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import operator
from types import StringTypes

from dsl_types import INT, LIST, INT2INT, INT2BOOL, INT2INT2INT, Type2String

__author__ = 'fyabc'


class MapToId(object):
    """Base class to map all instances to unique ids."""

    _AllInstances = []

    def __init__(self):
        self.id = len(self._AllInstances)
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

    # Table of functions and lambdas. Used for user to simplify the creation of function
    # Key: function or lambda name
    # Value: function or lambda
    functions_lambdas = {}

    @classmethod
    def _register_function(cls, name, **kwargs):
        if kwargs.pop('is_lambda', False):
            cls.functions_lambdas[name] = Lambda(name=name, **kwargs)
        else:
            cls.functions_lambdas[name] = Function(name=name, **kwargs)

    @classmethod
    def register_functions_lambdas(cls):
        """Set all functions and lambdas of the DSL."""

        # First-order functions
        cls._register_function('Head', func=lambda xs: xs[0] if len(xs) > 0 else None, arg_types=[LIST], ret_type=INT, )
        cls._register_function('Last', func=lambda xs: xs[-1] if len(xs) > 0 else None, arg_types=[LIST], ret_type=INT, )
        cls._register_function('Take', func=lambda n, xs: xs[:n], arg_types=[INT, LIST], ret_type=LIST, )
        cls._register_function('Drop', func=lambda n, xs: xs[n:], arg_types=[INT, LIST], ret_type=LIST, )
        cls._register_function('Access', func=lambda n, xs: xs[n] if 0 <= n < len(xs) else None, arg_types=[INT, LIST],
                               ret_type=LIST, )
        cls._register_function('Minimum', func=lambda xs: min(xs) if xs else None, arg_types=[LIST], ret_type=INT, )
        cls._register_function('Maximum', func=lambda xs: max(xs) if xs else None, arg_types=[LIST], ret_type=INT, )
        cls._register_function('Reverse', func=lambda xs: list(reversed(xs)), arg_types=[LIST], ret_type=LIST, )
        cls._register_function('Sort', func=sorted, arg_types=[LIST], ret_type=LIST, )
        cls._register_function('Sum', func=sum, arg_types=[LIST], ret_type=INT, )

        # High-order functions
        cls._register_function('Map', func=map, arg_types=[INT2INT, LIST], ret_type=LIST, )
        cls._register_function('Filter', func=filter, arg_types=[INT2BOOL, LIST], ret_type=LIST, )
        cls._register_function('Count', func=lambda f, xs: len(filter(f, xs)), arg_types=[INT2INT, LIST], ret_type=LIST)
        cls._register_function('ZipWith', func=lambda f, xs, ys: [f(x, y) for x, y in zip(xs, ys)],
                               arg_types=[INT2INT2INT, LIST, LIST], ret_type=LIST)

        def _scanl1(f, xs):
            if not xs:
                return []
            ys = [xs[0]]

            for i in range(1, len(xs)):
                ys.append(f(ys[-1], xs[i]))

            return ys
        cls._register_function('Scanl1', func=_scanl1, arg_types=[INT2INT2INT, LIST], ret_type=LIST, )

        # Lambdas
        cls._register_function('(+1)', func=lambda x: x + 1, type_=INT2INT, is_lambda=True, )
        cls._register_function('(-1)', func=lambda x: x - 1, type_=INT2INT, is_lambda=True, )
        cls._register_function('(*2)', func=lambda x: x * 2, type_=INT2INT, is_lambda=True, )
        cls._register_function('(/2)', func=lambda x: x // 2, type_=INT2INT, is_lambda=True, )
        cls._register_function('(*(-1))', func=lambda x: x * -1, type_=INT2INT, is_lambda=True, )
        cls._register_function('(**2)', func=lambda x: x ** 2, type_=INT2INT, is_lambda=True, )
        cls._register_function('(*3)', func=lambda x: x * 3, type_=INT2INT, is_lambda=True, )
        cls._register_function('(/3)', func=lambda x: x // 3, type_=INT2INT, is_lambda=True, )
        cls._register_function('(*4)', func=lambda x: x * 4, type_=INT2INT, is_lambda=True, )
        cls._register_function('(/4)', func=lambda x: x // 4, type_=INT2INT, is_lambda=True, )

        cls._register_function('(>0)', func=lambda x: x > 0, type_=INT2BOOL, is_lambda=True, )
        cls._register_function('(<0)', func=lambda x: x < 0, type_=INT2BOOL, is_lambda=True, )
        cls._register_function('(%2==0)', func=lambda x: x % 2 == 0, type_=INT2BOOL, is_lambda=True, )
        cls._register_function('(%2==1)', func=lambda x: x % 2 == 1, type_=INT2BOOL, is_lambda=True, )

        cls._register_function('(+)', func=operator.add, type_=INT2INT2INT, is_lambda=True, )
        cls._register_function('(-)', func=operator.sub, type_=INT2INT2INT, is_lambda=True, )
        cls._register_function('(*)', func=operator.mul, type_=INT2INT2INT, is_lambda=True, )
        cls._register_function('Min', func=lambda x, y: min((x, y)), type_=INT2INT2INT, is_lambda=True, )
        cls._register_function('Max', func=lambda x, y: max((x, y)), type_=INT2INT2INT, is_lambda=True, )

DSL.register_functions_lambdas()


class Variable(object):
    """The class of variable."""

    def __init__(self, name, index=None):
        self.name = name
        self.index = index

    def __str__(self):
        return self.name

    __repr__ = __str__


class Statement(object):
    """The class of statement."""

    def __init__(self, function, arg_list, ret):
        """

        :param function: int
            Function id.
        :param arg_list: list
            Arguments.
            Argument may be int (lambda id) or string (variable name)
            NOTE: String variables will be replaced into indices in program.
            String representation is used for human users.
        :param ret: str
            Return variable name.
        """
        self.ret = ret
        self.function = function
        self.arg_list = arg_list

    def var_name_to_index(self, var_index_map):
        """Transform variable name to variable index in program variables."""

        for i, arg in enumerate(self.arg_list):
            if isinstance(arg, StringTypes):
                try:
                    self.arg_list[i] = var_index_map[arg]
                except KeyError:
                    print('Unknown variable "{}"'.format(arg))
                    raise

        if isinstance(self.ret, StringTypes):
            self.ret = var_index_map[self.ret]

    def run(self, variables):
        """Run the statement, add the result to `variables`

        :param variables: Variable table
        """

        variables.append(Function.get(self.function)([
                                                         variables[arg.index] if isinstance(arg,
                                                                                            Variable) else Lambda.get(
                                                             arg)
                                                         for arg in self.arg_list
                                                         ]))

    @classmethod
    def from_string(cls, string):
        """Parse a string to a statement.

        String format:
            a = Filter (>0) x
        """

        ret, body = string.split('=')
        ret = ret.strip()

        function_and_args = body.strip().split(' ')
        function = function_and_args[0]
        args = function_and_args[1:]

        return cls(
            function=DSL.functions_lambdas[function].id,
            arg_list=[DSL.functions_lambdas[arg].id if arg in DSL.functions_lambdas else arg for arg in args],
            ret=ret,
        )

    def __str__(self):
        return '{} = {} {}'.format(
            self.ret,
            Function.get(self.function),
            ' '.join(str(arg) for arg in self.arg_list),
        )

    __repr__ = __str__


class Program(object):
    """The class of program.

    Some special variables:
    i0, i1, i2, ...: input variables
    """

    def __init__(self, input_info, statements):
        """

        :param input_info: list
            Input information.
            A list of input type ids.
        :param statements: list or str
            Statement information.
            A list of statements.

            If it is a string, it will be parsed into statements.
        """

        # List of variables.
        # [NOTE] The DSL is a SSA (Static Single Assignment) language:
        # result of each statement is a new read-only variable,
        # so variables can be put in a list.
        self.variables = [None for _ in range(len(input_info))]

        self.input_info = input_info

        if isinstance(statements, StringTypes):
            statements = [
                Statement.from_string(statement_string)
                for statement_string in statements.strip().split('\n')
                ]
        self.statements = statements

        # Transfer variable names to indices
        self.var_name_to_index()

    def var_name_to_index(self):
        var_index_map = {
            'i{}'.format(i): Variable('i{}'.format(i), i)
            for i in range(len(self.input_info))
            }

        for statement in self.statements:
            var_index_map[statement.ret] = Variable(statement.ret, len(var_index_map))

        for statement in self.statements:
            statement.var_name_to_index(var_index_map)

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
            self.variables[i] = input_data

        for statement in self.statements:
            statement.run(self.variables)

        return self.variables[-1]

    __call__ = run

    def __str__(self):
        return ''.join(
            '{}\n'.format('i{} = {}'.format(i, Type2String[input_info]))
            for i, input_info in enumerate(range(len(self.input_info)))
        ) + ''.join(
            '{}\n'.format(statement)
            for statement in self.statements
        )

    __repr__ = __str__


def dfs_program(io_pair_list):
    pass


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


if __name__ == '__main__':
    test()
