#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import operator
from collections import defaultdict
from types import StringTypes

import numpy as np

from .dsl_types import *
from .config import *

__author__ = 'fyabc'


class Function(object):
    """The class of function."""

    def __init__(self, name, func, arg_types, ret_type):
        self.id = None
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

    @property
    def n_args(self):
        return len(self.arg_types)


class Data(object):
    """The class of data."""

    def __init__(self, value):
        self.value = value
        self.type = get_type_id(value)

    def __str__(self):
        return 'Data({})'.format(self.value)

    __repr__ = __str__


class Lambda(object):
    """The class of lambda."""

    def __init__(self, name, func, type_):
        self.id = None
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
    name_table = {}

    # List of all functions and lambdas.
    # Map index to function or lambda.
    _function_lambda_list = []

    # List of all functions.
    # Value: function id
    functions = []

    # Table of all lambdas.
    # Key: lambda type id
    # Value: lambda id
    lambdas = defaultdict(list)

    @classmethod
    def get(cls, index):
        """Get the function or lambda by its id."""

        return cls._function_lambda_list[index]

    @classmethod
    def attribute_size(cls):
        return len(cls._function_lambda_list)

    @classmethod
    def _register_function(cls, name, **kwargs):
        is_lambda = kwargs.pop('is_lambda', False)

        if is_lambda:
            target = Lambda(name=name, **kwargs)
        else:
            target = Function(name=name, **kwargs)

        cls.name_table[name] = target
        target.id = len(cls._function_lambda_list)
        cls._function_lambda_list.append(target)

        if is_lambda:
            cls.lambdas[target.type].append(target.id)
        else:
            cls.functions.append(target.id)

    @classmethod
    def register_functions_lambdas(cls):
        """Set all functions and lambdas of the DSL."""

        # First-order functions
        cls._register_function('Head', func=lambda xs: xs[0] if len(xs) > 0 else None, arg_types=[LIST], ret_type=INT, )
        cls._register_function('Last', func=lambda xs: xs[-1] if len(xs) > 0 else None, arg_types=[LIST],
                               ret_type=INT, )
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

    @classmethod
    def auto(cls, i, prefix='v'):
        return cls('{}{}'.format(prefix, i), i)


class Statement(object):
    """The class of statement."""

    def __init__(self, function, arg_list, ret):
        """Build a statement.

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

        :param variables: Variable list
        """

        variables.append(
            DSL.get(self.function)([
                variables[arg.index] if isinstance(arg, Variable) else DSL.get(arg)
                for arg in self.arg_list
            ])
        )

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
            function=DSL.name_table[function].id,
            arg_list=[DSL.name_table[arg].id if arg in DSL.name_table else arg for arg in args],
            ret=ret,
        )

    def __str__(self):
        return '{} = {} {}'.format(
            self.ret,
            DSL.get(self.function),
            ' '.join(
                str(arg) if isinstance(arg, Variable) else str(DSL.get(arg))
                for arg in self.arg_list
            ),
        )

    __repr__ = __str__


class Program(object):
    """The class of program.

    Some special variables:
    i0, i1, i2, ...: input variables
    """

    def __init__(self, input_info, statements=None):
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
        self.statements = [] if statements is None else statements

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

    def prepare_input(self, input_list):
        """Set input variables"""

        for i, input_data in enumerate(input_list):
            if not isinstance(input_data, Data):
                input_data = Data(input_data)
            self.variables[i] = input_data

    def run_one_step(self, statement):
        """Run one statement. Used for DFS."""

        self.statements.append(statement)
        statement.run(self.variables)

    def roll_back(self):
        """Roll back one statement. Used for DFS."""

        self.statements.pop()
        self.variables.pop()

    def run(self, input_list):
        """Run the program with given input, return the output.

        :param input_list: list
            Input list.
            Each element is a `Data` object or raw data
        :return: Data
            The result.
        """

        self.prepare_input(input_list)

        for statement in self.statements:
            statement.run(self.variables)

        result = self.result

        self.variables = [None for _ in range(len(self.input_info))]

        return result

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

    def get_attribute(self):
        """Get the attribute of the program.

        :return numpy.array(shape=(function_lambda_size,), dtype=fX)
        """

        result = np.zeros(shape=(DSL.attribute_size()), dtype=fX)

        for statement in self.statements:
            result[statement.function] = 1.0
            for arg in statement.arg_list:
                if isinstance(arg, int):
                    result[arg] = 1.0

        return result

    @property
    def result(self):
        return self.variables[-1].value
