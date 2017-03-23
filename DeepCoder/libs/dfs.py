#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from dsl import Program, Variable, Statement, DSL
from dsl_types import get_type_id, IsLambda

__author__ = 'fyabc'


def _gen_function_statement(function, current_args, variables, used_variables, options, max_depth, depth=1):
    """Generate all available statements with given functions and variables."""

    if depth > max_depth:
        yield Statement(
            function=function,
            arg_list=current_args[:],
            ret=used_variables[len(variables)],
        )
        return

    arg_type = DSL.get(function).arg_types[depth - 1]

    if IsLambda[arg_type]:
        for lambda_ in options.get('lambda_order', DSL.lambdas)[arg_type]:
            current_args.append(lambda_)
            for statement in _gen_function_statement(function, current_args, variables, used_variables,
                                                     options, max_depth, depth + 1):
                yield statement
            current_args.pop()
    else:
        for i, data in enumerate(variables):
            if data.type == arg_type:
                current_args.append(used_variables[i])
                for statement in _gen_function_statement(function, current_args, variables, used_variables,
                                                         options, max_depth, depth + 1):
                    yield statement
                current_args.pop()


def gen_statement(variables, used_variables, options):
    """Generate all available statements with given variables."""

    for function in options.get('function_order', DSL.functions):
        n_args = DSL.get(function).n_args

        for statement in _gen_function_statement(function, [], variables, used_variables, options, n_args):
            yield statement


def _dfs(programs, depth, io_pair_list, used_variables, options):
    if depth > options['max_depth']:
        return False

    for statement in gen_statement(programs[0].variables, used_variables, options):
        for program in programs:
            program.run_one_step(statement)
        for i, program in enumerate(programs):
            if program.result != io_pair_list[i][1]:
                # A test failed, need more search
                break
        else:
            # Pass all tests, find a solution
            return True

        # Some tests failed, need more search
        result = _dfs(programs, depth + 1, io_pair_list, used_variables, options)
        if result:
            return True

        # Restore the programs
        for program in programs:
            program.roll_back()

    return False


def dfs_program(io_pair_list, **kwargs):
    """DFS to search the program.

    :param io_pair_list: list
        Each element is a tuple of (input, output)
            input is a list of raw data
            output is raw data
    :param kwargs:
    :return:
    """

    max_depth = kwargs.pop('max_depth', 5)

    # Number of examples
    m = len(io_pair_list)

    used_variables = [Variable.auto(i, 'i') for i in range(len(io_pair_list[0][0]))] + \
        [Variable.auto(i + len(io_pair_list[0][0])) for i in range(max_depth)]

    programs = [
        Program(
            input_info=[get_type_id(data) for data in io_pair_list[0][0]],
            statements=[],
        )
        for _ in range(m)
    ]

    for i, program in enumerate(programs):
        program.prepare_input(io_pair_list[i][0])

    options = {
        'max_depth': max_depth,
        'function_order': DSL.functions,
        'lambda_order': DSL.lambdas,
    }

    result = _dfs(programs, 1, io_pair_list, used_variables, options)

    if result:
        return programs[0]
    else:
        return None


def test():
    result = dfs_program(
        io_pair_list=[
            ([2, [3, 5, 4, 7, 5]], 7),
            ([3, [5, 4, 3, 10, 8, 11]], 12),
            ([3, [1, 2, 2, 9, 19]], 5),
        ],
        max_depth=3,
    )

    print(result)


if __name__ == '__main__':
    test()
