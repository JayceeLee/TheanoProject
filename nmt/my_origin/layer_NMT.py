#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import numpy as np
import theano
import theano.tensor as T

from utils import NMTConfig as ParamConfig
from utils_NMT import fX, p_, normal_weight, orthogonal_weight

__author__ = 'fyabc'


# Function of layers and their initializers.
# [NOTE]
# Layer
# The 1st argument (in common) is input_: a Theano tensor that represent the input.
# The 2nd argument (in common) is params: the dict of (tensor) parameters.
# The 3rd argument (in common) is prefix: the prefix layer name.
#
# Initializer
# The 1st argument (in common) is params: the dict of (numpy) parameters.
# The 2nd argument (in common) is prefix: the prefix layer name.
# Optional argument n_in: input size.
# Optional argument n_out: output size.
# Optional argument dim: dimension size (hidden size?).
#
# These functions return another Theano tensor as its output.

def dropout(input_, use_noise, rand):
    return T.switch(
        use_noise,
        input_ * rand.binomial(input_.shape, p=0.5, n=1, dtype=fX),
        input_ * 0.5
    )


# feed-forward layer: affine transformation + point-wise nonlinearity
def feed_forward(input_, params, prefix='rconv', activation=T.tanh, **kwargs):
    return activation(T.dot(input_, params[p_(prefix, 'W')]) + params[p_(prefix, 'b')])


def init_feed_forward(params, prefix='ff', n_in=None, n_out=None, orthogonal=True):
    n_in = ParamConfig['dim_proj'] if n_in is None else n_in
    n_out = ParamConfig['dim_proj'] if n_out is None else n_out

    params[p_(prefix, 'W')] = normal_weight(n_in, n_out, scale=0.01, orthogonal=orthogonal)
    params[p_(prefix, 'b')] = np.zeros((n_out,), dtype=fX)

    return params


# GRU layer
def gru(input_, params, prefix='gru', mask=None, **kwargs):
    n_steps = input_.shape[0]
    n_samples = input_.shape[1] if input_.ndim == 3 else 1

    dim = params[p_(prefix, 'Ux')].shape[1]

    mask = T.alloc(1., n_steps, 1) if mask is None else mask

    # utility function to slice a tensor
    def _slice(_x, n, _dim):
        if _x.ndim == 3:
            return _x[:, :, n * _dim:(n + 1) * _dim]
        return _x[:, n * _dim:(n + 1) * _dim]

    # input_ is the input word embeddings
    # input to the gates, concatenated
    input_ = T.dot(input_, params[p_(prefix, 'W')]) + params[p_(prefix, 'b')]

    # input to compute the hidden state proposal
    input_x = T.dot(input_, params[p_(prefix, 'Wx')]) + params[p_(prefix, 'bx')]

    # step function to be used by scan
    # args   : sequences             | outputs | non-seqs
    def _step(_mask, _input, _input_x, _hidden, _U, _Ux):
        p_react = T.dot(_hidden, _U) + _input

        # reset and update gates
        r = T.nnet.sigmoid(_slice(p_react, 0, dim))
        u = T.nnet.sigmoid(_slice(p_react, 1, dim))

        # compute the hidden state proposal
        p_react_x = T.dot(_hidden, _Ux) * r + _input_x

        # hidden state proposal
        h = T.tanh(p_react_x)

        # leaky integrate and obtain next hidden state
        h = u * _hidden + (1. - u) * h
        h = _mask[:, None] * h + (1. - _mask)[:, None] * _hidden

        return h

    # prepare scan arguments
    seqs = [mask, input_, input_x]
    init_states = [T.alloc(0., n_samples, dim)]
    shared_vars = [params[p_(prefix, 'U')], params[p_(prefix, 'Ux')]]

    result, _ = theano.scan(
        _step,
        sequences=seqs,
        outputs_info=init_states,
        non_sequences=shared_vars,
        name=p_(prefix, '_layers'),
        n_steps=n_steps,
        profile=False,
        strict=True,
    )

    return result


def init_gru(params, prefix='gru', n_in=None, dim=None):
    n_in = ParamConfig['dim_proj'] if n_in is None else n_in
    dim = ParamConfig['dim_proj'] if dim is None else dim

    # embedding to gates transformation weights, biases
    params[p_(prefix, 'W')] = np.concatenate([normal_weight(n_in, dim), normal_weight(n_in, dim)], axis=1)
    params[p_(prefix, 'b')] = np.zeros((2 * dim,), dtype=fX)

    # recurrent transformation weights for gates
    params[p_(prefix, 'U')] = np.concatenate([orthogonal_weight(dim), orthogonal_weight(dim)], axis=1)

    # embedding to hidden state proposal weights, biases
    params[p_(prefix, 'Wx')] = normal_weight(n_in, dim)
    params[p_(prefix, 'bx')] = np.zeros((dim,), dtype=fX)

    # recurrent transformation weights for hidden state proposal
    params[p_(prefix, 'Ux')] = orthogonal_weight(dim)

    return params


# Conditional GRU layer with Attention
def gru_cond(input_, params, prefix='gru_cond', mask=None, **kwargs):
    context = kwargs.pop('context', None)
    one_step = kwargs.pop('one_step', False)
    init_memory = kwargs.pop('init_memory', None)
    init_state = kwargs.pop('init_state', None)
    context_mask = kwargs.pop('context_mask', None)

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    n_steps = input_.shape[0]

    # todo
