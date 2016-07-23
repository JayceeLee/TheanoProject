#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

from keras.layers import Input, Dense
from keras.models import Model

__author__ = 'fyabc'


def callableLayers():
    """
    A layer instance is callable (on a tensor), and it returns a tensor
    Input tensor(s) and output tensor(s) can then be used to define a Model
    Such a model can be trained just like Keras Sequential models
    """

    # This returns a tensor
    inputs = Input(shape=(784,))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    # this creates a model that includes the Input layer and 3 Dense layers
    model = Model(input=inputs, output=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorial_crossentropy',
                  metrics=['accuracy'])

    # model.fit(data, labels)


def callableModels():
    pass


if __name__ == '__main__':
    pass
