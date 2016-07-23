#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation


__author__ = 'fyabc'


def first():
    model = Sequential()

    model.add(Dense(32, init='uniform', input_dim=784))
    # model.add(Dense(32, input_shape=(784,)))              # equals
    # model.add(Dense(32, batch_input_shape=(None, 784)))   # equals

    model.add(Activation('relu'))

    # model.add(LSTM(32, input_shape=(10, 64)))
    # model.add(LSTM(32, batch_input_shape=(None, 10, 64))) # equals
    # model.add(LSTM(32, input_length=10, input_dim=64))    # equals


def mergeLayer():
    from keras.layers import Merge

    leftBranch = Sequential()
    leftBranch.add(Dense(32, input_dim=784))

    rightBranch = Sequential()
    rightBranch.add(Dense(32, input_dim=784))

    merged = Merge([leftBranch, rightBranch], mode='concat')
    # Merge modes: sum(default, element-wise), concat, mul, ave(tensor average), dot, cos
    # or other modes: mode=lambda x, y: x - y

    finalModel = Sequential()
    finalModel.add(merged)
    finalModel.add(Dense(10, activation='softmax'))

    # metrics: 度量标准
    # for multi-class classification problems
    finalModel.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    # # for binary classification problems
    # model.compile(optimizer='rmsprop', loss=
    # # for a mean squared error regression problem
    # model.compile(optimizer='rmsprop', loss='mse')

    # finalModel.fit([inputData1, inputData2], targets)


def training():
    model = Sequential()
    model.add(Dense(1, input_dim=784, activation='softmax'))

    print('Compiling the model...', end='')
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print('done')

    data = np.random.random((1000, 784))
    labels = np.random.randint(2, size=(1000, 1))

    print('Fitting the model...', end='')
    model.fit(data, labels, nb_epoch=10, batch_size=32, verbose=1)
    print('done')

    # print(model.get_weights())

if __name__ == '__main__':
    training()
    pass
