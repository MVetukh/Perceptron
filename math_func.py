import numpy as np


def line(data):
    return np.array(data)


def dline(data):
    return np.ones_like(data[0])


def sigmoid(data):
    return 1 / (1 + np.exp(-data))


def dsigmoid(data):
    return sigmoid(data) * (1 - sigmoid(data))


def ReLu(x):
    return np.maximum(0.0, x)


def der_ReLu(x):
    r = np.where(x > 0, 1, 0)
    return r


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def dtanh(x):
    return 1 - tanh(x) * tanh(x)


def lineral(data):
    return 2 * data + 2


def quadratic(data):
    return  data ** 2 + 2*data + 1
