# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 17:07:20 2023

@author: pdlia
"""

import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


def tanh(data):
    return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))


def dtanh(data):
    return 1 - tanh(data) * tanh(data)


def line_active(data):
    return data #1/(1+np.exp(-data))


def dline(data):
    return  1 #line_active(data)*(1-line_active(data))#1


def lineral(data):
    return np.array(5 * data + 5)


def quadratic(data):
    return data ** 2 + 2 * data + 1


class Network:
    def __init__(self, sizes):
        self.number_layers = len(sizes)
        self.sizes = np.array(sizes)
        # self.biases = ([np.random.randn(y, 1) for y in sizes[1:]])
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # self.p = np.sum(np.delete(self.sizes, [0,-1]))
        # self.beta = 0.7*(self.p)**(sizes[0])
        # self.sum = []
        # for j in range(len(self.weights)):
        #   self.sumj = np.sqrt(sum(i*i for i in self.weights[j]))
        #   self.weights[j] = (self.beta*self.weights[j])/self.sumj

    def forward(self, a):
        for weight in self.weights:
            a = line_active(np.dot(weight, a))
        return a

    def backprop(self, data, value):
        # delta_bias = [np.zeros(b.shape) for b in self.biases]
        delta_weight = [np.zeros(weight.shape) for weight in self.weights]
        cost = (value - self.forward(data))
        z_in = []
        activation = data
        activations = [data]
        for weight in self.weights:
            z = np.dot(weight, activation)
            z_in.append(z)
            activation = line_active(z)
            activations.append(activation)
        sigma = cost * dline(z_in[-1])
        delta_weight[-1] = 0.001 * np.dot(sigma, activations[-2].T)
        # delta_bias[-1] = 0.1 * sigma

        for l in range(2, self.number_layers):
            z = z_in[-l]
            sigma_in = np.dot(self.weights[-l], sigma)
            sigma = sigma_in * dline(z)
            delta_weight[-l] = 0.001 * np.dot(sigma, activations[-l - 1].T)
        # delta_bias[-l] = 0.1 * sigm
        return delta_weight

    def train(self, argument, value, epochs):
        data = np.vstack((argument, value)).T

        for i in range(epochs):
            np.random.shuffle(data)
            delta_weight = self.backprop(data[:, 0].reshape(len(data), 1), data[:, 1].reshape(len(data), 1))
            self.weights = list(map(sum, zip(self.weights, delta_weight)))
            # self.biases = list(map(sum, zip(self.biases, delta_bias)))
        return self.weights


if __name__ == '__main__':
    x = np.array(range(-100, 100, 2))
    x = (x - x.mean()) / x.std()
    xtest = np.array(range(-95, 105, 2))
    xtest = (xtest - xtest.mean()) / xtest.std()
    y = lineral(x)
    ytest = lineral(xtest)

    net = Network([100, 10, 100])
    net.train(x, y, 6)
    predict = net.forward(xtest)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.scatter(x=xtest, y=ytest, color='orange')
    ax.scatter(x=xtest, y=predict, color='green')
    plt.show()
