# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 17:07:20 2023

@author: pdlia
"""

import numpy as np
import matplotlib.pyplot as plt


def line(data):
    return np.array(data)


def dline(data):
    return np.ones_like(data[0])

#
# def tanh(data):
#     return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))
#
#
# def dtanh(data):
#     return 1 - tanh(data) * tanh(data)


def lineral(data):
    return np.array(2 * data + 1)


def quadratic(data):
    return data ** 2 + 2 * data + 1


class Network:
    def __init__(self, sizes):
        self.number_layers = len(sizes)
        self.sizes = np.array(sizes)
        self.learning_rate = 0.001
        #  self.biases = ([np.random.randn(y, 1) for y in sizes[1:]])
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # self.p = np.sum(np.delete(self.sizes, [0,-1]))
        # self.beta = 0.7*(self.p)**(sizes[0])
        # self.sum = []
        # for j in range(len(self.weights)):
        #   self.sumj = np.sqrt(sum(i*i for i in self.weights[j]))
        #   self.weights[j] = (self.beta*self.weights[j])/self.sumj

    def forward(self, a):
        for weight in self.weights:
            a = line(np.dot(weight, a))
        return a

    # def cost(self, predict, y):
    #     return np.mean(np.sum((y - predict) ** 2))

    def backprop(self, data, value):
        # delta_bias = [np.zeros(b.shape) for b in self.biases]
        activation = data
        delta_weight = [np.zeros(weight.shape) for weight in self.weights]
        activations = [activation]
        cost = value - self.forward(activation)  # self.cost(self.forward(data), value)
        z_in = []
        for weight in self.weights:  # ,self.biases):
            z = np.dot(weight, activation)
            z_in.append(z)
            activation = line(z)
            activations.append(activation)
        sigma = cost * dline(z_in[-1])
        delta_weight[-1] = np.dot(sigma, activations[-2].T)
        # delta_bias[-1] = 0.1 * sigma

        for layer in range(2, self.number_layers):
            z = z_in[-layer]
            sigma_in = np.dot(self.weights[-layer + 1].T, sigma)
            sigma = sigma_in * dline(z)
            delta_weight[-layer] = np.dot(sigma, activations[-layer - 1].T)
        #  delta_bias[-l] = 0.1 * sigma
        return delta_weight  # ,delta_bias

    def train(self, argument, value, epochs):
        data = np.vstack((argument, value)).T

        for i in range(epochs):
            np.random.shuffle(data)
            for j in range(len(data)):
                delta_weight = self.backprop(data[j][0],data[j][1])#[self.backprop(x,y) for x, y in zip(data[:, 0].reshape(len(data), 1), data[:, 1].reshape(len(data), 1))]
        self.weights = [weight - self.learning_rate * delta for weight, delta in
                            zip(self.weights, delta_weight)]  # list(map(sum, zip(self.weights, delta_weight)))
        # self.biases = list(map(sum, zip(self.biases, delta_bias)))
        return self.weights


if __name__ == '__main__':
    x = np.array(range(-100, 100, 2))
    x = (x.max() - x) / (x.max() - x.min())
    xtest = np.array(range(-98, 102, 2))
    xtest = (xtest.max() - xtest) / (xtest.max() - xtest.min())
    y = lineral(x)
    ytest = lineral(xtest)

    net = Network([1, 100, 1])
    net.train(x, y, 500)
    predict = []
    for i in range(100):
        predict.append(net.forward(xtest[i].reshape(1,1)))
    print(predict)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.scatter(x=xtest, y=ytest, color='orange')
    ax.scatter(x=xtest, y=predict, color='green')
    plt.show()
