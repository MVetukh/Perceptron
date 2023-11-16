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
def sigmoid(data):
    return 1 / (1 + np.exp(-data))


#
#
def dsigmoid(data):
    return sigmoid(data) * (1 - sigmoid(data))


def lineral(data):
    return 2 * data + 2


def quadratic(data):
    return data ** 2 + 2 * data + 1


# class Network:
#     def __init__(self, sizes):
#         self.number_layers = len(sizes)
#         self.sizes = np.array(sizes)
#         self.learning_rate = 0.005
#        # self.biases = ([np.random.randn(y, 1) for y in sizes[1:]])
#         self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
#         # self.p = np.sum(np.delete(self.sizes, [0,-1]))
#         # self.beta = 0.7*(self.p)**(sizes[0])
#         # self.sum = []
#         # for j in range(len(self.weights)):
#         #   self.sumj = np.sqrt(sum(i*i for i in self.weights[j]))
#         #   self.weights[j] = (self.beta*self.weights[j])/self.sumj
#
#     def forward(self, a):
#         for weight in self.weights:
#             a = line(np.dot(weight, a))
#         return a
#
#     # def cost(self, predict, y):
#     #     return np.mean(np.sum((y - predict) ** 2))
#
#     def backprop(self, data, value):
#         #delta_bias = [np.zeros(b.shape) for b in self.biases]
#         activation = data
#         delta_weight = [np.zeros(weight.shape) for weight in self.weights]
#         activations = [activation]
#         cost = value - self.forward(activation)  # self.cost(self.forward(data), value)
#         z_in = []
#         for weight in self.weights:  # ,self.biases):
#             z = np.dot(weight, activation)
#             z_in.append(z)
#             activation = line(z)
#             activations.append(activation)
#         sigma = cost * dline(z_in[-1])
#         delta_weight[-1] = np.dot(sigma, activations[-2].T)
#         #delta_bias[-1] = sigma
#
#         for layer in range(2, self.number_layers):
#             z = z_in[-layer]
#             sigma_in = np.dot(self.weights[-layer + 1].T, sigma)
#             sigma = sigma_in * dline(z)
#             delta_weight[-layer] = np.dot(sigma, activations[-layer - 1].T)
#          #   delta_bias[-layer] = sigma
#         return delta_weight #delta_bias
#
#     def train(self, argument, value, epochs):
#         data = np.c_[argument, value]
#
#         for i in range(epochs):
#             np.random.shuffle(data)
#             for j in range(len(data)):
#                 delta_weight = self.backprop(np.array((data[j][0],data[j][1])).reshape(2,1),data[j][2].reshape(1,1))#[self.backprop(x,y) for x, y in zip(data[:, 0].reshape(len(data), 1), data[:, 1].reshape(len(data), 1))]
#                 self.weights = [weight - self.learning_rate * delta for weight, delta in
#                             zip(self.weights, delta_weight)]  # list(map(sum, zip(self.weights, delta_weight)))
#                # self.biases =[bias - self.learning_rate * delta for bias, delta in
#             #            zip(self.biases, delta_bias)]
#         return self.weights#, self.biases
#


class LinearApproximation:
    def __init__(self):
        self.weights = np.random.randn()
        self.bias = np.random.randn()

    def train(self, argument, value, learning_rate=0.1, epochs=510):
        for _ in range(epochs):
            predict = self.predict(argument)
            delta_weight = (1 / len(argument)) * np.dot(argument.T, (predict - value))
            delta_bias = (1 / len(argument)) * np.sum(predict - value)
            self.weights -= learning_rate * delta_weight
            self.bias -= learning_rate * delta_bias

    def predict(self, argument):
        return self.weights * argument + self.bias


class MLP:
    def __init__(self):
        self.Weight_1 = np.random.randn(1, 10)  # Веса первого слоя
        self.Weight_2 = np.random.randn(10, 1)  # Веса второго слоя

    def forward(self, argument):
        self.z2 = np.dot(argument, self.Weight_1)
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.Weight_2)
        predict = self.z3
        return predict

    def backpropogation(self, argument, value, predict, learning_rate):
        delta3 = (predict - value)
        delta_Weight_2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.Weight_2.T) * dsigmoid(self.z2)
        delta_Weight_1 = np.dot(argument.T, delta2)

        self.Weight_2 -= learning_rate * delta_Weight_2
        self.Weight_1 -= learning_rate * delta_Weight_1

    def train(self, argument, value, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            predict = self.forward(argument)
            self.backpropogation(argument, value, predict, learning_rate)


if __name__ == '__main__':
    x_train = np.linspace(0, 1, 100).reshape(-1, 1)
    y_train_line = lineral(x_train)
    y_train_quadratic = quadratic(x_train)

    x_test = np.linspace(100, 1, 200).reshape(-1, 1)
    y_test_line = lineral(x_test)
    y_test_quadratic = quadratic(x_test)

    Line_Aproximate = LinearApproximation()
    Line_Aproximate.train(x_train, y_train_line)
    y_predict_line = Line_Aproximate.predict(x_test)

    Math_Aproximate = MLP()
    Math_Aproximate.train(x_train, y_train_quadratic)
    y_predict_quadratic = Math_Aproximate.forward(x_test)


    plt.figure(figsize=(10, 10))
    plt.scatter(x=x_test, y=y_test_line, color='orange')
    plt.scatter(x=x_test, y=y_predict_line, color='green')

    plt.figure(figsize=(10, 10))
    plt.scatter(x=x_test, y=y_test_quadratic, color='orange')
    plt.scatter(x=x_test, y=y_predict_quadratic, color='green')

    plt.show()
