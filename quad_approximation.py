import numpy as np
import math_func


class Quadratic_Approximation:
    def __init__(self):
        self.Weight_1 = np.random.randn(1, 10)  # Веса первого слоя
        self.Weight_2 = np.random.randn(10, 1)  # Веса второго слоя

    def forward(self, argument):
        self.z2 = np.dot(argument, self.Weight_1)
        self.a2 = math_func.ReLu(self.z2)
        self.z3 = np.dot(self.a2, self.Weight_2)
        predict = self.z3
        return predict

    def backpropogation(self, argument, value, predict, learning_rate):
        delta3 = (predict - value)
        delta_Weight_2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.Weight_2.T) * math_func.der_ReLu(self.z2)
        delta_Weight_1 = np.dot(argument.T, delta2)

        self.Weight_2 -= learning_rate * delta_Weight_2
        self.Weight_1 -= learning_rate * delta_Weight_1

    def train(self, argument, value, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            predict = self.forward(argument)
            self.backpropogation(argument, value, predict, learning_rate)
