import numpy as np
import math_func


class Quadratic_Approximation:
    def __init__(self, input_layer, hidden_layer, output_layer):

        self.weight1 = np.random.randn(input_layer, hidden_layer)
        self.bias1 = np.zeros((1, hidden_layer))
        self.weight2 = np.random.randn(hidden_layer, output_layer)
        self.bias2 = np.zeros((1, output_layer))

    def forward(self, argument):
        self.z2 = np.dot(argument, self.weight1)+self.bias1
        self.a2 = math_func.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.weight2)+self.bias2
        predict = math_func.sigmoid(self.z3)
        return predict

    def backpropogation(self, argument, value, learning_rate):
        F = np.dot(argument, self.weight1)  # (bs, hidden_dim)
        E = F + self.bias1  # (bs, hidden_dim)
        D = math_func.sigmoid(E)  # (bs, hidden_dim)
        C = np.dot(D, self.weight2)  # (bs, out_dim)
        B = C + self.bias2  # (bs, out_dim)
        A = value - B  # (bs, out_dim)
        L = np.power(A, 2).mean()

        dLdA = 2 * A  # (bs, out_dim)
        dAdB = -1  # (bs, out_dim)
        dBdC = np.ones_like(C)  # (bs, out_dim)
        dBdb2 = np.ones_like(self.bias2)  # (bs, out_dim)
        dCdD = self.weight2.T  # (out_dim, hidden_dim)
        dCdw2 = D.T  # (hidden_dim, bs)
        dDdE = D * (1 - D)  # (bs, hidden_dim)
        dEdF = np.ones_like(F)  # (bs, hidden_dim)
        dEdb1 = np.ones_like(self.bias1)  # (bs, hidden_dim)
        dFdw1 = argument.T  # (in_dim, bs)

        dLdb2 = np.mean(dLdA * dAdB * dBdb2, axis=0, keepdims=True)  # (1, out_dim)
        dLdw2 = np.dot(dCdw2, dLdA * dAdB * dBdC)  # (bs, out_dim)
        dLdb1 = np.mean(
            np.dot(dLdA * dAdB * dBdC, dCdD) * dDdE * dEdb1, axis=0, keepdims=True
        )  # (1, hidden_dim)
        dLdw1 = np.dot(
            dFdw1, np.dot(dLdA * dAdB * dBdC, dCdD) * dDdE * dEdF
        )  # (bs, in_dim)

        self.bias2 -= learning_rate * dLdb2
        self.weight2 -= learning_rate * dLdw2
        self.bias1 -= learning_rate * dLdb1
        self.weight1 -= learning_rate * dLdw1
        # delta3 = (predict - value)
        # delta_weight_2 = np.dot(self.a2.T, delta3)
        #
        # delta2 = np.dot(delta3, self.layer_2.T) * math_func.der_ReLu(self.z2)
        # delta_weight_1 = np.dot(argument.T, delta2)
        #
        # self.layer_2 -= learning_rate * delta_weight_2
        # self.layer_1 -= learning_rate * delta_weight_1
        return L

    def train(self, argument, value, epochs=100, learning_rate=0.1):
        for epoch in range(epochs):
            #predict = self.forward(argument)
            self.backpropogation(argument, value, learning_rate)
