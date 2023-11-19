import numpy as np

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

