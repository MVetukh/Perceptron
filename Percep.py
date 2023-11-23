# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 17:07:20 2023

@author: pdlia
"""

import numpy as np
import matplotlib.pyplot as plt
import math_func
import line_approximation
import quad_approximation


if __name__ == '__main__':
    np.random.seed(1)
    x_train = np.linspace(0, 1, 100).reshape(-1, 1)
    y_train_line = math_func.lineral(x_train)
    y_train_quadratic = math_func.quadratic(x_train)

    x_test = np.linspace(100, 1, 200).reshape(-1, 1)
    y_test_line = math_func.lineral(x_test)
    y_test_quadratic = math_func.quadratic(x_test)

    Line_Aproximate = line_approximation.LinearApproximation()
    Line_Aproximate.train(x_train, y_train_line)
    y_predict_line = Line_Aproximate.predict(x_test)

    Quad_Aproximate = quad_approximation.Quadratic_Approximation()
    lost = []
    for i in range(2500):
        loss = Quad_Aproximate.backpropogation(x_train, y_train_quadratic)
        lost.append(loss)

    plt.plot(lost)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    y_predict_quadratic = Quad_Aproximate.predict(x_test)



    plt.figure(figsize=(10, 10))
    plt.scatter(x=x_test, y=y_test_line, color='orange')
    plt.scatter(x=x_test, y=y_predict_line, color='green')

    plt.figure(figsize=(10, 10))
    #plt.scatter(x=x_test, y=y_test_quadratic, color='orange')
    plt.scatter(x=x_test, y=y_predict_quadratic, color='green')

    plt.show()
