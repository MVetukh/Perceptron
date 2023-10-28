# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 17:07:20 2023

@author: pdlia
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def df(x):
    return 1 - f(x)*f(x)

def lineral(x):
  return 2*x-3

def quadratic(x):
  return x**2+2*x+1


class Network(object):
  def __init__(self,sizes):
    self.number_layers = len(sizes)
    self.sizes = np.array(sizes)
    self.biases = ([np.random.randn(y, 1) for y in sizes[1:]])
    self.weights =([np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])])
    # self.p = np.sum(np.delete(self.sizes, [0,-1]))
    # self.beta = 0.7*(self.p)**(sizes[0])
    # self.sum = []
    # for j in range(len(self.weights)):
    #   self.sumj = np.sqrt(sum(i*i for i in self.weights[j]))
    #   self.weights[j] = (self.beta*self.weights[j])/self.sumj
  def feedforward(self, a):
     for b, w in zip(self.biases, self.weights):
        a = f(np.dot(w, a)+b)
     return a

  def backprop(self,x,true):
    delta_b = [np.zeros(b.shape) for b in self.biases]
    delta_w = [np.zeros(w.shape) for w in self.weights]
    cost = (true - self.feedforward(x))
    z_in = []
    activation = x
    activations = [x]
    for b, w in zip(self.biases, self.weights):
      z = np.dot(w, activation)+b
      z_in.append(z)
      activation = f(z)
      activations.append(activation)
    sigma = cost*df(z_in[-1])
    delta_w[-1] = 0.1*np.dot(sigma,activations[-2].T)
    delta_b[-1] = 0.1*sigma

    for l in range(2,self.number_layers):
      z = z_in[-l]
      sigma_in = np.dot(self.weights[-l],sigma)
      sigma = sigma_in*df(z)
      delta_w[-l] = 0.1*np.dot(sigma,activations[-l+1].T)
      delta_b[-l] = 0.1*sigma
    return delta_w,delta_b


  def train(self,x,true):
      t = true
      epochs = len(true)
      for i in range(epochs):
        dw,db = self.backprop(x[i][0],t[i][0])
        self.weights = list(map(sum, zip(self.weights,dw)))
        self.biases = list(map(sum, zip(self.biases,db)))
      return self.weights, self.biases


x = np.array(range(-100,100,2))
x = (x - x.min())/(x.max() - x.min())
yl = lineral(x)
x = x.reshape(100,1)
yl = yl.reshape(100,1)


net = Network([1,4,1])
net.train(x,yl)


xtest = np.array(range(-50,150,2))
xtest = (xtest - xtest.min())/(xtest.max() - xtest.min())
pred = []
for i in range(len(xtest)):
    p = net.feedforward(xtest[i])
    pred.append(p)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()




ax.scatter(x = xtest,y = lineral(xtest), color='orange')
ax.scatter(x = xtest, y = pred,  color='green')


plt.show()