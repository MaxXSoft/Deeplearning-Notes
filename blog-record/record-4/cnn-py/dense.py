import numpy as np
from layer import Layer


# dense layer
class DenseLayer(Layer):
  def __init__(self, in_count, out_count):
    '''
    constructor of dense layer

    Parameters
    ---
    in_count: int
      input node count
    
    out_count: int
      output node count
    '''
    self.__w = np.random.randn(out_count, in_count) * 0.01
    self.__b = np.zeros((out_count, 1))
    self.__a = None
    self.__dw = None
    self.__db = None

  def forward(self, x):
    self.__a = x
    return np.dot(self.__w, x) + self.__b

  def backprop(self, d):
    m = d.shape[1]
    self.__dw = np.dot(d, self.__a.T) / m
    self.__db = np.sum(d, axis=1, keepdims=True) / m
    return np.dot(self.__w.T, d)

  def gradient(self, alpha):
    self.__w -= alpha * self.__dw
    self.__b -= alpha * self.__db
