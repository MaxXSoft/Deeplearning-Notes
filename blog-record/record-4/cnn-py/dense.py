import numpy as np
from layer import Layer


class DenseLayer(Layer):
  '''
  dense layer

  Parameters
  ---
  in_count: int
    input node count
  
  out_count: int
    output node count
  '''

  def __init__(self, in_count, out_count):
    self.__w = np.random.randn(out_count, in_count) * 0.01
    self.__b = np.zeros((out_count, 1))
    self.__ishape = None
    self.__a = None
    self.__dw = None
    self.__db = None

  def forward(self, x):
    self.__ishape = x.shape
    self.__a = x.reshape(self.__w.shape[1], -1)
    return np.dot(self.__w, self.__a) + self.__b

  def backprop(self, d):
    m = d.shape[1]
    self.__dw = np.dot(d, self.__a.T) / m
    self.__db = np.sum(d, axis=1, keepdims=True) / m
    return np.dot(self.__w.T, d).reshape(self.__ishape)

  def gradient(self, alpha):
    self.__w -= alpha * self.__dw
    self.__b -= alpha * self.__db
