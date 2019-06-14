import numpy as np
from layer import Layer


class SigmoidLayer(Layer):
  def __init__(self):
    self.__x = None

  def forward(self, x):
    self.__x = x
    return 1 / (1 + np.exp(-x))

  def backprop(self, d):
    s = 1 / (1 + np.exp(np.negative(self.__x)))
    return d * s * (1 - s)

  def gradient(self, alpha):
    pass


class ReLULayer(Layer):
  def __init__(self, leaky=0):
    self.__leaky = leaky
    self.__x = None

  def forward(self, x):
    self.__x = x
    return np.maximum(x, self.__leaky * x)

  def backprop(self, d):
    return d * np.where(self.__x > 0, 1, self.__leaky)

  def gradient(self, alpha):
    pass
