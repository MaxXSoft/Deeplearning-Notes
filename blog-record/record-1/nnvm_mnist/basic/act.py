import numpy as np


class Activation:
    def func(self, x):
        raise NotImplementedError

    def dfunc(self, x):
        raise NotImplementedError
    
    def compile(self, comp, data):
        raise NotImplementedError


class ReLU(Activation):
    def __init__(self, leaky=0):
        self.__leaky = leaky

    def func(self, x):
        return np.maximum(x, self.__leaky * x)

    def dfunc(self, x):
        return np.where(x > 0, 1, self.__leaky)
    
    def compile(self, comp, data):
        if self.__leaky:
            return comp.compile(op='leaky_relu', data=data,
                                alpha=self.__leaky)
        else:
            return comp.compile(op='relu', data=data)


class Sigmoid(Activation):
    def func(self, x):
        return 1 / (1 + np.exp(-x))

    def dfunc(self, x):
        s = self.func(x)
        return s * (1 - s)
    
    def compile(self, comp, data):
        return comp.compile(op='sigmoid', data=data)
