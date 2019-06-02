import numpy as np


class Loss:
    def func(self, out, expect):
        raise NotImplementedError

    def dfunc(self, out, expect):
        raise NotImplementedError


class Logistic(Loss):
    def func(self, out, expect):
        return -(expect * np.log(out) + (1 - expect) * np.log(1 - out))

    def dfunc(self, out, expect):
        f = out * (1 - out) + 1e-10
        return (out - expect) / f
