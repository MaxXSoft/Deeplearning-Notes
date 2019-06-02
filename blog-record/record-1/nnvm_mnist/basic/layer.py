import numpy as np
import act


class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backprop(self, d):
        raise NotImplementedError

    def gradient(self, alpha):
        raise NotImplementedError
    
    def compile(self, comp, data):
        raise NotImplementedError


class BasicLayer(Layer):
    def __init__(self, input_count, node_count, act_func=None):
        self.__w = np.random.randn(node_count, input_count) * 0.01
        self.__b = np.zeros((node_count, 1))
        self.__act = act_func if act_func else act.ReLU()
        self.__a = None
        self.__z = None
        self.__dw = None
        self.__db = None

    def forward(self, x):
        self.__a = x
        self.__z = np.dot(self.__w, x) + self.__b
        a = self.__act.func(self.__z)
        return a

    def backprop(self, da):
        dz = da * self.__act.dfunc(self.__z)
        m = da.shape[1]
        self.__dw = np.dot(dz, self.__a.T) / m
        self.__db = np.sum(dz, axis=1, keepdims=True) / m
        da = np.dot(self.__w.T, dz)
        return da

    def gradient(self, alpha):
        self.__w -= alpha * self.__dw
        self.__b -= alpha * self.__db
    
    def compile(self, comp, data):
        w = comp.new_const(self.__w)
        b = comp.new_const(self.__b.reshape(self.__b.shape[0]))
        z = comp.compile(op='dense', data=data, weight=w, bias=b,
                         units=self.__w.shape[0])
        return self.__act.compile(comp, z)


if __name__ == '__main__':
    import loss, random
    # generate train and test dataset
    train_i, train_o, test_i, test_o = [], [], [], []
    for _ in range(900):
        r = random.random()
        train_i.append(r)
        train_o.append(int(r > 0.5))
    train_i = np.array(train_i).reshape(1, len(train_i))
    train_o = np.array(train_o).reshape(1, len(train_o))
    for _ in range(100):
        r = random.random()
        test_i.append(r)
        test_o.append(int(r > 0.5))
    test_i = np.array(test_i).reshape(1, len(test_i))
    test_o = np.array(test_o).reshape(1, len(test_o))
    alp = 0.05
    # layer
    l0 = BasicLayer(1, 10)
    l1 = BasicLayer(10, 1, act_func=act.Sigmoid())
    for _ in range(1000):
        o = l0.forward(train_i)
        o = l1.forward(o)
        da = loss.Logistic().dfunc(o, train_o)
        da = l1.backprop(da)
        l0.backprop(da)
        l1.gradient(alp)
        l0.gradient(alp)
    # check
    o = l0.forward(train_i)
    o = l1.forward(o)
    rate_train = np.sum(train_o == np.round(o)) / train_o.shape[1]
    print('train dataset: %.2f%%' % (rate_train * 100))
    o = l0.forward(test_i)
    o = l1.forward(o)
    rate_test = np.sum(test_o == np.round(o)) / test_o.shape[1]
    print('test dataset: %.2f%%' % (rate_test * 100))
