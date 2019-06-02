import pickle
import layer
import loss

class Network:
    def __init__(self, loss_func=None):
        self.__layers = []
        self.__loss = loss_func if loss_func else loss.Logistic()

    def add_layer(self, input_count, node_count, act_func=None):
        l = layer.BasicLayer(input_count, node_count, act_func)
        self.__layers.append(l)

    def forward(self, x):
        a = x
        for i in self.__layers:
            a = i.forward(a)
        return a

    def backprop(self, out, expect):
        da = self.__loss.dfunc(out, expect)
        for i in reversed(self.__layers):
            da = i.backprop(da)
        return da

    def gradient(self, alpha):
        for i in self.__layers:
            i.gradient(alpha)

    def train(self, expect_in, expect_out, count, alpha):
        for _ in range(count):
            out = self.forward(expect_in)
            self.backprop(out, expect_out)
            self.gradient(alpha)

    def dump(self, file):
        with open(file, 'wb') as f:
            data = (self.__layers, self.__loss)
            pickle.dump(data, f)

    def load(self, file):
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                self.__layers, self.__loss = data
            return True
        except FileNotFoundError:
            print('warning: file "%s" not found, ignored' % (file))
            return False

    def generate(self, comp):
        var = comp.new_var('input')
        for i in self.__layers:
            var = i.compile(comp, var)
        return comp.generate(var)


if __name__ == '__main__':
    import numpy as np
    import random
    import act
    # generate train & test dataset
    train_i, train_o, test_i, test_o = [], [], [], []
    for _ in range(900):
        r1 = random.randint(1, 10)
        r2 = random.randint(1, 10)
        train_i.append([r1, r2])
        train_o.append([int(r1 + r2 > 12), int(r1 + r2 < 7)])
    train_i = np.array(train_i).reshape(len(train_i), 2).T
    train_o = np.array(train_o).reshape(len(train_o), 2).T
    for _ in range(100):
        r1 = random.randint(1, 10)
        r2 = random.randint(1, 10)
        test_i.append([r1, r2])
        test_o.append([int(r1 + r2 > 12), int(r1 + r2 < 7)])
    test_i = np.array(test_i).reshape(len(test_i), 2).T
    test_o = np.array(test_o).reshape(len(test_o), 2).T
    # build network
    n = Network()
    n.add_layer(2, 10)
    n.add_layer(10, 10)
    n.add_layer(10, 2, act_func=act.Sigmoid())
    # train
    n.train(train_i, train_o, 5000, 0.05)
    # test
    print('train:')
    o = n.forward(train_i)
    print(np.sum(train_o[0] == np.round(o[0])) / 900)
    print(np.sum(train_o[1] == np.round(o[1])) / 900)
    print('test:')
    o = n.forward(test_i)
    print(np.sum(test_o[0] == np.round(o[0])) / 100)
    print(np.sum(test_o[1] == np.round(o[1])) / 100)
    print('test out:')
    print(test_o)
    print(np.round(o))
