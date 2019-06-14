import pickle


class Network:
  def __init__(self):
    self.__layers = []

  def add_layer(self, l):
    self.__layers.append(l)

  def forward(self, x):
    a = x
    for i in self.__layers:
      a = i.forward(a)
    return a

  def __backprop(self, out, expect):
    # logistic regression
    f = out * (1 - out) + 1e-10
    d = (out - expect) / f
    for i in reversed(self.__layers):
      d = i.backprop(d)

  def __gradient(self, alpha):
    for i in self.__layers:
      i.gradient(alpha)

  def train(self, expect_in, expect_out, count, alpha):
    for i in range(count):
      print('%5d/%5d\r' % (i + 1, count), end='')
      out = self.forward(expect_in)
      self.__backprop(out, expect_out)
      self.__gradient(alpha)
    print()

  def dump(self, file):
    with open(file, 'wb') as f:
      pickle.dump(self.__layers, f)

  def load(self, file):
    with open(file, 'rb') as f:
      self.__layers = pickle.load(f)


if __name__ == '__main__':
  import numpy as np
  import conv
  import act
  import pool
  import dense
  # build
  net = Network()
  net.add_layer(conv.ConvLayer(28, 28, 1, 3, 10))
  net.add_layer(act.ReLULayer())
  net.add_layer(pool.PoolLayer(f=2, stride=2))
  net.add_layer(act.ReLULayer())
  net.add_layer(conv.ConvLayer(13, 13, 10, 4, 16))
  net.add_layer(act.ReLULayer())
  net.add_layer(pool.PoolLayer(f=2, stride=2))
  net.add_layer(act.ReLULayer())
  net.add_layer(conv.ConvLayer(5, 5, 16, 2, 10))
  net.add_layer(act.ReLULayer())
  net.add_layer(pool.PoolLayer(f=4, stride=1))
  net.add_layer(dense.DenseLayer(10, 10))
  net.add_layer(act.SigmoidLayer())
  # train
  i = np.random.randn(1, 28, 28, 1)
  o = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).reshape(10, 1)
  net.train(i, o, 100, 0.1)
  print(np.argmax(net.forward(i)) + 1)
