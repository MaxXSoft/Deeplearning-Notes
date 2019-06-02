import tvm
import nnvm
import nnvm.symbol as sym
import numpy as np


class GraphCompiler:
  def __init__(self):
    self._params = {}

  def new_var(self, name):
    raise NotImplementedError
  
  def new_const(self, value):
    name = '__param_%d' % (len(self._params))
    self._params[name] = value
    return self.new_var(name)

  def compile(self, **kwargs):
    raise NotImplementedError

  def generate(self, output):
    raise NotImplementedError


class NNVMCompiler(GraphCompiler):
  def new_var(self, name):
    return sym.Variable(name)

  def compile(self, **kwargs):
    if kwargs['op'] == 'dense':
      return sym.dense(data=kwargs['data'], weight=kwargs['weight'],
                       bias=kwargs['bias'], units=kwargs['units'])
    elif kwargs['op'] == 'relu':
      return sym.relu(data=kwargs['data'])
    elif kwargs['op'] == 'leaky_relu':
      return sym.leaky_relu(data=kwargs['data'], alpha=kwargs['alpha'])
    elif kwargs['op'] == 'sigmoid':
      return sym.sigmoid(data=kwargs['data'])
    else:
      raise RuntimeError('invalid operator')

  def generate(self, output):
    sym = nnvm.graph.create(output)
    params = {k: tvm.nd.array(np.array(v, dtype=np.float32))
              for k, v in self._params.items()}
    return sym, params
