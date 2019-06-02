import network
from compile import NNVMCompiler

import tvm
import nnvm.compiler
from tvm.contrib import graph_runtime

import numpy as np
from PIL import Image


def load_network(dump_file):
  n = network.Network()
  n.load(dump_file)
  return n

def compile_network(net):
  sym, params = net.generate(NNVMCompiler())
  print('----- compiled IR -----')
  print(sym.ir())
  print('-----------------------')
  return nnvm.compiler.build(
      sym, target='llvm', target_host='llvm', params=params,
      shape={'input': (1, 784)}, dtype={'input': 'float32'})


def read_image(file):
  img = Image.open(file).resize((28, 28)).convert('L')
  return np.array(img).reshape(28 * 28, 1) / 255


def get_runtime(graph, lib):
  return graph_runtime.create(graph, lib, tvm.cpu(0))

def run_compiled(mod, params, data, dtype='float32'):
  mod.set_input('input', tvm.nd.array(data.astype(dtype)))
  mod.set_input(**params)
  mod.run()
  tvm_output = mod.get_output(0, tvm.nd.empty(((1, 10)), dtype))
  return tvm_output.asnumpy()


if __name__ == '__main__':
  # compile network model
  n = load_network('dump/train-60000.nw')
  graph, lib, params = compile_network(n)
  # load images
  imgs = [read_image('img/max-%d.png' % (i)) for i in [3, 6, 9]]
  # get runtime
  mod = get_runtime(graph, lib)
  # run compiled module
  print()
  for img in imgs:
    print(np.argmax(run_compiled(mod, params, img)))
  # compare results between original and compiled module
  print()
  org = n.forward(imgs[0])
  comp = run_compiled(mod, params, imgs[0])
  print('deviation: %f * 1e-9' % (abs(np.average((org - comp) * 1e9))))
