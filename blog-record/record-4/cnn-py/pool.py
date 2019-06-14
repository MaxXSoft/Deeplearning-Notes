import numpy as np
from layer import Layer


# pooling layer
class PoolLayer(Layer):
  def __init__(self, **kwargs):
    '''
    constructor of pooling layer

    Parameters
    ---
    f: int
      kernel width
    
    stride: int
      stride of pooling process
    
    mode: str
      pooling mode, 'max' or 'average'
    '''
    self.__params = kwargs
    self.__cache = None

  @staticmethod
  def __pool(mat_nhwc, **kwargs):
    # extract shapes
    n, h, w, c = mat_nhwc.shape
    # extract params
    f = kwargs.get('f', 2)
    stride = kwargs.get('stride', 2)
    mode = kwargs.get('mode', 'max')
    # compute output shape
    oh = int((h - f) / stride) + 1
    ow = int((w - f) / stride) + 1
    # initialize output
    z = np.zeros((n, oh, ow, c))
    # compute
    for i in range(n):
      for y in range(oh):
        for x in range(ow):
          for k in range(c):
            # find the corners of the current slice
            v_start = y * stride
            v_end = v_start + f
            h_start = x * stride
            h_end = h_start + f
            # get slice
            slice_nhwc = mat_nhwc[i, v_start:v_end, h_start:h_end, k]
            # get output
            if mode == 'max':
              z[i, y, x, k] = np.max(slice_nhwc)
            elif mode == 'average':
              z[i, y, x, k] = np.mean(slice_nhwc)
    assert z.shape == (n, oh, ow, c)
    # store in cache
    cache = (mat_nhwc, kwargs)
    return z, cache

  @staticmethod
  def __get_pool_mask(x):
    return x == np.max(x)

  @staticmethod
  def __distribute_value(dz, shape):
    h, w = shape
    average = dz / (h * w)
    return np.ones(shape) * average

  @staticmethod
  def __pool_back(dz, cache, mode=None):
    # extract cache
    mat, args = cache
    # extract params
    f = args.get('f', 2)
    stride = args.get('stride', 2)
    if not mode:
      mode = args.get('mode', 'max')
    # extract shapes
    n, _, _, _ = mat.shape
    n, oh, ow, c = dz.shape
    # initialize dmat
    dmat = np.zeros(mat.shape)
    # compute
    for i in range(n):
      cur_mat = mat[i]
      for y in range(oh):
        for x in range(ow):
          for k in range(c):
            # find the corners of the current slice
            v_start = y * stride
            v_end = v_start + f
            h_start = x * stride
            h_end = h_start + f
            # compute output
            cur_dz = dz[i, y, x, k]
            if mode == 'max':
              cur_slice = cur_mat[v_start:v_end, h_start:h_end, k]
              mask = PoolLayer.__get_pool_mask(cur_slice)
              dmat[i, v_start:v_end, h_start:h_end,
                   k] += np.multiply(mask, cur_dz)
            elif mode == 'average':
              dmat[i, v_start:v_end, h_start:h_end,
                   k] += PoolLayer.__distribute_value(cur_dz, (f, f))
    assert dmat.shape == mat.shape
    return dmat

  def forward(self, x):
    z, cache = self.__pool(x, **self.__params)
    self.__cache = cache
    return z

  def backprop(self, d):
    return self.__pool_back(d, self.__cache)

  def gradient(self, alpha):
    pass
