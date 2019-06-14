import numpy as np


def pool(mat_nhwc, **kwargs):
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


def get_pool_mask(x):
  return x == np.max(x)


def distribute_value(dz, shape):
  h, w = shape
  average = dz / (h * w)
  return np.ones(shape) * average


def pool_back(dz, cache, mode=None):
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
            mask = get_pool_mask(cur_slice)
            dmat[i, v_start:v_end, h_start:h_end,
                 k] += np.multiply(mask, cur_dz)
          elif mode == 'average':
            dmat[i, v_start:v_end, h_start:h_end,
                 k] += distribute_value(cur_dz, (f, f))
  assert dmat.shape == mat.shape
  return dmat


if __name__ == '__main__':
  np.random.seed(1)
  A_prev = np.random.randn(5, 5, 3, 2)
  args = {"stride": 1, "f": 2}
  A, cache = pool(A_prev, **args)
  dA = np.random.randn(5, 4, 2, 2)

  dA_prev = pool_back(dA, cache, mode="max")
  print("mode = max")
  print('mean of dA = ', np.mean(dA))
  print('dA_prev[1,1] = ', dA_prev[1,1])  
  print()
  dA_prev = pool_back(dA, cache, mode="average")
  print("mode = average")
  print('mean of dA = ', np.mean(dA))
  print('dA_prev[1,1] = ', dA_prev[1,1]) 
