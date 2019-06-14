import numpy as np


def conv_pad(mat_nhwc, pad):
  dim = ((0, 0), (pad, pad), (pad, pad), (0, 0))
  return np.pad(mat_nhwc, dim, 'constant', constant_values=0)


def conv_step(slice_ffc, kernel_ffc, bias):
  ans = np.multiply(slice_ffc, kernel_ffc) + bias
  return np.sum(ans)


def conv(mat_nhwc, kernel_ffck, bias_111k, **kwargs):
  # extract shapes
  n, h, w, c_m = mat_nhwc.shape
  f, f, c_k, kc = kernel_ffck.shape
  assert c_m == c_k
  # extract params
  stride = kwargs.get('stride', 1)
  pad = kwargs.get('pad', 0)
  # compute output shape
  oh = int((h - f + 2 * pad) / stride) + 1
  ow = int((w - f + 2 * pad) / stride) + 1
  # initialize output
  z = np.zeros((n, oh, ow, kc))
  mat_pads = conv_pad(mat_nhwc, pad)
  # compute
  for i in range(n):
    mat_pad = mat_pads[i]
    for y in range(oh):
      for x in range(ow):
        for k in range(kc):
          # find the corners of the current slice
          v_start = y * stride
          v_end = v_start + f
          h_start = x * stride
          h_end = h_start + f
          # get slice
          slice_ffc = mat_pad[v_start:v_end, h_start:h_end, :]
          # get output
          cur_kernel = kernel_ffck[..., k]
          cur_bias = bias_111k[..., k]
          z[i, y, x, k] = conv_step(slice_ffc, cur_kernel, cur_bias)
  assert z.shape == (n, oh, ow, kc)
  # save in cache
  cache = (mat_nhwc, kernel_ffck, bias_111k, kwargs)
  return z, cache


def conv_back(dz, cache):
  # extract cache
  mat, kernel, _, args = cache
  # extract shapes of mat
  n, h, w, c = mat.shape
  # extract shapes of kernel
  f, f, c, kc = kernel.shape
  # extract params
  stride = args.get('stride', 1)
  pad = args.get('pad', 0)
  # extract shapes of dz
  n, oh, ow, kc = dz.shape
  # initialize output
  dmat = np.zeros((n, h, w, c))
  dw = np.zeros((f, f, c, kc))
  db = np.zeros((1, 1, 1, kc))
  # pad mat and dmat
  mat_pads = conv_pad(mat, pad)
  dm_pads = conv_pad(dmat, pad)
  # compute
  for i in range(n):
    mat_pad = mat_pads[i]
    dm_pad = dm_pads[i]
    for y in range(oh):
      for x in range(ow):
        for k in range(kc):
          # find the corners of the current slice
          v_start = y * stride
          v_end = v_start + f
          h_start = x * stride
          h_end = h_start + f
          # get slice
          mat_slice = mat_pad[v_start:v_end, h_start:h_end, :]
          # update gradients
          cur_dz = dz[i, y, x, k]
          dm_pad[v_start:v_end, h_start:h_end, :] += kernel[..., k] * cur_dz
          dw[..., c] += mat_slice * cur_dz
          db[..., c] += cur_dz
    dmat[i, ...] = dm_pad[pad:-pad, pad:-pad, :]
  assert dmat.shape == (n, h, w, c)
  return dmat, dw, db


if __name__ == '__main__':
  np.random.seed(1)
  mat = np.random.randn(10, 4, 4, 3)
  kernel = np.random.randn(2, 2, 3, 8)
  bias = np.random.randn(1, 1, 1, 8)
  args = {"pad": 2, "stride": 1}
  Z, cache_conv = conv(mat, kernel, bias, **args)
  print("Z's mean =", np.mean(Z))
  print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
  np.random.seed(1)
  dA, dW, db = conv_back(Z, cache_conv)
  print("dA_mean =", np.mean(dA))
  print("dW_mean =", np.mean(dW))
  print("db_mean =", np.mean(db))
