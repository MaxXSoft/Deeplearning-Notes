from struct import pack, unpack


def parse_idx1(file):
  with open(file, 'rb') as f:
    # check header
    header, count = unpack('>LL', f.read(8))
    if header != 0x00000801:
      raise RuntimeError('invalid idx1 file')
    # read labels
    labels = []
    for _ in range(count):
      labels.append(unpack('>B', f.read(1))[0])
    return labels


def parse_idx3(file):
  with open(file, 'rb') as f:
    # check header
    header, count, w, h = unpack('>LLLL', f.read(16))
    if header != 0x00000803:
      raise RuntimeError('invalid idx1 file')
    # read images
    images = []
    for _ in range(count):
      img = []
      for _ in range(h):
        l = []
        for _ in range(w):
          l.append(unpack('>B', f.read(1))[0])
        img.append(l)
      images.append(img)
    return images