from PIL import Image
import numpy as np
import network


def read_image(file):
    img = Image.open(file).resize((28, 28)).convert('L')
    return np.array(img).reshape(28 * 28, 1) / 255


def recognize(image, net):
    img = read_image(image)
    # build network
    n = network.Network()
    n.load(net)
    # recognize
    o = n.forward(img)
    return np.argmax(o)


if __name__ == '__main__':
    print(recognize('img/max-3.png', 'dump/train-60000.nw'))
    print(recognize('img/max-6.png', 'dump/train-60000.nw'))
    print(recognize('img/max-9.png', 'dump/train-60000.nw'))
