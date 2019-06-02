import network
import act
import pickle
import numpy as np


def load_data(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data[0], data[1]


def main(ld_dump=False, ld_data=True, train=True, test=True,
         batch_start=0, batch_size=100, img=None, lbl=None):
    # build network
    n = network.Network()
    n.add_layer(28 * 28, 16)
    n.add_layer(16, 16)
    n.add_layer(16, 10, act_func=act.Sigmoid())
    # read data
    if ld_data:
        print('loading data...')
        img, lbl = load_data('dump/train.dump')
    # read into matrix
    print('reading into matrix...')
    ei = []
    eo = []
    for i in range(batch_start, batch_start + batch_size):
        t = np.array(img[i]).reshape(28 * 28)
        ei.append(t / 255)
        nums = []
        for k in range(10):
            nums.append(int(lbl[i] == k))
        eo.append(nums)
    ei = np.array(ei).T
    eo = np.array(eo).reshape(batch_size, 10).T
    # load dump
    if ld_dump:
        n.load('dump/test.nw')
    # train
    if train:
        print('training (%d)...' % (batch_start))
        alpha = 0.05
        count = 5000
        n.train(ei, eo, count, alpha)
    # dump
    if train or not ld_dump:
        n.dump('dump/test.nw')
    # test
    if test:
        print('testing...')
        while True:
            try:
                i = int(input())
                img_in = np.array(img[i]).reshape(28 * 28, 1) / 255
                o = n.forward(img_in)
                print(np.argmax(o), lbl[i])
            except (KeyboardInterrupt, EOFError):
                break


def test(network_dump, img, lbl):
    # create network from dump
    n = network.Network()
    if not n.load(network_dump):
        raise RuntimeError('dump file not found')
    # run on current dataset
    total_count = len(img)
    pass_count = 0
    for i in range(total_count):
        ei = np.array(img[i]).reshape(28 * 28, 1) / 255
        eo = lbl[i]
        o = n.forward(ei)
        if np.argmax(o) == eo:
            pass_count += 1
    # print result
    rate = pass_count * 100 / total_count
    print('result: %d/%d, %.2f%%' % (pass_count, total_count, rate))


if __name__ == '__main__':
    # print('loading data...')
    # img, lbl = load_data('dump/train.dump')
    # bsize = 1000
    # for i in range(30000, 60000, bsize):
    #     main(ld_dump=True, ld_data=False, train=True, test=False,
    #          batch_start=i, batch_size=bsize, img=img, lbl=lbl)
    # main(ld_dump=True, ld_data=True, train=False, test=True)
    print('loading train dataset...')
    img_train, lbl_train = load_data('dump/train.dump')
    print('loading test dataset...')
    img_test, lbl_test = load_data('dump/test.dump')
    print('running train dataset...')
    test('dump/train-60000.nw', img_train, lbl_train)
    print('running test dataset...')
    test('dump/train-60000.nw', img_test, lbl_test)
