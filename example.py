import matplotlib.pyplot as plt
import numpy as np
#from _gco_python import cut_simple, cut_from_graph
from pygco import cut_simple


def example_binary():
    # generate trivial data
    x = np.ones((10, 10))
    x[:, 5:] = -1
    x_noisy = x + np.random.normal(0, 0.8, size=x.shape)
    x_thresh = x_noisy > .0

    # create unaries
    unaries = x_noisy
    # as we convert to int, we need to multipy to get sensible values
    unaries = (10 * np.dstack([unaries, -unaries]).copy("C")).astype(np.int32)
    # create potts pairwise
    pairwise = -10 * np.eye(2, dtype=np.int32)
    # do simple cut
    result = cut_simple(unaries, pairwise)

    # plot results
    plt.subplot(231, title="original")
    plt.imshow(x, interpolation='nearest')
    plt.subplot(232, title="noisy version")
    plt.imshow(x_noisy, interpolation='nearest')
    plt.subplot(233, title="rounded to integers")
    plt.imshow(unaries[:, :, 0], interpolation='nearest')
    plt.subplot(234, title="thresholding result")
    plt.imshow(x_thresh, interpolation='nearest')
    plt.subplot(235, title="cut_simple")
    plt.imshow(result, interpolation='nearest')

    #inds = np.arange(x.size).reshape(x.shape)
    #horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    #vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    #edges = np.vstack([horz, vert])
    #result = cut_from_graph(edges.tolist(), unaries, 1.1 * np.eye(2))

    #result_mrf = cut_from_graph(edges.tolist(), unaries,  np.eye(2) * 1.1)
    #plt.matshow(result_mrf.reshape(x.shape), interpolation='nearest')
    plt.show()


#def example_multinomial():
    #np.random.seed(45)
    #unaries = np.zeros((10, 12, 3))
    #unaries[:, :4, 0] = 1
    #unaries[:, 4:8, 1] = 1
    #unaries[:, 8:, 2] = 1
    #x = np.argmax(unaries, axis=2)
    #unaries_noisy = unaries + np.random.normal(size=unaries.shape)
    #x_thresh = np.argmax(unaries_noisy, axis=2)
    #unaries_noisy = unaries_noisy.reshape(-1, 3)

    #inds = np.arange(x.size).reshape(x.shape)
    #horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    #vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    #edges = np.vstack([horz, vert])
    #result = cut_from_graph(unaries_noisy, edges, 1.1 * np.ones(3))
    #binaries = np.eye(3) + np.ones((1, 1))
    #binaries[-1, 0] = 0
    #binaries[0, -1] = 0
    #print(binaries)
    #result_mrf = cut_from_graph(unaries_noisy, edges, binaries)
    #plt.subplot(141)
    #plt.imshow(x, interpolation="nearest")
    #plt.subplot(142)
    #plt.imshow(x_thresh, interpolation="nearest")
    #plt.subplot(143)
    #plt.imshow(result.reshape(x.shape), interpolation="nearest")
    #plt.subplot(144)
    #plt.imshow(result_mrf.reshape(x.shape), interpolation="nearest")
    #plt.show()


example_binary()
#example_multinomial()
