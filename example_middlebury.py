import Image
import matplotlib.pyplot as plt
import numpy as np
from gco import cut_simple
from daicrf import potts_mrf


def stereo_unaries(img1, img2):
    differences = []
    max_disp = 8
    for disp in np.arange(max_disp):
        if disp == 0:
            diff = np.sum((img1 - img2) ** 2, axis=2)
        else:
            diff = np.sum((img1[:, 2 * disp:, :] - img2[:, :-2 * disp, :]) **
                    2, axis=2)
        if disp != max_disp - 1:
            diff = diff[:, max_disp - disp - 1:disp - max_disp + 1]
        differences.append(diff)
    return np.dstack(differences)


def potts_example():
    img1 = np.asarray(Image.open("scene1.row3.col1.ppm")) / 255.
    img2 = np.asarray(Image.open("scene1.row3.col2.ppm")) / 255.
    unaries = stereo_unaries(img1, img2)
    n_disps = unaries.shape[2]

    newshape = unaries.shape[:2]
    potts_cut = cut_simple(unaries.shape[0], unaries.shape[1],
            unaries.reshape(-1, n_disps), -5 * np.eye(n_disps))
    x, y = np.ogrid[:n_disps, :n_disps]
    one_d_topology = np.abs(x - y).astype(np.float)

    one_d_cut = cut_simple(unaries.shape[0], unaries.shape[1],
            unaries.reshape(-1, n_disps), 5 * one_d_topology)
    # build edges for max product inference:
    inds = np.arange(np.prod(newshape)).reshape(newshape)
    horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
    vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
    edges = np.vstack([horz, vert])
    max_product = potts_mrf(np.exp(unaries.reshape(-1, n_disps)), edges, 10)
    plt.subplot(231)
    plt.imshow(img1)
    plt.subplot(232)
    plt.imshow(img1)
    plt.subplot(233)
    plt.imshow(np.argmax(unaries, axis=2), interpolation='nearest')
    plt.subplot(234)
    plt.imshow(potts_cut.reshape(newshape), interpolation='nearest')
    plt.subplot(235)
    plt.imshow(one_d_cut.reshape(newshape), interpolation='nearest')
    plt.subplot(236)
    plt.imshow(max_product.reshape(newshape), interpolation='nearest')
    plt.show()

potts_example()
