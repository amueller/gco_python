import pyublas
from gco_python import *

    #""" datacost is num_pixels*num_labels, smoothness_cost is num_labels*num_labels, num_pizels = width*height
        #it must be true that for each two labels i,j smoothness_cost(i,i) + smoothness_cost(j,j) <= smoothness_cost(i,j) + smoothness_cost(j,i)
        #V are vertical costs, H horizontal costs, both the same size as the original image.
    #"""


#def cut_simple(height, width, data_cost, smoothness_cost):
    #""" datacost is num_pixels*num_labels, smoothness_cost is num_labels*num_labels, num_pizels = width*height
        #it must be true that for each two labels i,j smoothness_cost(i,i) + smoothness_cost(j,j) <= smoothness_cost(i,j) + smoothness_cost(j,i)
    #"""

def demo():
    import numpy as np
    image = np.zeros((20, 40))
    image[0:4, 4:11] = 1
    original_image = image.copy()
    image = image + np.random.normal(scale=0.6,size=image.shape)
    image = np.maximum(image,np.zeros(image.shape))
    image = np.minimum(image,np.ones(image.shape))

    data_cost = np.vstack([image.ravel(), 1-image.ravel()]).T.copy("C")
    print(data_cost.shape)
    beta = 0.6
    smoothness_cost = np.array([[0, beta],[beta, 0]])

    import matplotlib.pyplot as plt

    #cost_V = image.copy()
    #cost_V[:-1,:] -= image[1:,:]
    #cost_V = cost_V**2
    #cost_H = image.copy()
    #cost_H[:,:-1] -= image[:,1:]
    #cost_H = cost_H**2

    #cost_V = np.ones_like(image)
    #cost_H = np.ones_like(image)

    segmentation = cut_simple(image.shape[0], image.shape[1], data_cost, smoothness_cost)
    #segmentation = cut_VH(data_cost, smoothness_cost, cost_V, cost_H)
    plt.figure()
    plt.subplot(3,1,1)
    plt.imshow(segmentation,interpolation="nearest")
    plt.colorbar()
    plt.subplot(3,1,2)
    plt.imshow((10*image).astype(np.int).reshape(original_image.shape),interpolation="nearest")
    plt.colorbar()
    plt.subplot(3,1,3)
    plt.imshow(original_image,interpolation="nearest")
    plt.colorbar()
    plt.show()

if __name__=="__main__":
    demo()
