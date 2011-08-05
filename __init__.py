import numpy as np
from scipy.weave import inline,converters
import os

def cut_VH(data_cost, smoothness_cost, V, H):
    """ datacost is num_pixels*num_labels, smoothness_cost is num_labels*num_labels, num_pizels = width*height
        it must be true that for each two labels i,j smoothness_cost(i,i) + smoothness_cost(j,j) <= smoothness_cost(i,j) + smoothness_cost(j,i)
        V are vertical costs, H horizontal costs, both the same size as the original image.
    """
    height = V.shape[0]
    width = V.shape[1]
    assert(H.shape[0]==height)
    assert(H.shape[1]==width)

    num_labels = smoothness_cost.shape[0]

    assert(data_cost.shape[1]==num_labels and smoothness_cost.shape[1]==num_labels)
    assert(data_cost.shape[0]==width*height)

    result = np.zeros((height,width),np.int32)
    data_cost = (10000 * data_cost).astype(np.int32)
    smoothness_cost = (100 * smoothness_cost).astype(np.int32)
    V = (100*V).astype(np.int32)
    H = (100*H).astype(np.int32)

    code = """
           int* data_p = (int*) &data_cost(0,0);
           int* smooth_p = (int*) &smoothness_cost(0,0);
           int* result_p = (int*) &result(0,0);
           int* V_p = (int*) &V(0,0);
           int* H_p = (int*) &H(0,0);
           GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width, height, num_labels);
           gc->setDataCost(data_p);
           gc->setSmoothCostVH(smooth_p, V_p, H_p);
           gc->expansion(5);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
           //gc->swap(2);
           for ( int  i = 0; i < width*height; i++ ){
                   result_p[i] = gc->whatLabel(i);
                   }
           delete gc;
           """
    inline(code,['data_cost','smoothness_cost','result','height','width','num_labels', 'V', 'H'],
            type_converters=converters.blitz, verbose=1, force=1,
            headers=['"GCoptimization.h"'], include_dirs=[os.path.dirname(__file__)], compiler="gcc", libraries=["gco"],
            library_dirs=[os.path.join(os.getcwd(), os.path.dirname(__file__))],
            runtime_library_dirs=[os.path.join(os.getcwd(), os.path.dirname(__file__))])
    return result

def cut_simple(height, width, data_cost, smoothness_cost):
    """ datacost is num_pixels*num_labels, smoothness_cost is num_labels*num_labels, num_pizels = width*height
        it must be true that for each two labels i,j smoothness_cost(i,i) + smoothness_cost(j,j) <= smoothness_cost(i,j) + smoothness_cost(j,i)
    """


    num_labels = smoothness_cost.shape[0]
    assert(data_cost.shape[1]==num_labels and smoothness_cost.shape[1]==num_labels)
    assert(data_cost.shape[0]==width*height)

    result = np.zeros((height, width),np.int32)
    data_cost = (100 * data_cost).astype(np.int32)
    smoothness_cost = (100 * smoothness_cost).astype(np.int32)

    code = """
           int* data_p = (int*) &data_cost(0,0);
           int* smooth_p = (int*) &smoothness_cost(0,0);
           int* result_p = (int*) &result(0,0);
           GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width, height, num_labels);
           gc->setDataCost(data_p);
           gc->setSmoothCost(smooth_p);
           gc->expansion(5);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
           //gc->swap(2);
           for ( int  i = 0; i < width*height; i++ ){
                   result_p[i] = gc->whatLabel(i);
                   }
           delete gc;
           """
    inline(code,['data_cost','smoothness_cost','result','height','width','num_labels'],
            type_converters=converters.blitz, verbose=1, force=1,
            headers=['"GCoptimization.h"'], include_dirs=[os.path.dirname(__file__)], compiler="gcc", libraries=["gco"],
            library_dirs=[os.path.join(os.getcwd(), os.path.dirname(__file__))],
            runtime_library_dirs=[os.path.join(os.getcwd(), os.path.dirname(__file__))])
    return result

def demo():
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

    cost_V = image.copy()
    cost_V[:-1,:] -= image[1:,:]
    cost_V = cost_V**2
    cost_H = image.copy()
    cost_H[:,:-1] -= image[:,1:]
    cost_H = cost_H**2

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
