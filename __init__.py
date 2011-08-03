import numpy as np
from scipy.weave import inline,converters
import os

def cut(width, height, data_cost, smoothness_cost):
    """ datacost is num_pixels*num_labels, smoothness_cost is num_labels*num_labels, num_pizels = width*height
        it must be true that for each two labels i,j smoothness_cost(i,i) + smoothness_cost(j,j) <= smoothness_cost(i,j) + smoothness_cost(j,i)
    """


    num_labels = smoothness_cost.shape[0]
    assert(data_cost.shape[1]==num_labels and smoothness_cost.shape[1]==num_labels)
    assert(data_cost.shape[0]==width*height)

    result = np.zeros((width,height),np.int32)
    data_cost = (100 * data_cost).astype(np.int32)
    smoothness_cost = (100 * smoothness_cost).astype(np.int32)

    code = """
           int* data_p = (int*) &data_cost(0,0);
           int* smooth_p = (int*) &smoothness_cost(0,0);
           int* result_p = (int*) &result(0,0);
           GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width, height, num_labels);
           gc->setDataCost(data_p);
           gc->setSmoothCost(smooth_p);
           std::cout << "Before optimization energy is " << gc->compute_energy()<<std::endl;
           gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
           //gc->swap(2);
           std::cout << "After optimization energy is " << gc->compute_energy()<<std::endl;
           for ( int  i = 0; i < width*height; i++ ){
                   result_p[i] = gc->whatLabel(i);
                   }
           delete gc;
           """
    inline(code,['data_cost','smoothness_cost','result','width','height','num_labels'],
            type_converters=converters.blitz, verbose=0, force=0,
            headers=['"GCoptimization.h"'], include_dirs=[os.path.dirname(__file__)], compiler="gcc", libraries=["gco"], 
            library_dirs=[os.path.join(os.getcwd(), os.path.dirname(__file__))],
            runtime_library_dirs=[os.path.join(os.getcwd(), os.path.dirname(__file__))])
    return result

def demo():
    image = np.zeros((10, 20))
    image[4:8, 4:11] = 1
    original_image = image.copy()
    image = image + np.random.normal(scale=0.3,size=image.shape)
    image = np.maximum(image,np.zeros(image.shape))
    image = np.minimum(image,np.ones(image.shape))

    data_cost = np.vstack([image.ravel(), 1-image.ravel()]).T.copy("C")
    #data_cost = np.vstack([np.zeros_like(image.ravel()), np.ones_like(image.ravel())]).T.copy("C")
    print(data_cost.shape)
    beta = 0.3
    smoothness_cost = np.array([[0, beta],[beta, 0]])

    segmentation = cut(image.shape[0], image.shape[1], data_cost, smoothness_cost)
    import matplotlib.pyplot as plt
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
