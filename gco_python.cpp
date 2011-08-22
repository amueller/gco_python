#include <cassert>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/overloads.hpp>
#include <pyublas/numpy.hpp>

#include "GCoptimization.h"

pyublas::numpy_matrix<int> cut_simple(const int& height, const int& width,
    const pyublas::numpy_matrix<double> & data_cost,
    const pyublas::numpy_matrix<double>& smoothness_cost){

    const int num_labels = smoothness_cost.size1();
    assert(data_cost.size1()==width*height);
    assert(data_cost.size2()==num_labels);
    assert(smoothness_cost.size2()==num_labels);

    boost::numeric::ublas::matrix<int> data_cost_int(num_labels, height*width);
    boost::numeric::ublas::matrix<int> smoothness_cost_int(num_labels, num_labels);

    boost::numeric::ublas::matrix<int> result(height,width);

    // rounding doubles to ints for more stable optimization
    const int precision = 100;

    for (int i=0; i<height*width*num_labels; i++){
        data_cost_int.data()[i] = precision * precision * data_cost.data()[i];
    }
    for (int i=0; i<num_labels*num_labels; i++){
        smoothness_cost_int.data()[i] = precision * smoothness_cost.data()[i];
    }
    
    GCoptimizationGridGraph gc(width, height, num_labels);
    gc.setDataCost(data_cost_int.data().begin());
    gc.setSmoothCost(smoothness_cost_int.data().begin());
    gc.expansion(5);// run expansion for 5 iterations. For swap use gc.swap(num_iterations);
    for ( int  i = 0; i < width*height; i++ ){
        result.data()[i] = gc.whatLabel(i);
    }

    return result;
}


pyublas::numpy_matrix<int>
cut_VH(const pyublas::numpy_matrix<double> & data_cost,
    const pyublas::numpy_matrix<double>& smoothness_cost,
    const pyublas::numpy_matrix<double>& V,
    const pyublas::numpy_matrix<double>& H){

    const int height = V.size1();
    const int width = V.size2();
    const int num_labels = smoothness_cost.size1();
    
    // assert consistent matrix sizes
    assert(H.size1()==height);
    assert(H.size2()==width);
    assert(data_cost.size1()==width*height);
    assert(data_cost.size2()==num_labels);
    assert(smoothness_cost.size2()==num_labels);

    // rounding doubles to ints for more stable optimization
    const int precision = 100;

    boost::numeric::ublas::matrix<int> data_cost_int(num_labels, height*width);
    boost::numeric::ublas::matrix<int> smoothness_cost_int(num_labels, num_labels);
    boost::numeric::ublas::matrix<int> V_int(height,width);
    boost::numeric::ublas::matrix<int> H_int(height,width);
    boost::numeric::ublas::matrix<int> result(height,width);
    for (int i=0; i<height*width; i++){
        V_int.data()[i] = precision * V.data()[i];
        H_int.data()[i] = precision * H.data()[i];
    }
    for (int i=0; i<height*width*num_labels; i++){
        data_cost_int.data()[i] = precision * precision * data_cost.data()[i];
    }
    for (int i=0; i<num_labels*num_labels; i++){
        smoothness_cost_int.data()[i] = precision * smoothness_cost.data()[i];
    }

    GCoptimizationGridGraph gc(width, height, num_labels);
    gc.setDataCost(data_cost_int.data().begin());
    gc.setSmoothCostVH(smoothness_cost_int.data().begin(), V_int.data().begin(), H_int.data().begin());
    gc.expansion(5);// run expansion for 5 iterations. For swap use gc.swap(num_iterations);
    for ( int  i = 0; i < width*height; i++ ){
        result.data()[i] = gc.whatLabel(i);
    }

    return result;
    }

BOOST_PYTHON_MODULE(gco_python)
    {
        boost::python::def("cut_VH", cut_VH);
        boost::python::def("cut_simple", cut_simple);
    }
