#include <cassert>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/exception_translator.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <exception>
#include <pyublas/numpy.hpp>


#include "GCoptimization.h"

using namespace boost;

pyublas::numpy_matrix<int> cut_simple(const int& height, const int& width,
    const pyublas::numpy_matrix<double> & data_cost,
    const pyublas::numpy_matrix<double>& smoothness_cost){

    const int num_labels = smoothness_cost.size1();
    assert(data_cost.size1()==width*height);
    assert(data_cost.size2()==num_labels);
    assert(smoothness_cost.size2()==num_labels);

    boost::numeric::ublas::matrix<int> data_cost_int(height*width, num_labels);
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

    boost::numeric::ublas::matrix<int> data_cost_int(height*width, num_labels);
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

pyublas::numpy_vector<int>
cut_from_graph_weighted(const boost::python::list & graph,
    const pyublas::numpy_matrix<double>& data_cost,
    const pyublas::numpy_matrix<double>& smoothness_cost){
    const int num_labels = smoothness_cost.size1();
    const int num_vertices = data_cost.size1();
    
    if (smoothness_cost.size2() == num_labels)
        throw std::runtime_error("Smoothnes cost size wrong");
    if (data_cost.size2()==num_labels)
        throw std::runtime_error("data cost size wrong");

    // rounding doubles to ints for more stable optimization
    boost::numeric::ublas::matrix<int> smoothness_cost_int(num_labels, num_labels);
    boost::numeric::ublas::vector<int> result(num_vertices);
    boost::numeric::ublas::matrix<int> data_cost_int(num_vertices, num_labels);

    for (int i=0; i<num_vertices*num_labels; i++){
        data_cost_int.data()[i] = std::ceil(data_cost.data()[i]);
    }
    for (int i=0; i<num_labels*num_labels; i++){
        smoothness_cost_int.data()[i] = std::ceil(smoothness_cost.data()[i]);
    }

    GCoptimizationGeneralGraph gc(num_vertices, num_labels);
    for(python::ssize_t i=0;i<len(graph);i++) {
        gc.setNeighbors(python::extract<int>(python::list(graph[i])[0]), python::extract<int>(python::list(graph[i])[1]), python::extract<int>(python::list(graph[i])[2]));
    }
    gc.setDataCost(data_cost_int.data().begin());
    gc.setSmoothCost(smoothness_cost_int.data().begin());
    gc.swap(5);// run expansion for 5 iterations. For swap use gc.swap(num_iterations);
    for ( int  i = 0; i < num_vertices; i++ ){
        result.data()[i] = gc.whatLabel(i);
    }

    return result;
}

pyublas::numpy_vector<int>
cut_from_graph(const boost::python::list & graph,
    const pyublas::numpy_matrix<double>& data_cost,
    const pyublas::numpy_matrix<double>& smoothness_cost){
    const int num_labels = smoothness_cost.size1();
    const int num_vertices = data_cost.size1();

    assert(smoothness_cost.size2() == num_labels);
    assert(data_cost.size2()==num_labels);

    // rounding doubles to ints for more stable optimization
    boost::numeric::ublas::matrix<int> smoothness_cost_int(num_labels, num_labels);
    boost::numeric::ublas::vector<int> result(num_vertices);
    boost::numeric::ublas::matrix<int> data_cost_int(num_vertices, num_labels);

    for (int i=0; i<num_vertices*num_labels; i++){
        data_cost_int.data()[i] = std::ceil(data_cost.data()[i]);
    }
    for (int i=0; i<num_labels*num_labels; i++){
        smoothness_cost_int.data()[i] = std::ceil(smoothness_cost.data()[i]);
    }

    GCoptimizationGeneralGraph gc(num_vertices, num_labels);
    for(python::ssize_t i=0;i<len(graph);i++) {
        gc.setNeighbors(python::extract<int>(python::list(graph[i])[0]), python::extract<int>(python::list(graph[i])[1]));
    }
    gc.setDataCost(data_cost_int.data().begin());
    gc.setSmoothCost(smoothness_cost_int.data().begin());
    gc.swap(5);// run expansion for 5 iterations. For swap use gc.swap(num_iterations);
    for ( int  i = 0; i < num_vertices; i++ ){
        result.data()[i] = gc.whatLabel(i);
    }

    return result;
}

boost::python::list
cut_from_segments(const pyublas::numpy_vector<unsigned char> & masks,
    const pyublas::numpy_matrix<double>& data_cost,
    const pyublas::numpy_matrix<double>& smoothness_cost, double fg_bias){
    const size_t ndims = masks.ndim();
    assert(ndims==3);

    const npy_intp * const dims = masks.dims();
    const int num_masks = dims[2];
    const int height = dims[0];
    const int width = dims[1];
    const int num_labels = smoothness_cost.size1();
    assert(smoothness_cost.size2() == num_labels);
    assert(data_cost.size1()==num_masks);
    assert(data_cost.size2()==num_labels);

    GCoptimizationGeneralGraph gc(width*height + num_masks, num_labels);

    // rounding doubles to ints for more stable optimization
    const int precision = 100;
    boost::numeric::ublas::matrix<int> smoothness_cost_int(num_labels, num_labels);
    boost::numeric::ublas::matrix<int> result_pixels(height, width);
    boost::numeric::ublas::vector<int> result_segments(num_masks);
    boost::numeric::ublas::matrix<int> data_cost_int((num_masks + width*height), num_labels);
   
    // data cost for pixels constant, for segments given by data_cost 
    for (int l=0; l<num_labels; l++){
        for (int i=0; i<width*height; i++){
            data_cost_int(i, l) = precision * precision * (l==0? 1:fg_bias);
        }
        for (int i=0; i<num_masks; i++){
            data_cost_int(i+width*height, l) = precision * precision * data_cost(i,l);
        }
    }
    // smoothness cost!
    for (int i=0; i<num_labels*num_labels; i++){
        smoothness_cost_int.data()[i] =  precision * smoothness_cost.data()[i];
    }

    // iterate over all pixels and over all segments for neighborhoods
    for (int i = 0; i < height; i++){
      for (int j = 0; j < width; j++) {
          // pixel grid graph
          //if (i > 0)
              //gc.setNeighbors(i*width+j,(i-1)*width+j);
          //if (j > 0)
              //gc.setNeighbors(i*width+j,i*width+j-1);
          // segments nodes:
          for (int s = 0; s < num_masks; s++){
              if (masks(num_masks*(width * i + j) + s)>0){
                  gc.setNeighbors(i*width+j, width*height + s);
              }
          }
      }
    }

    gc.setDataCost(data_cost_int.data().begin());
    gc.setSmoothCost(smoothness_cost_int.data().begin());
    std::cout << "data energy:" << gc.giveDataEnergy() << std::endl;
    gc.expansion(2);// run expansion for 5 iterations. For swap use gc.swap(num_iterations);
    std::cout << "data energy after optimization:" << gc.giveDataEnergy() << std::endl;
    for (int  i = 0; i < height; i++ )
        for (int j = 0; j < width; j++)
              result_pixels(i, j) = gc.whatLabel(i*width+j);
          //result_pixels.data()[i] = data_cost_int(1,i);
    
    for ( int  i = 0; i < num_masks; i++ ){
          result_segments[i] = gc.whatLabel(height*width + i);
          //result_segments[i] = data_cost_int(0, width*height + i);
    }
    boost::python::list results;
    results.append(result_pixels);
    results.append(result_segments);
    return results;

}

//struct GCException;

void exception_translator(const GCException& x) {
  PyErr_SetString(PyExc_UserWarning, x.message);
}


BOOST_PYTHON_MODULE(_gco_python)
    {
        boost::python::register_exception_translator<GCException>(&exception_translator);
        boost::python::def("cut_VH", cut_VH);
        boost::python::def("cut_simple", cut_simple);
        boost::python::def("cut_from_graph", cut_from_graph);
        boost::python::def("cut_from_graph_weighted", cut_from_graph_weighted);
        boost::python::def("cut_from_segments", cut_from_segments);
    }
