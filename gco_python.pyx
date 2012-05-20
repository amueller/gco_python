import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "GCoptimization.h":
    cdef cppclass GCoptimizationGridGraph:
        GCoptimizationGridGraph(int width, int height, int n_labels)
        void setDataCost(int *)
        void setSmoothCost(int *)
        void expansion(int n_iterations)
        void swap(int n_iterations)
        void gc.setSmoothCostVH(int* pairwise, int* V, int* H)
        int whatLabel(int node)

    cdef cppclass GCoptimizationGeneralGraph:
        GCoptimizationGeneralGraph(int n_vertices, int n_labels)
        void setDataCost(int *)
        void setSmoothCost(int *)
        void setNeighbors(int, int)
        void expansion(int n_iterations)
        void swap(int n_iterations)
        int whatLabel(int node)


def cut_simple(np.ndarray[np.int32_t, ndim=3, mode='c'] data_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] smoothness_cost, n_iter=5,
        algorithm='expansion'):

    if data_cost.shape[2] != smoothness_cost.shape[0]:
        raise ValueError("data_cost and smoothness_cost have incompatible shapes.\n"
            "data_cost must be height x width x n_labels, smoothness_cost must be n_labels x n_labels.\n"
            "Got: data_cost: (%d, %d, %d), smoothness_cost: (%d, %d)"
            %(data_cost.shape[0], data_cost.shape[1], data_cost.shape[2],
                smoothness_cost.shape[0], smoothness_cost.shape[1]))
    if smoothness_cost.shape[1] != smoothness_cost.shape[0]:
        raise ValueError("smoothness_cost must be a square matrix.")
    cdef int h = data_cost.shape[1]
    cdef int w = data_cost.shape[0]
    cdef int n_labels = smoothness_cost.shape[0]

    cdef GCoptimizationGridGraph* gc = new GCoptimizationGridGraph(h, w, n_labels)
    gc.setDataCost(<int*>data_cost.data)
    gc.setSmoothCost(<int*>smoothness_cost.data)
    if algorithm == 'swap':
        gc.swap(n_iter)
    elif algorithm == 'expansion':
        gc.expansion(n_iter)

    cdef np.npy_intp result_shape[2]
    result_shape[0] = w
    result_shape[1] = h
    cdef np.ndarray[np.int32_t, ndim=2] result = np.PyArray_SimpleNew(2, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(w * h):
        result_ptr[i] = gc.whatLabel(i)
    return result


def cut_from_graph(np.ndarray[np.int32_t, ndim=2, mode='c'] edges,
        np.ndarray[np.int32_t, ndim=2, mode='c'] data_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] smoothness_cost, n_iter=5,
        algorithm='expansion'):

    if data_cost.shape[1] != smoothness_cost.shape[0]:
        raise ValueError("data_cost and smoothness_cost have incompatible shapes.\n"
            "data_cost must be height x width x n_labels, smoothness_cost must be n_labels x n_labels.\n"
            "Got: data_cost: (%d, %d), smoothness_cost: (%d, %d)"
            %(data_cost.shape[0], data_cost.shape[1],
                smoothness_cost.shape[0], smoothness_cost.shape[1]))
    if smoothness_cost.shape[1] != smoothness_cost.shape[0]:
        raise ValueError("smoothness_cost must be a square matrix.")
    cdef int n_vertices = data_cost.shape[0]
    cdef int n_labels = smoothness_cost.shape[0]

    cdef GCoptimizationGeneralGraph* gc = new GCoptimizationGeneralGraph(n_vertices, n_labels)
    for e in edges:
        gc.setNeighbors(e[0], e[1])
    gc.setDataCost(<int*>data_cost.data)
    gc.setSmoothCost(<int*>smoothness_cost.data)
    if algorithm == 'swap':
        gc.swap(n_iter)
    elif algorithm == 'expansion':
        gc.expansion(n_iter)

    cdef np.npy_intp result_shape[1]
    result_shape[0] = n_vertices
    cdef np.ndarray[np.int32_t, ndim=1] result = np.PyArray_SimpleNew(1, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(n_vertices):
        result_ptr[i] = gc.whatLabel(i)
    return result
