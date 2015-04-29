import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "GCoptimization.h":
    cdef cppclass GCoptimizationGridGraph:
        GCoptimizationGridGraph(int width, int height, int n_labels) except +
        void setDataCost(int *) except +
        void setSmoothCost(int *) except +
        void expansion(int n_iterations) except +
        void swap(int n_iterations) except +
        void setSmoothCostVH(int* pairwise, int* V, int* H) except +
        int whatLabel(int node) except +

    cdef cppclass GCoptimizationGeneralGraph:
        GCoptimizationGeneralGraph(int n_vertices, int n_labels) except +
        void setDataCost(int *) except +
        void setSmoothCost(int *) except +
        void setNeighbors(int, int) except +
        void setNeighbors(int, int, int) except +
        void expansion(int n_iterations) except +
        void swap(int n_iterations) except +
        int whatLabel(int node) except +


def cut_simple(np.ndarray[np.int32_t, ndim=3, mode='c'] unary_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] pairwise_cost, n_iter=5,
        algorithm='expansion'):
    """
    Apply multi-label graphcuts to grid graph.

    Parameters
    ----------
    unary_cost: ndarray, int32, shape=(width, height, n_labels)
        Unary potentials
    pairwise_cost: ndarray, int32, shape=(n_labels, n_labels)
        Pairwise potentials for label compatibility
    n_iter: int, (default=5)
        Number of iterations
    algorithm: string, `expansion` or `swap`, default=expansion
        Whether to perform alpha-expansion or alpha-beta-swaps.
    """

    if unary_cost.shape[2] != pairwise_cost.shape[0]:
        raise ValueError("unary_cost and pairwise_cost have incompatible shapes.\n"
            "unary_cost must be height x width x n_labels, pairwise_cost must be n_labels x n_labels.\n"
            "Got: unary_cost: (%d, %d, %d), pairwise_cost: (%d, %d)"
            %(unary_cost.shape[0], unary_cost.shape[1], unary_cost.shape[2],
                pairwise_cost.shape[0], pairwise_cost.shape[1]))
    if pairwise_cost.shape[1] != pairwise_cost.shape[0]:
        raise ValueError("pairwise_cost must be a square matrix.")
    cdef int h = unary_cost.shape[1]
    cdef int w = unary_cost.shape[0]
    cdef int n_labels = pairwise_cost.shape[0]
    if (pairwise_cost != pairwise_cost.T).any():
        raise ValueError("pairwise_cost must be symmetric.")

    cdef GCoptimizationGridGraph* gc = new GCoptimizationGridGraph(h, w, n_labels)
    gc.setDataCost(<int*>unary_cost.data)
    gc.setSmoothCost(<int*>pairwise_cost.data)
    if algorithm == 'swap':
        gc.swap(n_iter)
    elif algorithm == 'expansion':
        gc.expansion(n_iter)
    else:
        raise ValueError("algorithm should be either `swap` or `expansion`. Got: %s" % algorithm)

    cdef np.npy_intp result_shape[2]
    result_shape[0] = w
    result_shape[1] = h
    cdef np.ndarray[np.int32_t, ndim=2] result = np.PyArray_SimpleNew(2, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(w * h):
        result_ptr[i] = gc.whatLabel(i)

    del gc
    return result

def cut_simple_vh(np.ndarray[np.int32_t, ndim=3, mode='c'] unary_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] pairwise_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] costV,
        np.ndarray[np.int32_t, ndim=2, mode='c'] costH, 
        n_iter=5,
        algorithm='expansion'):
    """
    Apply multi-label graphcuts to grid graph.

    Parameters
    ----------
    unary_cost: ndarray, int32, shape=(width, height, n_labels)
        Unary potentials
    pairwise_cost: ndarray, int32, shape=(n_labels, n_labels)
        Pairwise potentials for label compatibility
    costV: ndarray, int32, shape=(width, height)
        Vertical edge weights
    costH: ndarray, int32, shape=(width, height)
        Horizontal edge weights
    n_iter: int, (default=5)
        Number of iterations
    algorithm: string, `expansion` or `swap`, default=expansion
        Whether to perform alpha-expansion or alpha-beta-swaps.
    """

    if unary_cost.shape[2] != pairwise_cost.shape[0]:
        raise ValueError("unary_cost and pairwise_cost have incompatible shapes.\n"
            "unary_cost must be height x width x n_labels, pairwise_cost must be n_labels x n_labels.\n"
            "Got: unary_cost: (%d, %d, %d), pairwise_cost: (%d, %d)"
            %(unary_cost.shape[0], unary_cost.shape[1], unary_cost.shape[2],
                pairwise_cost.shape[0], pairwise_cost.shape[1]))
    if pairwise_cost.shape[1] != pairwise_cost.shape[0]:
        raise ValueError("pairwise_cost must be a square matrix.")
    cdef int h = unary_cost.shape[1]
    cdef int w = unary_cost.shape[0]
    cdef int n_labels = pairwise_cost.shape[0]
    if (pairwise_cost != pairwise_cost.T).any():
        raise ValueError("pairwise_cost must be symmetric.")
    if costV.shape[0] != w or costH.shape[0] != w or costV.shape[1] != h or costH.shape[1] != h:
        raise ValueError("incorrect costV or costH dimensions.")

    cdef GCoptimizationGridGraph* gc = new GCoptimizationGridGraph(h, w, n_labels)
    gc.setDataCost(<int*>unary_cost.data)
    gc.setSmoothCostVH(<int*>pairwise_cost.data, <int*>costV.data, <int*>costH.data)
    if algorithm == 'swap':
        gc.swap(n_iter)
    elif algorithm == 'expansion':
        gc.expansion(n_iter)
    else:
        raise ValueError("algorithm should be either `swap` or `expansion`. Got: %s" % algorithm)

    cdef np.npy_intp result_shape[2]
    result_shape[0] = w
    result_shape[1] = h
    cdef np.ndarray[np.int32_t, ndim=2] result = np.PyArray_SimpleNew(2, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(w * h):
        result_ptr[i] = gc.whatLabel(i)
    del gc
    return result


def cut_from_graph(np.ndarray[np.int32_t, ndim=2, mode='c'] edges,
        np.ndarray[np.int32_t, ndim=2, mode='c'] unary_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] pairwise_cost, n_iter=5,
        algorithm='expansion'):
    """
    Apply multi-label graphcuts to arbitrary graph given by `edges`.

    Parameters
    ----------
    edges: ndarray, int32, shape(n_edges, 2 or 3)
        Rows correspond to edges in graph, given as vertex indices.
        if edges is n_edges x 3 then third parameter is used as edge weight
    unary_cost: ndarray, int32, shape=(n_vertices, n_labels)
        Unary potentials
    pairwise_cost: ndarray, int32, shape=(n_labels, n_labels)
        Pairwise potentials for label compatibility
    n_iter: int, (default=5)
        Number of iterations
    algorithm: string, `expansion` or `swap`, default=expansion
        Whether to perform alpha-expansion or alpha-beta-swaps.
    """
    if (pairwise_cost != pairwise_cost.T).any():
        raise ValueError("pairwise_cost must be symmetric.")

    if unary_cost.shape[1] != pairwise_cost.shape[0]:
        raise ValueError("unary_cost and pairwise_cost have incompatible shapes.\n"
            "unary_cost must be height x width x n_labels, pairwise_cost must be n_labels x n_labels.\n"
            "Got: unary_cost: (%d, %d), pairwise_cost: (%d, %d)"
            %(unary_cost.shape[0], unary_cost.shape[1],
                pairwise_cost.shape[0], pairwise_cost.shape[1]))
    if pairwise_cost.shape[1] != pairwise_cost.shape[0]:
        raise ValueError("pairwise_cost must be a square matrix.")
    cdef int n_vertices = unary_cost.shape[0]
    cdef int n_labels = pairwise_cost.shape[0]

    cdef GCoptimizationGeneralGraph* gc = new GCoptimizationGeneralGraph(n_vertices, n_labels)
    for e in edges:
        if e.shape[0] == 3:
            gc.setNeighbors(e[0], e[1], e[2])
        else:
            gc.setNeighbors(e[0], e[1])
    gc.setDataCost(<int*>unary_cost.data)
    gc.setSmoothCost(<int*>pairwise_cost.data)
    if algorithm == 'swap':
        gc.swap(n_iter)
    elif algorithm == 'expansion':
        gc.expansion(n_iter)
    else:
        raise ValueError("algorithm should be either `swap` or `expansion`. Got: %s" % algorithm)

    cdef np.npy_intp result_shape[1]
    result_shape[0] = n_vertices
    cdef np.ndarray[np.int32_t, ndim=1] result = np.PyArray_SimpleNew(1, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(n_vertices):
        result_ptr[i] = gc.whatLabel(i)
    del gc
    return result
