gco python
==========

Python wrappers for GCO alpha-expansion and alpha-beta-swaps.
These wrappers provide a high level interface for graph cut
inference for multi-label problems.

See my blog for examples and comments: [peekaboo-vision.blogspot.com](https://peekaboo-vision.blogspot.com)



Installation
------------

- Download gco from http://vision.csd.uwo.ca/code/ and compile.

- Download and install Cython (use your package manager).

- Set the path in setup.py  to point to the path of gco.

- run ``python setup.py build``

- Install or add the build path (build/lib.XXX) to your ``PYTHONPATH``

- Run example.py for a simple example.

Usage
-----
GCO implements alpha expansion and alpha beta swaps using graphcuts.
These can be used to efficiently find low energy configurations of certain energy functions.
Note that from a probabilistic viewpoint, GCO works in log-space.

This package gives a high level interface to gco, providing the following functions:

``cut_simple``:
    Graph cut on a 2D grid using a global label affinity-matrix.

``cut_VH``:
    NOT DONE YET
    Graph cut on a 2D grid using a global label affinity-matrix and edge-weights.
    ``V`` contains the weights for vertical edges, ``H`` for horizontal ones.

``cut_from_graph``:
    Graph cut on an arbitrary graph with global label affinity-matrix.

``cut_from_graph_weighted``:
    NOT DONE YET
    Graph cut on an arbitrary graph with global label affinity-matrix and
    edgeweights.

See ``example.py`` and ``example_middlebury.py`` for examples and the gco README
for more details.
