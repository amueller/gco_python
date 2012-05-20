gco python
==========

Python wrappers for GCO alpha-expansion and alpha-beta-swaps.
These wrappers provide a high level interface for graph cut
inference for multi-label problems.

See my blog for examples and comments: [peekaboo-vision.blogspot.com](https://peekaboo-vision.blogspot.com)



Installation
------------

- Download gco from http://vision.csd.uwo.ca/code/ and compile.

- Install boost-python (on debian/ubuntu: sudo apt-get install libboost-python-dev).

- Download and install pyublas: http://mathema.tician.de/software/pyublas.

- Set the paths in ``Makefile`` to point to the path of your Python includes and gco.

- run ``make``

- Add the path to `gco_python` to your ``PYTHONPATH``

- Run ``python -c "import gco_python; gco_python.demo()"`` for a test.


Usage
-----
GCO implements alpha expansion and alpha beta swaps using graphcuts.
These can be used to efficiently find low energy configurations of certain energy functions.
Note that from a probabilistic viewpoint, GCO works in log-space.

This package gives a high level interface to gco, providing the following functions:

``cut_simple``:
    Graph cut on a 2D grid using a global label affinity-matrix.

``cut_VH``:
    Graph cut on a 2D grid using a global label affinity-matrix and edge-weights.
    ``V`` contains the weights for vertical edges, ``H`` for horizontal ones.

``cut_from_grpah``:
    Graph cut on an arbitrary graph with global label affinity-matrix.

``cut_from_grpah``:
    Graph cut on an arbitrary graph with global label affinity-matrix and
    edgeweights.

See ``example.py`` and ``example_middlebury.py`` for examples.
