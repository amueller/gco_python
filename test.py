import unittest
from pygco import cut_simple, cut_from_graph
import numpy as np


class TestPyGco(unittest.TestCase):
    """Test the main pygco methods."""

    def setUp(self):
        """Set the random seed for reproducability."""
        np.random.seed(1234)

    def binary_data(self):
        # generate trivial data
        x = np.ones((10, 10))
        x[:, 5:] = -1

        x_noisy = x + np.random.normal(0, 0.8, size=x.shape)

        # create unaries
        unaries = x_noisy
        # Split into two channels, positive and negative
        unaries = np.dstack([unaries, -unaries])
        # as we convert to int, we need to multipy to get sensible values
        unaries = (10 * unaries.copy("C")).astype(np.int32)

        expected = np.zeros(x.shape, dtype=np.int32)
        # The left side has a high cost for class 0 and the right side
        # has a high cost for class 1
        expected[:, :5] = 1

        # create potts pairwise
        pairwise = -10 * np.eye(2, dtype=np.int32)

        # construct edges from a grid graph
        inds = np.arange(x.size).reshape(x.shape)
        horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
        vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
        edges = np.vstack([horz, vert]).astype(np.int32)

        return unaries, pairwise, edges, expected

    def test_cut_simple(self):
        """Test the cut_simple method."""
        unaries, pairwise, edges, expected = self.binary_data()

        # do simple cut
        result = cut_simple(unaries, pairwise)

        self.assertTrue(np.array_equal(result, expected))

    def test_cut_from_grpah(self):
        """Test the cut_from_graph method."""
        unaries, pairwise, edges, expected = self.binary_data()

        # do simple cut
        result = cut_from_graph(edges, unaries.reshape(-1, 2), pairwise)
        result = result.reshape(unaries.shape[:2])

        self.assertTrue(np.array_equal(result, expected))

if __name__ == '__main__':
    unittest.main()
