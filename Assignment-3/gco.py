"""Graph cuts optimization using PyMaxFlow.
Installation:       pip install PyMaxflow
GitHub:             https://github.com/pmneila/PyMaxflow
Documentation:      http://pmneila.github.io/PyMaxflow
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import maxflow
import numpy as np
from scipy.sparse import csr_matrix


def graphcut(unary, pairwise):
    """Solves binary s-t cut optimization problem for given unary and pairwise capacities.

      Args:
        unary: A `numpy.ndarray`. Typically of the following types: `float32`, `float64`, `int32`, `int64`.
          Capacities for edges between non-terminal nodes (pixels) and terminal nodes (source/target).
          Dimension is expected to be 2xN, where N is the number of non-terminal nodes, and,
          the first/second row contains the capacities for edges connected with the source/target node, respectively.

        pairwise: A `scipy.sparse.csr_matrix`. Typically of the following types: `float32`, `float64`, `int32`, `int64`.
          An upper triangular sparse matrix containing the capacities for edges between non-terminal nodes, whose
          dimension is expected to be NxN. Since forward and backward edges between non-terminal nodes are expected to
          have the same capacity, this sparse matrix should only have non-zeros in the upper triangular part.

      Returns:
        A `nd.array` of type `int32`. Assigned labels (source=0 or target=1) minimizing the costs.
      """

    assert (isinstance(unary, np.ndarray) and isinstance(pairwise, csr_matrix))
    assert (unary.dtype in [np.int32, np.int64, np.float32, np.float64])
    assert (pairwise.dtype in [np.int32, np.int64, np.float32, np.float64])
    assert (unary.shape[0] == 2)
    assert (np.equal(pairwise.shape, unary.shape[1]).all())

    # create graph with two terminal nodes
    num_sites = unary.shape[1]
    graph = maxflow.Graph[float](num_sites, 2)

    # Add non-terminal nodes
    nodes = graph.add_nodes(num_sites)

    # Add edges between nodes
    cx = pairwise.tocoo()
    for row, col, val in zip(cx.row, cx.col, cx.data):
        graph.add_edge(nodes[row], nodes[col], val, val)
        if col <= row:
            warnings.warn("Pairwise capacity matrix should be upper triangular!")

    # Set capacities for terminal edges
    for j in range(num_sites):
        graph.add_tedge(nodes[j], unary[1, j], unary[0, j])

    # Perform the cut
    graph.maxflow()

    # Retrieve labels
    labels = np.array([graph.get_segment(nodes[i]) for i in range(num_sites)])
    return labels


def test_graphcut():
    """Runs a simple example for the function above."""

    # unary potentials for 8 pixels
    unary = np.array([
        -64.6044, -56.7481, -64.7098, -52.5734, -64.5179, -63.9195, -62.8145, -62.3820,
        -64.4310, -56.7481, -64.5164, -63.8380, -62.0581, -58.3476, -62.8145, -62.3820]).reshape((2, 8))

    # pairwise capacities between pixels: This should be an upper triangular matrix
    pairwise = csr_matrix(np.array([0, 2, 2, 0, 0, 0, 0, 0,
                                    0, 0, 0, 2, 0, 0, 0, 0,
                                    0, 0, 0, 2, 2, 0, 0, 0,
                                    0, 0, 0, 0, 0, 2, 0, 0,
                                    0, 0, 0, 0, 0, 2, 2, 0,
                                    0, 0, 0, 0, 0, 0, 0, 2,
                                    0, 0, 0, 0, 0, 0, 0, 2,
                                    0, 0, 0, 0, 0, 0, 0, 0]).reshape((8, 8)))

    # NOTE: for performance reasons you should directly construct a sparse matrix (without an intermediate dense matrix)

    # perform the graph cut with given capacities
    labels = graphcut(unary, pairwise)

    # print resulting labels
    print("Unary:\n", unary)
    print("Pairwise:\n", pairwise.todense())
    print("Labels:\n", labels)
