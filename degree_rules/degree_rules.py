"""
Generates data for the 'degree rules' data set, where every node implements
the following rules:

1. delete the node if the degree is larger than 3.
2. add an edge to another node if both nodes share at least one neighbor.
3. add a node if the node degree is smaller than 3.

Rule 1 is always executed before rule 2 and rule 2 is always executed before
rule 3. When a rule applies to several nodes, older nodes take precedence.
Rules are applied in parallel for all connected components.

"""

# Copyright (C) 2020-2021
# Benjamin Paaßen
# The University of Sydney

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys

sys.path.append('../graph_edit')

import numpy as np
import graph_edits
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph
from networkx.generators.degree_seq import configuration_model

# Source code abbreviated from:
__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

def next_step(A, n_max = None):
    """ Generates the next step for a graph with adjacency matrix A,
    according to the rules.

    Parameters
    ----------
    A: class numpy.array
        A graph, given as an adjacency matrix.
    n_max: int (default = len(A))
        A reference number of dimensions for the node encoding, which will be
        a one-hot coding padded with zeros if necessary.

    Returns
    -------
    edits: list
        A list of graph edits that should be applied next.
    delta: class numpy.array
        The node len(A) x 1 edit vector with entries +1 for node insertions and
        -1 for node deletions. This vector considers all nodes for which a rule
        applies, disregarding the preference rules.
    Epsilon: class numpy array
        The egde len(A) x len(A) edit matrix with entries +1 for egde
        insertions and -1 for edge deletions. This matrix considers all edges
        for which a rule applies, disregarding the preference rules.

    """
    if n_max is None:
        n_max = len(A)
    elif n_max < len(A):
        raise ValueError('n_max must be at least len(A) = %d to ensure that the graph is codeable.' % len(A))
    # first, distribute the graph into connected components
    # and process smaller connected components first.
    Cs = []
    remaining = set(range(len(A)))
    while(remaining):
        i = min(remaining)
        C = []
        stk = [i]
        while(stk):
            i = stk.pop()
            if(i not in remaining):
                continue
            remaining.remove(i)
            C.append(i)
            for j in np.where(A[i, :])[0]:
                if(j in remaining):
                    stk.append(j)
        Cs.append(np.array(C, dtype=int))
    # process all connected components
    edits = []
    dels  = []
    new_node_idx = len(A)
    for C in Cs:
        C.sort()
        # sort the nodes inside the component according to degree
        degrees = np.sum(A[C, :], axis=1)
        deg_ordered = np.argsort(degrees)
        # evaluate rule 1, i.e. delete nodes with degree higher than 3,
        # where nodes with higher degree take precedence
        if(degrees[deg_ordered[-1]] > 3):
            dels.append(graph_edits.NodeDeletion(C[deg_ordered[-1]]))
            continue
        # evaluate rule 2, i.e. connect nodes with shared neighbors,
        # where tuples with lower degrees take precedence
        rule2mat = np.logical_and(np.dot(A[C, :], A[:, C]) > 0.5, np.logical_not(A[C, :][:, C]))
        np.fill_diagonal(rule2mat, False)
        if np.any(rule2mat):
            degmat   = np.expand_dims(degrees, 0) + np.expand_dims(degrees, 1)
            degmat[np.logical_not(rule2mat)] = np.max(degmat) + 1
            i, j = np.unravel_index(np.argmin(degmat), degmat.shape)
            i, j = C[i], C[j]
            edits.append(graph_edits.EdgeInsertion(i, j, False))
            continue
        # evaluate rule 3, i.e. add new nodes to nodes with a low degree,
        # preferring nodes with lowest degrees
        if(degrees[deg_ordered[0]] < 3):
            new_node_code = np.zeros(n_max)
            new_node_code[new_node_idx] = 1
            new_node_idx += 1
            edits.append(graph_edits.NodeInsertion(C[deg_ordered[0]], new_node_code, False))
    # append deletions at the end of the edits list
    edits += dels
    # compute delta and Epsilon
    delta = np.zeros(len(A))
    Epsilon = np.zeros(A.shape)
    degrees = np.sum(A, axis=1)
    delta[degrees > 3] = -1. # rule 1
    Epsilon[np.logical_and(np.dot(A, A) > 0.5, np.logical_not(A))] = 1. # rule 2
    np.fill_diagonal(Epsilon, 0.) # correct for self-connections
    delta[degrees < 3] = 1. # rule 3
    # return results
    return edits, delta, Epsilon

def generate_time_series(A, n_max = None, t_max = 100):
    """ Generates a time series of graphs, in terms of their adjacency matrix,
    based on the degree rules from the given seed graph. The time series stops
    once the graph converges to a stable state.

    Parameters
    ----------
    A: class numpy.array
        An initial adjacency matrix for an undirected graph.
    n_max: int (default = 4 * len(A))
        a reference number of dimensions for the node encoding, which will be a
        one-hot coding padded with zeros if necessary.
    t_max: int (default = 100)
        A maximum number of steps.

    Returns
    -------
    As: list
        A time series of graphs until the graph converges to a stable state
        or t_max steps have been done.
    Xs: list
        a time series of node attribute matrices. These are just one-hot coding
        vectors of the node id.
    deltas: list
        a time series of node edit vectors where deltas[t][i] = +1 if node i
        spawns a new node at time step t, deltas[t][i] = -1 if node i is
        deleted at step t, and deltas[t][i] = 0 otherwise.
    Epsilons: list
        a time series of edge edit matrices, where Epsilons[t][i, j] = +1 if
        nodes i and j are newly connected at time t, Epsilons[t][i, j] = -1 if
        the egde (i, j) is removed at time t, and Epsilons[t][i, j] = 0
        otherwise.

    """
    if n_max is None:
        n_max = 4 * len(A)
    elif n_max < 4 * len(A):
        raise ValueError('n_max must be at least 4 * len(A) = %d to ensure that even after growth beyond its initial size the graph still is codeable.' % (4 * len(A)))

    # number the nodes in each connected component
    Cs = []
    remaining = set(range(len(A)))
    while(remaining):
        i = min(remaining)
        C = []
        stk = [i]
        while(stk):
            i = stk.pop()
            if(i not in remaining):
                continue
            remaining.remove(i)
            C.append(i)
            for j in np.where(A[i, :])[0]:
                if(j in remaining):
                    stk.append(j)
        Cs.append(np.array(C, dtype=int))
    As = [A]
    X  = np.eye(len(A), n_max)
    Xs = [X]
    deltas = []
    Epsilons = []
    t = 1
    while(t_max is None or t < t_max):
        edits, delta, Epsilon = next_step(A, n_max)
        m = len(A)
        if(len(edits) == 0):
            break
        # apply edits
        for edit in edits:
            A, X = edit.apply(A, X)
        As.append(A)
        Xs.append(X)
        deltas.append(delta)
        Epsilons.append(Epsilon)
        t += 1
    deltas.append(delta)
    Epsilons.append(Epsilon)
    return As, Xs, deltas, Epsilons

def generate_time_series_from_random_matrix(N, p = 0.5, n_max = None, t_max = 100):
    """ Generates a random, undirected, initial graph and lets it evolve
    until the graph converges or until the maximum number of steps is reached.

    Parameters
    ----------
    N: int
        The number of nodes for the initial graph.
    p: float in the range [0., 1.] (default = 0.5)
        The likelihood of edges being present.
    n_max: int (default = 4 * N)
        a reference number of dimensions for the node encoding, which will
        be a one-hot coding padded with zeros if necessary.
    t_max: int (default = 100)
        A maximum number of steps.

    Returns
    -------
    As: list
        A time series of graphs until the graph converges to a stable state
        or t_max steps have been done.
    Xs: list
        a time series of node attribute matrices. These are just one-hot coding
        vectors of the node id.
    deltas: list
        a time series of node edit vectors where deltas[t][i] = +1 if node i
        spawns a new node at time step t, deltas[t][i] = -1 if node i is
        deleted at step t, and deltas[t][i] = 0 otherwise.
    Epsilons: list
        a time series of edge edit matrices, where Epsilons[t][i, j] = +1 if
        nodes i and j are newly connected at time t, Epsilons[t][i, j] = -1 if
        the egde (i, j) is removed at time t, and Epsilons[t][i, j] = 0
        otherwise.

    """
    # generate a matrix of random numbers
    A = np.random.rand(N, N)
    # symmetrize it to make it undirected
    A = 0.5 * (A + A.T)
    # round it to integers
    A[A >= 1. - p] = 1.
    A = np.round(A)
    # remove the diagonal
    A -= np.diag(np.diag(A))
    # return evolution
    return generate_time_series(A, n_max = n_max, t_max = t_max)

################################################################################################
# Additional functions for experiments in 4.2.1

def generate_time_series_from_erdos_reny(N, p = 0.5, n_max = None, t_max = 100, seed = None):
    """ Generates a random, undirected, initial graph from Erdos Renyi model and lets it evolve
        until the graph converges or until the maximum number of steps is reached.

        Parameters
        ----------
        N: int
            The number of nodes for the initial graph.
        p: float in the range [0., 1.] (default = 0.5)
            The likelihood of edges being present.
        n_max: int (default = 4 * N)
            a reference number of dimensions for the node encoding, which will
            be a one-hot coding padded with zeros if necessary.
        t_max: int (default = 100)
            A maximum number of steps.

        Returns
        -------
        As: list
            A time series of graphs until the graph converges to a stable state
            or t_max steps have been done.
        Xs: list
            a time series of node attribute matrices. These are just one-hot coding
            vectors of the node id.
        deltas: list
            a time series of node edit vectors where deltas[t][i] = +1 if node i
            spawns a new node at time step t, deltas[t][i] = -1 if node i is
            deleted at step t, and deltas[t][i] = 0 otherwise.
        Epsilons: list
            a time series of edge edit matrices, where Epsilons[t][i, j] = +1 if
            nodes i and j are newly connected at time t, Epsilons[t][i, j] = -1 if
            the egde (i, j) is removed at time t, and Epsilons[t][i, j] = 0
            otherwise.

        """
    G = erdos_renyi_graph(N, p, seed)
    A = nx.to_numpy_array(G)

    return generate_time_series(A, n_max = n_max, t_max = t_max)

def generate_time_series_from_configuration_model(N, n_max = None, t_max = 100, seed = None):
    """ Generates a random, undirected, initial graph from configuration model and lets it evolve
        until the graph converges or until the maximum number of steps is reached.

        Parameters
        ----------
        N: int
            The number of nodes for the initial graph.
        p: float in the range [0., 1.] (default = 0.5)
            The likelihood of edges being present.
        n_max: int (default = 4 * N)
            a reference number of dimensions for the node encoding, which will
            be a one-hot coding padded with zeros if necessary.
        t_max: int (default = 100)
            A maximum number of steps.

        Returns
        -------
        As: list
            A time series of graphs until the graph converges to a stable state
            or t_max steps have been done.
        Xs: list
            a time series of node attribute matrices. These are just one-hot coding
            vectors of the node id.
        deltas: list
            a time series of node edit vectors where deltas[t][i] = +1 if node i
            spawns a new node at time step t, deltas[t][i] = -1 if node i is
            deleted at step t, and deltas[t][i] = 0 otherwise.
        Epsilons: list
            a time series of edge edit matrices, where Epsilons[t][i, j] = +1 if
            nodes i and j are newly connected at time t, Epsilons[t][i, j] = -1 if
            the egde (i, j) is removed at time t, and Epsilons[t][i, j] = 0
            otherwise.

        """
    sequence = nx.random_powerlaw_tree_sequence(N, seed = seed, tries = 500)
    G = nx.configuration_model(sequence)
    A = nx.to_numpy_array(G)

    return generate_time_series(A, n_max=n_max, t_max=t_max)



def create_test_set(N, n, generator=generate_time_series_from_random_matrix, p=0.5, n_max=None, t_max=None):
    '''
    Creates unique test set, where none of the graphs, in time series that are generated, are the same
    Parameters
    ----------
    N, int, number of sets to create
    n, number of nodes in the initial graph
    generator, function that generates series (from random matrix, erdos renyi,...
    p, probability for generator
    n_max
    t_max

    Returns: list of sets of generated time series, list of unique As that appear in all time series
    -------

    '''
    test_set = []
    unique_As = []
    while len(test_set) < N:
        #print(len(test_set))
        As, Xs, deltas, Epsilons = generator(n, p, n_max, t_max)
        # i counter for the graphs, so we will know which series to add.
        for i, A in enumerate(As):
            try:
                if A not in unique_As:
                    # if graph is not yet seen add it to the list
                    unique_As.append(A)
                    if len(As) == i+1:
                        # we came to an end
                        test_set.append((As, Xs, deltas, Epsilons))
                else:
                    if len(As) == i+1:
                        pass
                        # we came to the end
                    else:
                        # add the series to the current counter
                        As, Xs, deltas, Epsilons = As[:i + 1], Xs[:i + 1], deltas[:i + 1], Epsilons[:i + 1]
                    test_set.append((As, Xs, deltas, Epsilons))
                    break

            except:
                # matrices are not of equal sizes and comparison failes. It does not matter, they are not the same :)
                # check if we are at the end
                if len(As) == i + 1:
                    # we came to an end
                    test_set.append((As, Xs, deltas, Epsilons))


    return test_set, unique_As

def generate_unique_time_series(n, unique_As, generator=generate_time_series_from_random_matrix,  p=0.5, n_max=None, t_max=None):
    train_set = []
    while len(train_set) < 1:
        As, Xs, deltas, Epsilons = generator(n, p, n_max, t_max)
        # i counter for the graphs, so we will know which series to add.
        for i, A in enumerate(As):
            try:
                if A not in unique_As:
                    # if graph is not in the test set, we continue
                    if len(As) == i + 1:
                        # we came to an end
                        train_set.append((As, Xs, deltas, Epsilons))
                else:
                    if len(As) == i + 1:
                        pass
                        # we came to the end
                    else:
                        # add the series to the current counter
                        As, Xs, deltas, Epsilons = As[:i + 1], Xs[:i + 1], deltas[:i + 1], Epsilons[:i + 1]
                    train_set.append((As, Xs, deltas, Epsilons))
                    break

            except:
                # matrices are not of equal sizes and comparison failes. It does not matter, they are not the same :)
                # check if we are at the end
                if len(As) == i + 1:
                    # we came to an end
                    train_set.append((As, Xs, deltas, Epsilons))


    return train_set[0]



