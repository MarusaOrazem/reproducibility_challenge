"""
Implementes a simple data set based on John Conway's
'Game of Life'. In particular, this implementes the
cyclic shapes 'blinker', 'beacon', 'toad', 'clock',
and 'glider', which are placed at random positions of a
pre-defined grid.

"""

# Copyright (C) 2019-2021
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

import random
import numpy as np

import sys

sys.path.append('../graph_edit')
import graph_edits as ge

# Source code abbreviated from:
__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

def generate_time_series(X, T):
    """ Simulates the game of life for the grid X for T time steps.

    Parameters
    ----------
    X: class numpy.array
        A m x n binary matrix providing information about which node is
        initialy alive (1) or dead (0).
    T: int
        The number of simulation steps.

    Returns
    -------
    A: class numpy.array
        The adjacency matrix for the input grid.
    Xs: list
        A time series of node attribute matrices.
    deltas: list
        A time series of node edit vectors, such as Xs[t+1] = Xs[t] + deltas[t].

    """
    (m, n) = X.shape
    # first, reshape X into a vector
    X = np.reshape(X, (m*n,1))
    # generate the adjacency matrix for the grid
    A = np.zeros((m*n, m*n))
    # iterate over all grid positions
    for i in range(m):
        for j in range(n):
            # get the coordinate (i, j) in raveled position
            k = i * n + j
            if i < m - 1:
                # if we are not in the last column, connect to the
                # right neighbor
                l = (i+1) * n + j
                A[k, l] = 1.
                A[l, k] = 1.
                if j < n - 1:
                    # if we are not in the last row, connect to the
                    # below right neighbor
                    l = (i+1) * n + j + 1
                    A[k, l] = 1.
                    A[l, k] = 1.
            if j < n - 1:
                # if we are not in the last row, connect to the
                # bottom neighbor
                l = i * n + j + 1
                A[k, l] = 1.
                A[l, k] = 1.
                if i > 0:
                    # if we are not in the first row, connect to the
                    # above right neighbor
                    l = (i-1) * n + j + 1
                    A[k, l] = 1.
                    A[l, k] = 1.

    Xs = []
    deltas = []
    for t in range(T):
        # record current node state
        Xs.append(X)
        # A node is alive in the next generation if 2 * the number of
        # all alive neighbors plus the node state itself is between
        # 5 and 7
        Xnext = (X + 2 * np.dot(A, X))
        Xnext[Xnext < 4.5] = 0.
        Xnext[Xnext > 7.5] = 0.
        Xnext[Xnext > 4.] = 1.
        # the change vector is Xnext - X
        delta = (Xnext - X).squeeze(1)
        # record the change vector
        deltas.append(delta)
        # update X
        X = Xnext
    return A, Xs, deltas

# define the initial shapes
shapes = {
    'blinker' : np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.float),
    'glider' : np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float),
    'beacon' : np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]], dtype=np.float),
    'toad' :   np.array([[0, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]], dtype=np.float),
    'clock' :  np.array([[0, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float)
}

my_shapes = {
    'mazing': np.array([[0,0,1,0,0,0,0], [0,1,0,1,0,0,0], [0,0,0,0,0,0,0], [1,1,0,0,0,1,0], [1,0,0,0,0,0,1], [0,0,0,1,0,1,0], [0,0,1,1,0,0,0]], dtype = np.float),
    'mold' : np.array([[0,0,1,0,1,0], [0,0,0,0,0,1], [0,1,0,0,1,0], [1,0,1,0,1,0], [1,0,0,1,0,0], [0,1,1,0,0,0]], dtype = np.float),
    'unix' : np.array([[0,0,0,0,0,1,1,0,0], [1,1,0,1,0,0,0,0,0], [1,1,0,0,1,0,0,0,1], [0,0,0,0,0,0,1,0,1], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,1,0], [0,0,0,0,0,0,1,1,0]], dtype=np.float),
    'caterer' : np.array([[0,1,1,1,0,0], [0,0,0,0,0,1], [1,0,0,0,0,1], [0,0,0,0,1,0], [0,1,1,0,0,0], [0,1,0,0,0,0], [0,1,0,0,0,0], [0,1,0,0,0,0]], dtype= np.float)
}

def generate_random_time_series(grid_size = 10, num_shapes = 1, p = 0.1, T = 32, use_my_shapes = False):
    """ Generates a random time series over a grid of a certain size for a
    certain number of steps. The grid is initialized with reference shapes
    placed on random locations on the grid.

    Parameters
    ----------
    grid_size: int (default = 10)
        The size of the grid.
    num_shapes: int (default = 1)
        The number of shapes to be placed.
    p: float in range [0., 1.] (default = 0.1)
        The fraction of grid cells that are randomly alive initially.
    T: int (default = 32)
        The number of simulation steps.

    Returns
    -------
    A: class numpy.array
        The adjacency matrix for the input grid.
    Xs: list
        A time series of node attribute matrices.
    deltas: list
        A time series of node edit vectors, such as Xs[t+1] = Xs[t] + deltas[t].

    """
    # initialize the grid randomly
    X = np.random.rand(grid_size, grid_size)
    X[X >  1. - p] = 1.
    X[X <= 1. - p] = 0.
    if grid_size > 4:
        # iterate over the shapes
        for s in range(num_shapes):
            # select a random shape
            if use_my_shapes:
                shape = list(my_shapes.keys())[random.randrange(len(my_shapes))]
                X_shape = my_shapes[shape]
            else:
                shape = list(shapes.keys())[random.randrange(len(shapes))]
                X_shape = shapes[shape]
            (m_shape, n_shape) = X_shape.shape
            # select a random location
            i = random.randrange(grid_size - m_shape)
            j = random.randrange(grid_size - n_shape)
            # place the shape on the grid
            X[i:(i+m_shape), :][:, j:(j+n_shape)] = X_shape
    # simulate the game of life
    return generate_time_series(X, T)


#############################################################################################
# Additionl functions for experiments that are not described in the paper, because the results were the same. Functions above create unique test/train split as in
# other experiments and it can additionaly create initial grids from other shapes.

def create_test_set(N, generator=generate_random_time_series, grid_size = 10, num_shapes = 1, p = 0.1, T = 32, use_my_shapes = False):
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
    unique_Xs = []
    while len(test_set) < N:
        #print(len(test_set))
        # print(len(test_set))
        A, Xs, deltas = generator(grid_size = grid_size, num_shapes = num_shapes, p = p, T = T, use_my_shapes = use_my_shapes)
        # i counter for the graphs, so we will know which series to add.
        As = [A] * len(Xs)
        Epsilons = [np.zeros_like(A)] * len(Xs)
        if unique_Xs == []:
            unique_Xs = Xs
        stop = False
        for i, x in enumerate(Xs):
            #print("-------")
            #print(x)
            for u_x in unique_Xs:
                if stop:
                    #print("stop")
                    continue
                if x.shape == u_x.shape:
                    #print("shape")
                    if (x != u_x).any() and (i +1 == len(Xs)):
                        test_set.append((As, Xs, deltas, Epsilons))
                        unique_Xs.append(x)
                        stop = True
                    elif (x == u_x).all():
                        #print("SAME")
                        As, Xs, deltas, Epsilons = As[:i + 1], Xs[:i + 1], deltas[:i + 1], Epsilons[:i + 1]
                        test_set.append((As, Xs, deltas, Epsilons))
                        stop = True
                    elif (x != u_x).any():
                        unique_Xs.append(x)
                else:
                    # diff shape, they are different
                    #print("----DIFF")
                    unique_Xs.append(x)
                    if i+1 == len(Xs):
                        test_set.append((As, Xs, deltas, Epsilons))


    return test_set, unique_Xs

def generate_unique_time_series( unique_Xs, generator=generate_random_time_series,  grid_size = 10, num_shapes = 1, p = 0.1, T = 32, use_my_shapes = False):
    train_set = []
    while len(train_set) < 1:
        A, Xs, deltas = generator(grid_size = grid_size, num_shapes = num_shapes, p = p, T = T, use_my_shapes = use_my_shapes)
        # i counter for the graphs, so we will know which series to add.
        for i, x in enumerate(Xs):
            for u_x in unique_Xs:
                if x.shape == u_x.shape:
                    if (x != u_x).any() and (i + 1 == len(Xs)):
                        return A, Xs, deltas
                    if (x == u_x).all():
                        A, Xs, deltas = A, Xs[:i + 1], deltas[:i + 1]
                        return A, Xs, deltas


    return train_set
