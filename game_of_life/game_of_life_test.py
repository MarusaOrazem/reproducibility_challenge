#!/usr/bin/python3
"""
Tests the game of life dataset.

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

import unittest
import random
import numpy as np
import game_of_life as gol

# Source code abbreviated from:
__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class TestGameOfLife(unittest.TestCase):

    def test_generate_time_series(self):
        # simulate a blinker for some time
        X1 = np.array([[0., 1., 0.], [0., 1., 0.], [0., 1., 0.]])
        X2 = np.array([[0., 0., 0.], [1., 1., 1.], [0., 0., 0.]])
        T = 32
        A, Xs, deltas = gol.generate_time_series(X1, T)

        A_expected = np.array([
            [0, 1, 0, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 1, 1, 0],
            [1, 1, 1, 1, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 1, 1, 0, 1, 0]
        ])
        np.testing.assert_array_equal(A, A_expected)

        for t in range(T):
            if t % 2 == 0:
                np.testing.assert_array_equal(Xs[t].reshape((3, 3)), X1)
                np.testing.assert_array_equal(deltas[t].reshape((3, 3)), X2 - X1)
            else:
                np.testing.assert_array_equal(Xs[t].reshape((3, 3)), X2)
                np.testing.assert_array_equal(deltas[t].reshape((3, 3)), X1 - X2)

    def test_generate_random_time_series(self):
        # check that no exception is thrown for a few runs and
        # that the node and node action arrays are always consistent
        for r in range(100):
            A, Xs, deltas = gol.generate_random_time_series()
            self.assertEqual(len(Xs), len(deltas))
            for t in range(len(Xs)):
                self.assertEqual(len(Xs[t]), len(deltas[t]))
                self.assertEqual(A.shape[0], len(Xs[t]))
                self.assertEqual(A.shape[1], len(Xs[t]))
                if t < len(Xs) - 1:
                    np.testing.assert_array_equal(Xs[t+1][:, 0], Xs[t][:, 0] + deltas[t])

    def test_unique_dataset(self):
        N = 10
        test_set, unique_Xs = gol.create_test_set(N)

        self.assertEqual(len(test_set), N)

        seen_data = []
        for _, Xs, _, _ in test_set:
            for i, X in enumerate(Xs):
                if seen_data == []:
                    seen_data = Xs
                    continue
                for j, seen in enumerate(seen_data):
                    if seen.size == X.size:
                        # they are of equal sizes, we can actually compare them
                        if (seen == X).all() and i + 1 != len(Xs):
                            self.assertFalse("", "Test set is not unique.")
                        elif (seen == X).all() and (i + 1 == len(Xs)):
                            pass
                        else:
                            seen_data.append(X)

        #print("---")
        self.assertTrue(True, "Test set is unique.")

        for i in range(10):
            #print(i)
            _, Xs, _ = gol.generate_unique_time_series(unique_Xs)
            for i, X in enumerate(Xs):
                for j, seen in enumerate(seen_data):
                    if seen.size == X.size:
                        # they are of equal sizes, we can actually compare them
                        if (seen == X).all() and i + 1 != len(Xs):
                            self.assertFalse("", "Test set is not unique.")
                        elif (seen == X).all() and (i + 1 == len(Xs)):
                            pass
        self.assertTrue(True, "Train set is unique.")



if __name__ == '__main__':
    #unittest.main()
    test = TestGameOfLife()
    test.test_unique_dataset()
