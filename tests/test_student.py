from __future__ import print_function

import unittest
from unittest import TestCase

import numpy as np

from numpy.testing import assert_, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal

from hmmlearn.utils import normalize

from autohmm import student

np.seterr(all='warn')

def test_alpha_set_unique():
    h = student.STUDENT(n_unique=2, n_tied=2, n_lags=3, shared_alpha=False)
    h.alpha_ = np.array([[0.1, 0.2, 0.5],
                         [0.2, 0.5, 0.5]])

    correct_alpha = np.array([[0.1, 0.2, 0.5],
                              [0.1, 0.2, 0.5],
                              [0.1, 0.2, 0.5],
                              [0.2, 0.5, 0.5],
                              [0.2, 0.5, 0.5],
                              [0.2, 0.5, 0.5]])

    assert_array_equal(h._alpha_, correct_alpha)

class ARMultivariateStudentHMM(TestCase):
    def setUp(self):
        self.prng = np.random.RandomState(14)  # TODO: remove fixed seeding
        self.n_unique = 2
        self.n_features = 2
        self.n_lags = 2
        self.startprob = np.array([0.6, 0.4])
        self.transmat = np.array([[0.8, 0.2],
                                  [0.1, 0.9]])

        self.mu = np.array([[5.6, 10.4],
                            [-4.7, 1.2]])

        self.alpha = np.array([[0.01, 0.2],
                               [0.03, 0.04]])

        self.precision = np.array([[[0.8, 0.1],
                                    [0.1, 0.5]],
                                   [[0.9, 0.3],
                                    [0.3, 0.8]]])


        self.h = student.STUDENT(n_unique=self.n_unique, n_lags=self.n_lags,
                                 random_state=self.prng,
                                 n_features=self.n_features, shared_alpha=False,
                                 verbose=False, init_params = 'samt')

        self.h.precision_ = self.precision
        self.h.startprob_ = self.startprob
        self.h.transmat_ = self.transmat
        self.h.mu_ = self.mu
        self.h.alpha_ = self.alpha

    def test_fit(self, **kwargs):
        h = self.h
        lengths = 50000
        X, true_state_sequence = h.sample(lengths, random_state=self.prng)

        # Perturb Parameters
        h.transmat_ = np.array([[0.6, 0.4],
                                [0.5, 0.5]])

        h.mu_ = np.array([[6.7, 15.4],
                          [-4.0, 0.3]])

        h.alpha_ = np.array([[0.03, 0.3],
                             [0.07, 0.02]])

        h.precision_ = np.array([[[0.7, 0.3],
                                  [0.3, 0.6]],
                                 [[1.2, 0.2],
                                  [0.2, 0.8]]])

        h.fit(X)


        assert_array_almost_equal(h.mu_.reshape(-1), self.mu.reshape(-1),
                                  decimal=1)
        assert_array_almost_equal(h.precision_.reshape(-1),
                                  self.precision.reshape(-1),
                                  decimal=1)
        assert_array_almost_equal(h.transmat_.reshape(-1),
                                  self.transmat.reshape(-1), decimal=1)
        assert_array_almost_equal(h.alpha_.reshape(-1),
                                  self.alpha.reshape(-1), decimal=2)

class ARStudentTHMM(TestCase):
    def setUp(self):
        self.prng = np.random.RandomState(16)

        self.n_tied = 2
        self.n_unique = 2
        self.n_lags = 1
        self.startprob = np.array([0.6, 0.4])
        self.transmat = np.array([[0.7, 0.3],
                                  [0.4, 0.6]])

        self.precision = np.array([[0.5],
                                   [0.3]])
        self.mu = np.array([[0.7],
                            [-2.0]])

        self.alpha = np.array([[0.4]])

        self.h = student.STUDENT(n_unique=self.n_unique, n_lags = self.n_lags,
                                 n_tied =self.n_tied, random_state=self.prng,
                                 precision_bounds=np.array([-1e5, 1e5]),
                                 init_params = 'smatp', shared_alpha = True,
                                 n_iter_min = 40)

        self.h.startprob_ = self.startprob
        self.h.transmat_ = self.transmat
        self.h.mu_ = self.mu
        self.h.alpha_ = self.alpha
        self.h.precision_ = self.precision

    def test_fit(self, params='mtaps', **kwargs):

        h = self.h
        h.params = params
        lengths = 100000

        # Sample
        X, sequence = h.sample(lengths, random_state=self.prng)

        # Perturb Parameters, fit and check for recovery

        h.mu_ = np.array([[1.],
                          [-2.5]])

        h.transmat_ = np.array([[0.9, 0.1],
                                [0.7, 0.3]])

        h.precision_ = np.array([[0.4],
                                 [0.2]])

        h.alpha_ = np.array([[0.3]])

        self.transmat = np.array([[0.7, 0.3, 0, 0, 0, 0],
                                  [0, 0.7, 0.3, 0, 0, 0],
                                  [0, 0, 0.7, 0.3, 0, 0],
                                  [0, 0, 0, 0.6, 0.4, 0],
                                  [0, 0, 0, 0, 0.6, 0.4],
                                  [0.4, 0, 0, 0, 0, 0.6]])

        h.fit(X)

        assert_array_almost_equal(h.mu_.reshape(-1),
                                  self.mu.reshape(-1), decimal=1)
        assert_array_almost_equal(h.transmat_.reshape(-1),
                                  self.transmat.reshape(-1), decimal=1)
        assert_array_almost_equal(h.precision_.reshape(-1),
                                  self.precision.reshape(-1), decimal=0)
        assert_array_almost_equal(h.alpha_.reshape(-1),
                                  self.alpha.reshape(-1), decimal=1)


if __name__ == '__main__':
    unittest.main()
