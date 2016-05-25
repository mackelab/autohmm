from __future__ import print_function

import unittest
from unittest import TestCase

import numpy as np

from numpy.testing import assert_, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal

from hmmlearn.utils import normalize

from autohmm import tm

np.seterr(all='warn')

def test_precision_prior_wrong_nb():
    with assert_raises(ValueError):
        m = tm.THMM(n_unique = 2)
        m.precision_prior_ = np.array([0.7, 0.8, 0.9])

def test_precision_prior_unique():
    m = tm.THMM(n_unique = 2, n_tied = 1)
    m.precision_prior_ = np.array([[0.7], [0.3]])
    correct_prior = np.array([0.7, 0.7, 0.3, 0.3])
    correct_prior = correct_prior.reshape(4, 1, 1)
    assert_array_equal(m._precision_prior_, correct_prior)

def fit_hmm_and_monitor_log_likelihood(h, X, n_iter=1):
    #h.n_iter = 1        # make sure we do a single iteration at a time
    #h.init_params = ''  # and don't re-init params
    h.fit(X)
    loglikelihoods = np.empty(n_iter, dtype=float)
    #for i in range(n_iter):
    #    h.fit(X)
    #    loglikelihoods[i], _ = h.score_samples(X)
    return loglikelihoods

class PlainGaussianHMM(TestCase):
    def setUp(self):
        self.prng = np.random.RandomState(2)

        self.n_unique = 2
        self.n_components = 2
        self.startprob = np.array([0.6, 0.4])
        self.transmat = np.array([[0.7, 0.3],
                                  [0.4, 0.6]])
        self.mu = np.array([0.7, -2.0])
        self.precision = np.array([[500],
                                   [250]])

        self.h = tm.THMM(n_unique=self.n_unique,
                        random_state=self.prng,
                         init_params = 'stmw',
                         precision_bounds = np.array([-1e5, 1e5]))
        self.h.startprob_ = self.startprob
        self.h.transmat_ = self.transmat
        self.h.mu_ = self.mu
        self.h.precision_ = self.precision

    def test_fit(self, params='sptmw', **kwargs):
        h = self.h
        h.params = params

        lengths = 70000
        X, _state_sequence = h.sample(lengths, random_state=self.prng)

        h.precision_ = np.array([[700],
                                 [150]])
        h.mu_ = np.array([2.6, 3.4])
        h.transmat_ = np.array([[0.85, 0.15],
                                [0.2, 0.8]])
        # TODO: Test more parameters, generate test cases
        trainll = fit_hmm_and_monitor_log_likelihood(h, X)

        # Check that the log-likelihood is always increasing during training.
        #diff = np.diff(trainll)
        #self.assertTrue(np.all(diff >= -1e-6),
        #                "Decreasing log-likelihood: {0}" .format(diff))


        assert_array_almost_equal(h.mu_.reshape(-1),
                                  self.mu.reshape(-1), decimal=1)

        assert_array_almost_equal(h.transmat_.reshape(-1),
                                  self.transmat.reshape(-1), decimal=1)
        assert_array_almost_equal(h.precision_.reshape(-1)/100,
                                  self.precision.reshape(-1)/100, decimal =1)

class MultivariateGaussianHMM(TestCase):
    def setUp(self):
        self.prng = np.random.RandomState(2)
        self.n_tied = 2
        self.n_features = 2
        self.startprob = np.array([0.6, 0.4])
        self.transmat = np.array([[0.7, 0.3], [0.4, 0.6]])

        self.mu = np.array([[4.5, -1.5],
                            [-0.7, -10.4]])

        self.precision = np.array([[[0.5, 0.15],
                                    [0.15, 0.4]],
                                   [[0.6, 0.1],
                                    [0.1, 0.35]]])

        self.h = tm.THMM(n_unique=2, n_tied =self.n_tied,
                         n_features=self.n_features,
                         random_state=self.prng,
                         precision_bounds=np.array([-1e5, 1e5]),
                         init_params = 'stmaw', params='stmapw')
        self.h.startprob_ = self.startprob
        self.h.transmat_ = self.transmat
        self.h.mu_ = self.mu
        self.h.precision_ = self.precision

    def test_fit(self, params='stmpaw', **kwargs):
        h = self.h
        h.params = params
        lengths = 100000
        X, _state_sequence = h.sample(lengths, random_state=self.prng)

        # Perturb

        h.precision_ = np.array([[[0.4, 0.12],
                                  [0.12, 0.45]],
                                 [[0.7, 0.2],
                                  [0.2, 0.5]]])
        h.transmat_ = np.array([[0.5, 0.5], [0.2, 0.8]])
        h.mu_ = np.array([[5.8, -0.1],
                          [-3.3, -9.6]])

        self.transmat = np.array([[0.7, 0.3, 0, 0, 0, 0],
                                 [0, 0.7, 0.3, 0, 0, 0],
                                 [0, 0, 0.7, 0.3, 0, 0],
                                 [0, 0, 0, 0.6, 0.4, 0],
                                 [0, 0, 0, 0, 0.6, 0.4],
                                 [0.4, 0, 0, 0, 0, 0.6]])

        # TODO: Test more parameters, generate test cases
        trainll = fit_hmm_and_monitor_log_likelihood(h, X)
        # Check that the log-likelihood is always increasing during training.
        #diff = np.diff(trainll)
        #self.assertTrue(np.all(diff >= -1e-6),
        #                "Decreasing log-likelihood: {0}" .format(diff))

        assert_array_almost_equal(h.transmat_.reshape(-1),
                                  self.transmat.reshape(-1), decimal=1)
        assert_array_almost_equal(h.mu_.reshape(-1),
                                  self.mu.reshape(-1), decimal=1)
        assert_array_almost_equal(h.precision_.reshape(-1),
                                  self.precision.reshape(-1), decimal=1)

class TiedGaussianHMM(TestCase):
    def setUp(self):
        self.prng = np.random.RandomState(42)

        self.n_tied = 2
        self.n_unique = 2
        self.startprob = np.array([0.6, 0.4])
        self.transmat = np.array([[0.7, 0.3],
                                  [0.4, 0.6]])
        self.precision = np.array([[0.5],
                                   [0.3]])
        self.mu = np.array([[0.7],
                            [-2.0]])
        self.h = tm.THMM(n_unique=self.n_unique, n_tied =self.n_tied, random_state=self.prng,
                         precision_bounds=np.array([-1e5, 1e5]), init_params = 'stmaw')
        self.h.startprob_ = self.startprob
        self.h.transmat_ = self.transmat
        self.h.mu_ = self.mu
        self.h.precision_ = self.precision

    def test_fit(self, params='stmpaw', **kwargs):
        h = self.h
        h.params = params
        lengths = 70000
        X, _state_sequence = h.sample(lengths, random_state=self.prng)

        h.mu_ = np.array([[3.5],
                          [-3.9]])
        h.transmat_ = np.array([[0.9, 0.1],
                                [0.7, 0.3]])
        h.precision_ = np.array([[0.4],
                                 [0.2]])

        self.transmat = np.array([[0.7, 0.3, 0, 0, 0, 0],
                                  [0, 0.7, 0.3, 0, 0, 0],
                                  [0, 0, 0.7, 0.3, 0, 0],
                                  [0, 0, 0, 0.6, 0.4, 0],
                                  [0, 0, 0, 0, 0.6, 0.4],
                                  [0.4, 0, 0, 0, 0, 0.6]])

        # TODO: Test more parameters, generate test cases

        trainll = fit_hmm_and_monitor_log_likelihood(h, X)

        # Check that the log-likelihood is always increasing during training.
        #diff = np.diff(trainll)
        #self.assertTrue(np.all(diff >= -1e-6),
        #                "Decreasing log-likelihood: {0}" .format(diff))
        assert_array_almost_equal(h.mu_.reshape(-1),
                                  self.mu.reshape(-1), decimal=1)
        assert_array_almost_equal(h.transmat_.reshape(-1),
                                  self.transmat.reshape(-1), decimal=1)
        assert_array_almost_equal(h.precision_.reshape(-1),
                                  self.precision.reshape(-1), decimal=0)

if __name__ == '__main__':
    unittest.main()
