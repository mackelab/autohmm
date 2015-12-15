from __future__ import print_function

import unittest
from unittest import TestCase

import numpy as np

from numpy.testing import assert_, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal

from hmmlearn import hmm
from hmmlearn.utils import normalize

from autohmm import ar

from nose.plugins.skip import SkipTest

np.seterr(all='warn')

def test_alpha_wrong_nlags():
    with assert_raises(ValueError):
        h = ar.ARTHMM(n_unique=2, n_lags=1, shared_alpha=False)
        h.alpha_ = np.array([[0.1, 0.2], [0.2, 0.2]])

def test_alpha_unqual_despite_tied():
    with assert_raises(ValueError):
        h = ar.ARTHMM(n_unique=2, n_lags=1, shared_alpha=True)
        h.alpha_ = np.array([[0.1], [0.2]])

def test_alpha_set_unique():
    h = ar.ARTHMM(n_unique=2, n_tied=2, n_lags=2, shared_alpha=False)
    h.alpha_ = np.array([[0.1, 0.2], [0.2, 0.5]])
    correct_alpha = np.array([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2],
                              [0.2, 0.5], [0.2, 0.5], [0.2, 0.5]])
    assert_array_equal(h._alpha_, correct_alpha)

def fit_hmm_and_monitor_log_likelihood(h, X, n_iter=1):
    h.n_iter = 1        # make sure we do a single iteration at a time
    h.init_params = ''  # and don't re-init params
    loglikelihoods = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        h.fit(X)
        loglikelihoods[i], _ = h.score_samples(X)
    return loglikelihoods

class ARGaussianHMM(TestCase):
    def setUp(self):
        self.prng = np.random.RandomState(14)  # TODO: remove fixed seeding

        self.n_unique = 2
        self.n_lags = 1
        self.startprob = np.array([0.6, 0.4])
        self.transmat = np.array([[0.8, 0.2], [0.1, 0.9]])
        self.mu = np.array([2.0, -2.0])
        self.var = np.array([0.1, 0.3])
        self.alpha = np.array([[0.8], [0.25]])
        self.shared_alpha = False

        self.h = ar.ARTHMM(n_unique=self.n_unique,
                           n_lags=self.n_lags,
                           random_state=self.prng,
                           shared_alpha=self.shared_alpha,
                           verbose=False)
        self.h.startprob_ = self.startprob
        self.h.transmat_ = self.transmat
        self.h.mu_ = self.mu
        self.h.var_ = self.var
        self.h.alpha_ = self.alpha

    def test_fit(self, params='stpmaw', n_iter=15, **kwargs):
        h = self.h
        h.params = params

        lengths = 5000
        X, _state_sequence = h.sample(lengths, random_state=self.prng)

        # perturb parameters
        h.startprob_ = normalize(self.prng.rand(self.n_unique))
        h.transmat_ = normalize(self.prng.rand(self.n_unique,
                                               self.n_unique), axis=1)
        h.alpha_ = np.array([[0.001], [0.001]])
        h.var_ = np.array([0.5, 0.5])
        h.mu = np.array([10.0, 8.0])

        trainll = fit_hmm_and_monitor_log_likelihood(
            h, X, n_iter=n_iter)

        # Check that the log-likelihood is always increasing during training.
        #diff = np.diff(trainll)
        #self.assertTrue(np.all(diff >= -1e-6),
        #                "Decreasing log-likelihood: {0}" .format(diff))

        assert_array_almost_equal(h.mu_.reshape(-1), self.mu.reshape(-1),
                                  decimal=1)
        assert_array_almost_equal(h.var_.reshape(-1), self.var.reshape(-1),
                                  decimal=1)
        assert_array_almost_equal(h.transmat_.reshape(-1),
                                  self.transmat.reshape(-1), decimal=1)
        assert_array_almost_equal(h.alpha_.reshape(-1),
                                  self.alpha.reshape(-1), decimal=1)

if __name__ == '__main__':
    unittest.main()
