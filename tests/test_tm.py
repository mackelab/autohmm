from __future__ import print_function

import unittest
from unittest import TestCase

import numpy as np

from numpy.testing import assert_, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal

from hmmlearn.utils import normalize

from autohmm import tm

np.seterr(all='warn')

def fit_hmm_and_monitor_log_likelihood(h, X, n_iter=1):
    h.n_iter = 1        # make sure we do a single iteration at a time
    h.init_params = ''  # and don't re-init params
    loglikelihoods = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        h.fit(X)
        loglikelihoods[i], _ = h.score_samples(X)
    return loglikelihoods

class PlainGaussianHMM(TestCase):
    def setUp(self):
        self.prng = np.random.RandomState(42)

        self.n_components = 2
        self.startprob = np.array([0.6, 0.4])
        self.transmat = np.array([[0.7, 0.3], [0.4, 0.6]])
        self.mu = np.array([0.7, -2.0])
        self.var = np.array([0.2, 0.2])

        self.h = tm.THMM(n_unique=self.n_components, random_state=self.prng)
        self.h.startprob_ = self.startprob
        self.h.transmat_ = self.transmat
        self.h.mu_ = self.mu
        self.h.var_ = self.var

    def test_fit(self, params='stpmaw', n_iter=5, **kwargs):
        h = self.h
        h.params = params

        lengths = 1000
        X, _state_sequence = h.sample(lengths, random_state=self.prng)

        # Perturb
        h.startprob_ = normalize(self.prng.rand(self.n_components))
        h.transmat_ = normalize(self.prng.rand(self.n_components,
                                               self.n_components), axis=1)

        # TODO: Test more parameters, generate test cases
        trainll = fit_hmm_and_monitor_log_likelihood(
            h, X, n_iter=n_iter)

        # Check that the log-likelihood is always increasing during training.
        #diff = np.diff(trainll)
        #self.assertTrue(np.all(diff >= -1e-6),
        #                "Decreasing log-likelihood: {0}" .format(diff))

        assert_array_almost_equal(h.mu_.reshape(-1),
                                  self.mu.reshape(-1), decimal=1)
        assert_array_almost_equal(h.var_.reshape(-1),
                                  self.var.reshape(-1), decimal=1)
        assert_array_almost_equal(h.transmat_.reshape(-1),
                                  self.transmat.reshape(-1), decimal=2)

if __name__ == '__main__':
    unittest.main()
