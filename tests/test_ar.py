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

# test alpha setter
def test_alpha_set_unique():
    h = ar.ARTHMM(n_unique=2, n_tied=2, n_lags=3, shared_alpha=False)
    h.alpha_ = np.array([[0.1, 0.2, 0.5],
                         [0.2, 0.5, 0.5]])

    correct_alpha = np.array([[0.1, 0.2, 0.5],
                              [0.1, 0.2, 0.5],
                              [0.1, 0.2, 0.5],
                              [0.2, 0.5, 0.5],
                              [0.2, 0.5, 0.5],
                              [0.2, 0.5, 0.5]])

    assert_array_equal(h._alpha_, correct_alpha)

def test_alpha_wrong_unique():
    with assert_raises(ValueError):
        h = ar.ARTHMM(n_unique=2, n_lags=1, shared_alpha=False)
        h.alpha_ = np.array([[0.1, 0.2],
                             [0.2, 0.2],
                             [0.2, 0.2]])

def test_alpha_wrong_nlags():
    with assert_raises(ValueError):
        h = ar.ARTHMM(n_unique=2, n_lags=1, shared_alpha=False)
        h.alpha_ = np.array([[0.1, 0.2],
                             [0.2, 0.2]])

def test_alpha_unequal_despite_tied():
    with assert_raises(ValueError):
        h = ar.ARTHMM(n_unique=2, n_lags=1, shared_alpha=True)
        h.alpha_ = np.array([[0.1], [0.2]])

# test alpha getter
def test_alpha_getter():
    h = ar.ARTHMM(n_unique=2, n_tied=2, n_lags=3, shared_alpha=False)
    new_alpha = np.array([[0.1, 0.2, 0.5],
                         [0.2, 0.5, 0.5]])
    h.alpha_ = new_alpha
    assert_array_equal(h.alpha_, new_alpha)


def fit_hmm_and_monitor_log_likelihood(h, X, n_iter=1):
    #h.n_iter = 1        # make sure we do a single iteration at a time
    #h.init_params = ''  # and don't re-init params
    loglikelihoods = np.empty(n_iter, dtype=float)
    h.fit(X)
    #for i in range(n_iter):
    #    h.fit(X)
    #    loglikelihoods[i], _ = h.score_samples(X)
    return loglikelihoods

class ARMultivariateGaussianHMM(TestCase):
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
                               [0.003, 0.004]])

        self.precision = np.array([[[0.8, 0.1],
                                    [0.1, 0.5]],
                                   [[0.9, 0.3],
                                    [0.3, 0.8]]])

        self.h = ar.ARTHMM(n_unique=self.n_unique, n_lags=self.n_lags,
                            random_state=self.prng, n_features=self.n_features,
                            shared_alpha=False, verbose=False,
                            precision_bounds=np.array([-1e5, 1e5]),
                            init_params = 'samt')

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
                             [0.007, 0.002]])

        h.precision_ = np.array([[[0.7, 0.3],
                                  [0.3, 0.6]],
                                 [[1.2, 0.2],
                                  [0.2, 0.8]]])

        trainll = fit_hmm_and_monitor_log_likelihood(h, X)

        # Check that the log-likelihood is always increasing during training.
        #diff = np.diff(trainll)
        #self.assertTrue(np.all(diff >= -1e-6),
        #                "Decreasing log-likelihood: {0}" .format(diff))

        assert_array_almost_equal(h.mu_.reshape(-1), self.mu.reshape(-1),
                                 decimal=1)
        assert_array_almost_equal(h.precision_.reshape(-1), self.precision.reshape(-1),
                                  decimal=1)
        assert_array_almost_equal(h.transmat_.reshape(-1),
                                 self.transmat.reshape(-1), decimal=1)
        assert_array_almost_equal(h.alpha_.reshape(-1),
                                 self.alpha.reshape(-1), decimal=3)


class ARGaussianHMM(TestCase):
    def setUp(self):
        self.prng = np.random.RandomState(14)  # TODO: remove fixed seeding

        self.n_unique = 2
        self.n_lags = 1
        self.startprob = np.array([0.6, 0.4])
        self.transmat = np.array([[0.8, 0.2], [0.1, 0.9]])
        self.mu = np.array([[-2.0], [12.0]])
        self.alpha = np.array([0.08, 0.02])
        self.shared_alpha = False
        self.precision = np.array([[0.8],
                                   [0.9]])
        self.h = ar.ARTHMM(n_unique=self.n_unique,
                           n_lags=self.n_lags,
                           random_state=self.prng,
                           shared_alpha=self.shared_alpha,
                           verbose=False, init_params = 'samt',
                           precision_bounds=np.array([-1e5, 1e5]))
        self.h.startprob_ = self.startprob
        self.h.precision_ = self.precision
        self.h.transmat_ = self.transmat
        self.h.mu_ = self.mu
        self.h.alpha_ = self.alpha

    def test_fit(self, **kwargs):

        h = self.h
        lengths = 50000
        X, true_state_sequence = h.sample(lengths, random_state=self.prng)

        # Perturb Parameters
        h.alpha_ = np.array([[0.05],
                             [0.03]])

        h.mu_ = np.array([-4.5, 8.0])

        h.transmat_ = np.array([[0.5, 0.5],
                                [0.2, 0.8]])

        h.precision_ = np.array([[0.6],
                                 [1.3]])

        # Fit and Check Recovery
        trainll = fit_hmm_and_monitor_log_likelihood(h, X)

        # Check that the log-likelihood is always increasing during training.
        #diff = np.diff(trainll)
        #self.assertTrue(np.all(diff >= -1e-6),
        #                "Decreasing log-likelihood: {0}" .format(diff))

        assert_array_almost_equal(h.mu_.reshape(-1), self.mu.reshape(-1),
                                  decimal=1)
        assert_array_almost_equal(h.precision_.reshape(-1),
                                  self.precision.reshape(-1), decimal=1)
        assert_array_almost_equal(h.transmat_.reshape(-1),
                                  self.transmat.reshape(-1), decimal=1)
        assert_array_almost_equal(h.alpha_.reshape(-1),
                                  self.alpha.reshape(-1), decimal=2)

        # TODO: add test for decoding
        #logprob, decoded_state_sequence = h.decode(X)
        #assert_array_equal(true_state_sequence, decoded_state_sequence)


if __name__ == '__main__':
    unittest.main()
