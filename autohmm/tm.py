from __future__ import division, print_function, absolute_import

import string
import warnings
import time

import sys
from collections import deque

import numpy as np

import scipy.sparse as sp

from scipy import linalg
from scipy import stats
from scipy.stats import norm

from sklearn import cluster
from sklearn.utils import check_random_state
from sklearn.mixture import sample_gaussian

from hmmlearn.base import _hmmc
from hmmlearn.base import _BaseHMM
from hmmlearn.utils import normalize, logsumexp, iter_from_X_lengths

from . import auto

__all__ = ['THMM']

ZEROLOGPROB = -1e200
EPS = np.finfo(float).eps
NEGINF = -np.inf

decoder_algorithms = frozenset(("viterbi", "map"))

class THMM(_BaseHMM):
    """Hidden Markov Model with Tied States, Gaussian Observations and Priors

    Parameters
    ----------
    n_unique : int
        Number of unique components.

    n_tied : int
        Number of tied states for each component.

    tied_precision : bool
        If set to true, precision (inverse variance) is shared across states.

    algorithm : string
        Decoding algorithm.

    params : string
        Controls which parameters are updated in the training
        process. Defaults to all parameters.

    init_params : string
        Controls which parameters are initialized prior to
        training. Defaults to all parameters.

    startprob_init : array, shape (``n_unique``)
        Initial state occupation distribution.

    startprob_prior : array, shape (``n_unique``)
        Pseudo-observations (counts).

    transmat_init : array, shape (``n_unique``, ``n_unique``)
        Matrix of transition probabilities between states.

    transmat_prior : array, shape (``n_unique``, ``n_unique``)
        Pseudo-observations (counts).

    mu_init : array, shape (``n_unique``)
        Initial mean parameters for each state.

    mu_weight : int
        Weight of mu prior, shared across components.

    mu_prior : array, shape (``n_unique``)
        Prior on mu.

    precision_init : array, shape (``n_unique``)
        Initial precision (inverse variance) parameters for each state.

    precision_weight : int
        Weight of precision (inverse variance) prior.

    precision_prior : array, shape (``n_unique``)
        Prior on precision (inverse variance).

    tol : float
        Convergence threshold, below which EM will stop.

    n_iter : int
        Number of iterations to perform maximally.

    n_iter_min : int
        Number of iterations to perform minimally.

    n_iter_update : int
        Number of iterations per M-Step.

    random_state : int
        Sets seed.

    verbose : bool
        When ``True`` convergence reports are printed.

    Attributes
    ----------
    n_components : int
        Number of total components

    mu_ : array, shape (``n_unique``)

    precision_ : array, shape (``n_unique``)

    transmat_ :  array, shape (``n_unique``, ``n_unique``)

    startprob_ :  array, shape (``n_unique``, ``n_unique``)
    """

    # The _BaseHMM class of hmmlearn is used for forward, backward,
    # and viterbi passes (optimized for speed).

    def __init__(self, n_unique=2, n_tied=0, tied_precision=False,
                 algorithm="viterbi",
                 params=string.ascii_letters, init_params=string.ascii_letters,
                 startprob_init=None, startprob_prior=1.0,
                 transmat_init=None, transmat_prior=1.0,
                 mu_init=None, mu_weight=0.0, mu_prior=None,
                 precision_init=None, precision_weight=0.0,
                 precision_prior=None, tol=1e-4,
                 n_iter=25, n_iter_min=2, n_iter_update=1,
                 random_state=None, verbose=False):
        _BaseHMM.__init__(self, n_components=n_unique*(1+n_tied),
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          params=params,
                          init_params=init_params)
        self.n_tied = n_tied
        self.n_chain = n_tied+1

        self.n_unique = n_unique
        self.n_components = n_unique * self.n_chain

        # only univariate observations implemented
        self.n_features = 1

        self.tied_precision = tied_precision

        # parameters are passed to a setters
        # setters check for shape and set default values
        self.startprob_ = startprob_init
        self.transmat_ = transmat_init
        self.transmat_prior_ = transmat_prior
        self.mu_ = mu_init
        self.mu_weight_ = mu_weight
        self.mu_prior_ = mu_prior
        self.precision_ = precision_init
        self.precision_weight_ = precision_weight
        self.precision_prior_ = precision_prior

        self.n_iter = n_iter
        self.n_iter_min = n_iter_min
        self.n_iter_update = n_iter_update

        self.tol = tol
        self.verbose = verbose

    def _init(self, data, lengths=None, params='stmp'):
        X = data['obs']

        if 's' in params:
            self.startprob_.fill(1.0 / self.n_components)

        if 't' in params or 'm' in params or 'p' in params:
            kmmod = cluster.KMeans(
                n_clusters=self.n_unique,
                random_state=self.random_state).fit(X)
            kmeans = kmmod.cluster_centers_

        if 't' in params:
            # TODO: estimate transitions from data (!) / consider n_tied=1
            if self.n_tied == 0:
                transmat = np.ones([self.n_components, self.n_components])
                np.fill_diagonal(transmat, 10.0)
                self.transmat_ = transmat  # .90 for self-transition
            else:
                transmat = np.zeros((self.n_components, self.n_components))
                transmat[range(self.n_components),
                         range(self.n_components)] = 100.0  # diagonal
                transmat[range(self.n_components-1),
                         range(1, self.n_components)] = 1.0  # diagonal + 1
                transmat[[r * (self.n_chain) - 1
                          for r in range(1, self.n_unique+1)
                          for c in range(self.n_unique-1)],
                         [c * (self.n_chain)
                          for r in range(self.n_unique)
                          for c in range(self.n_unique) if c != r]] = 1.0
                self.transmat_ = np.copy(transmat)

        if 'm' in params:
            mu = np.zeros(self.n_components)
            for u in range(self.n_unique):
                for t in range(self.n_chain):
                    mu[u*(self.n_chain)+t] = kmeans[u, 0]
            self.mu_ = np.copy(mu)

        if 'p' in params:
            self._precision_ = np.zeros(self.n_components)
            if self.tied_precision is True:
                precs = []
            for u in range(self.n_unique):
                for t in range(self.n_chain):
                    if self.tied_precision is False:
                        self._precision_[u*(self.n_chain)+t] = 1.0 / np.var(
                            X[kmmod.labels_ == u])
                    else:
                        precs.extend([1.0 / np.var(X[kmmod.labels_ == u])])
            if self.tied_precision is True:
                self.precision_ = np.array(np.mean(precs)).reshape(-1)

    def _compute_log_likelihood(self, data, from_=0, to_=-1):
        if not hasattr(self, 'auto'):
            # if auto class is not instantiated yet, it will be done from here
            self.auto = auto.THMM(model=self)
            self.auto.init()

        values = {'m': self.mu_,
                  'p': self.precision_}
        values.update({'xn': data['obs'][from_:to_]})

        if self.tied_precision:
            precision = self.precision_[0]
            values.update({'p': precision})

        ll_eval = self.auto.hmm_ll(values)
        rep = self.n_chain
        return np.repeat(ll_eval, rep).reshape(-1, self.n_unique*rep)

    def _do_mstep(self, stats, params):
        if 's' in params:  # update identical to _BaseHMM
            startprob_ = self.startprob_prior - 1.0 + stats['start']
            normalize(startprob_)
            self.startprob_ = np.where(self.startprob_ <= np.finfo(float).eps,
                                       self.startprob_, startprob_)
        if 't' in params:
            if self.n_tied ==0:  # update identical to _BaseHMM
                transmat_ = self.transmat_prior - 1.0 + stats['trans']
                normalize(transmat_, axis=1)
                self.transmat_ = np.where(self.transmat_ <= np.finfo(float).eps,
                                          self.transmat_, transmat_)
            else:
                transmat_ = np.zeros((self.n_components, self.n_components))
                transitionCnts = stats['trans'] + self.transmat_prior
                for b in range(self.n_unique):
                    block = np.zeros((self.n_chain, self.n_components))
                    block_trans = transitionCnts[range(b*self.n_chain,
                                                (b+1)*self.n_chain), :]
                    block_exit_idx = (np.repeat(self.n_chain-1,
                                                self.n_unique-1),
                                      [ix for ix in
                                       range(0, self.n_unique*self.n_chain,
                                             self.n_chain)
                                       if ix != b*self.n_chain])
                    block_chain_idx = (range(0, self.n_chain),
                                       range(b*self.n_chain,
                                             b*self.n_chain+self.n_chain))
                    block_offchain_idx = (range(0, self.n_tied),
                                          range(b*self.n_chain+1,
                                                b*self.n_chain+self.n_chain))
                    block[block_chain_idx] = block_trans[block_chain_idx]
                    block[block_exit_idx] = block_trans[block_exit_idx]
                    block_norm = block.sum()
                    block_chain_sum = block[block_chain_idx].sum()
                    block[block_chain_idx] = block_chain_sum
                    block /= block_norm
                    block[block_offchain_idx] = 1-block[block_chain_idx][0]
                    transmat_[range(b*self.n_chain,
                              (b+1)*self.n_chain), :] = block
                self.transmat_ = np.copy(transmat_)

    def _do_mstep_grad(self, puc, data):
        wrt = [str(p) for p in self.auto.wrt if str(p) in self.params]
        for update_idx in range(self.n_iter_update):
            for p in wrt:
                values = {'m': self.mu_,
                          'p': self.precision_,
                          'mw': self.mu_weight_,
                          'mp': self.mu_prior_,
                          'pw': self.precision_weight_,
                          'pp': self.precision_prior_,
                          'xn': data['obs'],
                          'gn': puc  # posteriors unique concatenated
                         }

                if self.tied_precision:
                    precision = self.precision_[0]
                    values.update({'p': precision})

                result = self.auto.optim(p, values)
                self._set_value(p, result)

    def _process_inputs(self, X):
        return {'obs': X.reshape(-1,1)}

    def fit(self, X, lengths=None):
        """Estimate model parameters.

        An initialization step is performed before entering the
        EM-algorithm. If you want to avoid this step for a subset of
        the parameters, pass proper ``init_params`` keyword argument
        to estimator's constructor.

        Parameters
        ----------
        X : array-like, shape (n_samples, 1)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """
        data = self._process_inputs(X)
        self._do_fit(data, lengths)
        return self

    def _do_fit(self, data, lengths):
        self._init(data, lengths=lengths, params=self.init_params)

        X = data['obs']
        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter,
                                           self.n_iter_min, self.verbose)
        for iter in range(self.n_iter):
            stats = self._initialize_sufficient_statistics()
            puc = np.zeros((X.shape[0], self.n_unique))  # posteriors unique
                                                         # concatenated
            curr_logprob = 0
            for i, j in iter_from_X_lengths(X, lengths):
                framelogprob = self._compute_log_likelihood(data, from_=i,
                                                            to_=j)
                logprob, fwdlattice = self._do_forward_pass(framelogprob)
                curr_logprob += logprob
                bwdlattice = self._do_backward_pass(framelogprob)
                posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
                self._accumulate_sufficient_statistics(
                    stats, X[i:j], framelogprob, posteriors, fwdlattice,
                    bwdlattice, self.params)
                if self.n_tied > 0:
                    for u in range(self.n_unique):
                        cols = range(u*(self.n_chain),
                                     u*(self.n_chain)+(self.n_chain))
                        puc[i:j, u] = np.sum(posteriors[:, cols], axis=1)
                else:
                    puc[i:j, :] = posteriors

            self.monitor_.report(curr_logprob)
            if self.monitor_.converged:
                break

            self._do_mstep(stats, self.params)
            self._do_mstep_grad(puc, data)  # not working with sufficient
                                            # statistics, to have a more
                                            # general framework
        return self

    def score(self, X):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array_like, shape (n)
            Each row corresponds to a single data point.

        Returns
        -------
        logprob : float
            Log likelihood of the ``X``.
        """
        data = self._process_inputs(X)
        return self._do_score(data)

    def _do_score(self, data):
        framelogprob = self._compute_log_likelihood(data)
        logprob, _ = self._do_forward_pass(framelogprob)
        return logprob

    def score_samples(self, X):
        """Compute the log probability under the model and compute posteriors.

        Parameters
        ----------
        X : array_like, shape (n)
            Each row corresponds to a single point in the sequence.

        Returns
        -------
        logprob : float
            Log likelihood of the sequence ``X``

        posteriors : array_like, shape (n, n_components)
            Posterior probabilities of each state for each observation
        """
        data = self._process_inputs(X)
        return self._do_score_samples(data)

    def _do_score_samples(self, data):
        X = data['obs']
        framelogprob = self._compute_log_likelihood(data)
        logprob, fwdlattice = self._do_forward_pass(framelogprob)
        bwdlattice = self._do_backward_pass(framelogprob)
        gamma = fwdlattice + bwdlattice
        # gamma is guaranteed to be correctly normalized by logprob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively.
        posteriors = np.exp(gamma.T - logsumexp(gamma, axis=1)).T
        posteriors += np.finfo(np.float64).eps
        posteriors /= np.sum(posteriors, axis=1).reshape((-1, 1))
        return logprob, posteriors

    def decode(self, X, algorithm="viterbi"):
        """Find most likely state sequence corresponding to ``obs``.
        Uses the selected algorithm for decoding.

        Parameters
        ----------
        X : array_like, shape (n)
            Each row corresponds to a single point in the sequence.

        algorithm : string, one of the ``decoder_algorithms``
            decoder algorithm to be used.
            NOTE: Only Viterbi supported for now.

        Returns
        -------
        logprob : float
            Log probability of the maximum likelihood path through the HMM

        reduced_state_sequence : array_like, shape (n,)
            Index of the most likely states for each observation (accounting
            for tied states by giving them the same index)

        state_sequence : array_like, shape (n,)
            Index of the most likely states for each observation
        """
        data = self._process_inputs(X)
        return self._do_decode(data, algorithm)

    def _do_decode(self, data, algorithm):
        if self.algorithm in decoder_algorithms:
            algorithm = self.algorithm
        elif algorithm in decoder_algorithms:
            algorithm = algorithm
        decoder = {"viterbi": self._decode_viterbi,
                   "map": self._decode_map}
        logprob, reduced_state_sequence, state_sequence = \
            decoder[algorithm](data)
        return logprob, reduced_state_sequence, state_sequence

    def _process_sequence(self, state_sequence):
        """Reduces a state sequence (for tied states), if requested.

        Parameters
        ----------
        state_sequence : array_like, shape (n,)
            Index of the most likely states for each observation.

        Returns
        -------
        reduced_sequence : array_like, shape (n,)
            Index of the most likely states for each observation, treating tied
            states are the same state.
        """
        if self.n_tied == 0:
            return state_sequence

        reduced_sequence = np.zeros(len(state_sequence))

        limits = [u*(self.n_chain) for u in range(self.n_unique+1)]
        for s in range(self.n_unique):
            reduced_sequence[np.logical_and(state_sequence >= limits[s],
                             state_sequence < limits[s+1])] = s

        return reduced_sequence

    def _decode_viterbi(self, data):
        """Find most likely state sequence using the Viterbi algorithm.

        Returns
        -------
        viterbi_logprob : float
            Log probability of the maximum likelihood path through the HMM.

        state_sequence : array_like, shape (n,)
            Index of the most likely states for each observation.
        """
        framelogprob = self._compute_log_likelihood(data)
        viterbi_logprob, state_sequence = self._do_viterbi_pass(framelogprob)
        return viterbi_logprob, self._process_sequence(state_sequence), \
               state_sequence

    def _decode_map(self, data):
        """Find most likely state sequence corresponding using
        maximum a posteriori estimation.

        Returns
        -------
        map_logprob : float
            Log probability of the maximum likelihood path through the HMM

        state_sequence : array_like, shape (n,)
            Index of the most likely states for each observation
        """
        framelogprob = self._compute_log_likelihood(data)
        logprob, fwdlattice = self._do_forward_pass(framelogprob)
        bwdlattice = self._do_backward_pass(framelogprob)
        gamma = fwdlattice + bwdlattice
        # gamma is guaranteed to be correctly normalized by logprob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively.
        posteriors = np.exp(gamma.T - logsumexp(gamma, axis=1)).T
        posteriors += np.finfo(np.float64).eps
        posteriors /= np.sum(posteriors, axis=1).reshape((-1, 1))

        state_sequence = np.argmax(posteriors, axis=1)
        map_logprob = np.max(posteriors, axis=1).sum()
        return map_logprob, self._process_sequence(state_sequence), \
               state_sequence

    def sample(self, n_samples=2000, observed_states=None, random_state=None):
        """Generate random samples from the model.

        Parameters
        ----------
        n : int
            Number of samples to generate.

        observed_states : array
            If provided, states are not sampled.

        random_state: RandomState or an int seed
            A random number generator instance. If None is given, the
            object's random_state is used

        Returns
        -------
        samples : array_like, length (``n_samples``)
                  List of samples

        ustates : array_like, shape (``n_samples``)
                 List of hidden states
        """
        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        samples = np.zeros(n_samples)
        states = np.zeros(n_samples)

        if observed_states is None:
            startprob_pdf = np.exp(np.copy(self._log_startprob))
            startdist = stats.rv_discrete(name='custm',
                                      values=(np.arange(startprob_pdf.shape[0]),
                                                        startprob_pdf),
                                      seed=random_state)
            states[0] = startdist.rvs(size=1)[0]

            transmat_pdf = np.exp(np.copy(self._log_transmat))
            transmat_cdf = np.cumsum(transmat_pdf, 1)

            nrand = random_state.rand(n_samples)
            for idx in range(1,n_samples):
                newstate = (transmat_cdf[states[idx-1]] > nrand[idx-1]).argmax()
                states[idx] = newstate
        else:
            states = observed_states

        mu = np.copy(self._mu_)
        precision = np.copy(self._precision_)
        for idx in range(n_samples):
            mean_ = self._mu_[states[idx]]
            var_ = np.sqrt(1/precision[states[idx]])
            samples[idx] = norm.rvs(loc=mean_, scale=var_, size=1,
                                    random_state=random_state)
        ustates = self._process_sequence(states)
        return samples, ustates

    def _get_mu(self):
        if self.n_tied == 0:
            return self._mu_
        else:
            return self._mu_[[u*(self.n_chain) for u in range(self.n_unique)]]

    def _set_mu(self, mu_val):
        mu = np.zeros(self.n_components)
        if mu_val is not None:
            mu_val = np.asarray(mu_val)
            if len(mu_val) == self.n_components:
                mu = mu_val.copy()
            elif len(mu_val) == self.n_unique:
                for u in range(self.n_unique):
                    for t in range(self.n_chain):
                        mu[u*(self.n_chain)+t] = mu_val[u].copy()
            else:
                raise ValueError("cannot match shape of mu")
        self._mu_ = mu.reshape(-1,1)

    mu_ = property(_get_mu, _set_mu)

    def _get_mu_weight(self):
        return self._mu_weight_

    def _set_mu_weight(self, mu_weight):
        if mu_weight is None:
            self._mu_weight_ = 0.0
        else:
            self._mu_weight_ = float(mu_weight)

    mu_weight_ = property(_get_mu_weight, _set_mu_weight)

    def _get_mu_prior(self):
        if self.n_tied == 0:
            return self._mu_prior_
        else:
            return self._mu_prior_[
                [u*(self.n_chain) for u in range(self.n_unique)]]

    def _set_mu_prior(self, mu_prior):
        if mu_prior is None:
            self._mu_prior_ = np.zeros(self.n_components)
        else:
            mu_prior = np.asarray(mu_prior)
            if len(mu_prior) == self.n_unique:
                self._mu_prior_ = mu_prior.copy()
            else:
                raise ValueError("cannot match shape of mu_prior")

    mu_prior_ = property(_get_mu_prior, _set_mu_prior)

    def _get_precision(self):
        if self.n_tied == 0:
            return self._precision_
        else:
            return self._precision_[
                [u*(self.n_chain) for u in range(self.n_unique)]]

    def _set_precision(self, precision_val):
        if precision_val is None:
            self._precision_ = np.zeros(self.n_components)
        else:
            precision_val = np.asarray(precision_val)
            if self.tied_precision is True:
                if np.max(precision_val) != np.min(precision_val):
                    raise ValueError("elements not equal (precision_val)")
                if len(precision_val) == 1:
                    self._precision_ = np.tile(precision_val,
                                               self.n_components)
                elif len(precision_val) == self.n_unique:
                    self._precision_ = np.tile(precision_val[0],
                                               self.n_components)
                else:
                    raise ValueError("cannot match shape of precision")
            else:
                if len(precision_val) == self.n_components:
                    self._precision_ = precision_val.copy()
                elif len(precision_val) == self.n_unique:
                    self._precision_ = np.zeros(self.n_components)
                    for u in range(self.n_unique):
                        for t in range(self.n_chain):
                            self._precision_[u*(self.n_chain)+t] = precision_val[u].copy()
                else:
                    raise ValueError("cannot match shape of precision")

    precision_ = property(_get_precision, _set_precision)

    def _get_precision_weight(self):
        return self._precision_weight_

    def _set_precision_weight(self, precision_weight):
        if precision_weight is None:
            self._precision_weight_ = 0.0
        else:
            self._precision_weight_ = float(precision_weight)

    precision_weight_ = property(_get_precision_weight, _set_precision_weight)

    def _get_precision_prior(self):
        if self.n_tied == 0:
            return self._precision_prior_
        else:
            return self._precision_prior_[
                [u*(self.n_chain) for u in range(self.n_unique)]]

    def _set_precision_prior(self, precision_prior):
        if precision_prior is None:
            self._precision_prior_ = np.zeros(self.n_components)
        else:
            precision_prior = np.asarray(precision_prior)
            if self.tied_precision is True:
                if np.max(precision_prior) != np.min(precision_prior):
                    raise ValueError("elements not equal (tied_precision)")
                if len(precision_prior) == 1:
                    self._precision_prior_ = np.tile(precision_prior,
                                                     self.n_components)
                elif len(precision_prior) == self.n_unique:
                    self._precision_prior_ = precision_prior.copy()
                else:
                    raise ValueError("cannot match shape of precision_prior")
            else:
                if len(precision_prior) == 1:
                    self._precision_prior_ = np.tile(precision_prior,
                                                     self.n_components)
                elif len(precision_prior) == self.n_unique:
                    self._precision_prior_ = precision_prior.copy()  # TODO: make sure precision prior has dim n_components (!)
                else:
                    raise ValueError("cannot match shape of precision_prior")

    precision_prior_ = property(_get_precision_prior, _set_precision_prior)

    def _get_var(self):
        return 1.0 / self._get_precision()

    def _set_var(self, var_val):
        return self._set_precision(1.0 / var_val)

    var_ = property(_get_var, _set_var)

    def _get_var_weight(self):
        return self._get_precision_weight()

    def _set_var_weight(self, var_weight):
        self._set_precision_weight(var_weight)

    var_weight_ = property(_get_var_weight, _set_var_weight)

    def _get_var_prior(self):
        return 1.0 / self._get_precision_prior()

    def _set_var_prior(self, var_prior):
        var_prior = np.asarray(var_prior)
        self._set_precision_prior(1.0 / var_prior)

    var_prior_ = property(_get_var_prior, _set_var_prior)

    def _get_startprob(self):
        return np.exp(self._log_startprob)

    def _set_startprob(self, startprob):
        if startprob is None:
            startprob = np.tile(1.0 / self.n_components, self.n_components)
        else:
            startprob = np.asarray(startprob, dtype=np.float)

            if not np.alltrue(startprob <= 1.0):
                startprob = normalize(startprob)

            if len(startprob) != self.n_components:
                if len(startprob) == self.n_unique:
                    startprob_split = np.copy(startprob) / (1.0+self.n_tied)
                    startprob = np.zeros(self.n_components)
                    for u in range(self.n_unique):
                        for t in range(self.n_chain):
                            startprob[u*(self.n_chain)+t] = \
                                startprob_split[u].copy()
                else:
                    raise ValueError("cannot match shape of startprob")

        if not np.allclose(np.sum(startprob), 1.0):
            raise ValueError('startprob must sum to 1.0')

        self._log_startprob = np.log(np.asarray(startprob).copy())

    startprob_ = property(_get_startprob, _set_startprob)

    def _get_startprob_prior(self):
        return self.startprob_prior

    def _set_startprob_prior(self, startprob_prior):
        if startprob_prior is None:
            startprob_prior = np.zeros(self.n_components)
        else:
            startprob_prior = np.asarray(startprob_prior, dtype=np.float)

            if len(startprob_prior) != self.n_components:
                if len(startprob_prior) == self.n_unique:
                    startprob_prior_split = np.copy(startprob_prior) / \
                        (1.0 + self.n_tied)
                    startprob_prior = np.zeros(self.n_components)
                    for u in range(self.n_unique):
                        for t in range(self.n_chain):
                            startprob_prior[u*(self.n_chain)+t] = \
                                startprob_prior_split[u].copy()
                else:
                    raise ValueError("cannot match shape of startprob")

        self.startprob_prior = np.asarray(startprob_prior).copy()

    startprob_prior_ = property(_get_startprob_prior, _set_startprob_prior)

    def _ntied_transmat(self, transmat_val):  # TODO: document choices
        transmat = np.empty((0, self.n_components))
        for r in range(self.n_unique):
            row = np.empty((self.n_chain, 0))
            for c in range(self.n_unique):
                if r == c:
                    subm = np.array(sp.diags([transmat_val[r, c],
                                    1 - transmat_val[r, c]], [0, 1],
                                    shape=(self.n_chain,
                                           self.n_chain)).todense())
                else:
                    lower_left = np.zeros((self.n_chain, self.n_chain))
                    lower_left[self.n_tied, 0] = 1.0
                    subm = np.kron(transmat_val[r, c], lower_left)
                row = np.hstack((row, subm))
            transmat = np.vstack((transmat, row))
        return transmat

    def _ntied_transmat_prior(self, transmat_val):  # TODO: document choices
        transmat = np.empty((0, self.n_components))
        for r in range(self.n_unique):
            row = np.empty((self.n_chain, 0))
            for c in range(self.n_unique):
                if r == c:
                    subm = np.array(sp.diags([transmat_val[r, c],
                                    1.0], [0, 1],
                        shape=(self.n_chain, self.n_chain)).todense())
                else:
                    lower_left = np.zeros((self.n_chain, self.n_chain))
                    lower_left[self.n_tied, 0] = 1.0
                    subm = np.kron(transmat_val[r, c], lower_left)
                row = np.hstack((row, subm))
            transmat = np.vstack((transmat, row))
        return transmat

    def _get_transmat(self):
        return np.exp(self._log_transmat)

    def _set_transmat(self, transmat_val):
        if transmat_val is None:
            transmat = np.tile(1.0 / self.n_components,
                               (self.n_components, self.n_components))
        else:
            transmat_val[np.isnan(transmat_val)] = 0.0

            if not np.alltrue(transmat_val <= 1.0):
                transmat_val = normalize(transmat_val, axis=1)

            if (np.asarray(transmat_val).shape == (self.n_components,
                                                   self.n_components)):
                transmat = np.copy(transmat_val)
            elif transmat_val.shape[0] == self.n_unique:
                transmat = self._ntied_transmat(transmat_val)
            else:
                raise ValueError("cannot match shape of transmat")

        if not np.all(np.allclose(np.sum(transmat, axis=1), 1.0)):
            raise ValueError('Rows of transmat must sum to 1.0')

        self._log_transmat = np.log(np.asarray(transmat).copy())
        underflow_idx = np.isnan(self._log_transmat)
        self._log_transmat[underflow_idx] = NEGINF

    transmat_ = property(_get_transmat, _set_transmat)

    def _get_transmat_prior(self):
        return self.transmat_prior

    def _set_transmat_prior(self, transmat_prior):
        if transmat_prior is None or transmat_prior == 1.0:
            transmat_prior = np.zeros((self.n_components, self.n_components))
        else:
            transmat_prior = np.asarray(transmat_prior)
            if transmat_prior.shape != (self.n_components, self.n_components):
                if transmat_prior.shape[0] == self.n_unique:
                    transmat_prior = np.copy(
                        self._ntied_transmat_prior(transmat_prior))
                else:
                    raise ValueError("cannot match shape of transmat")
        self.transmat_prior = np.asarray(transmat_prior).copy()

    transmat_prior_ = property(_get_transmat_prior, _set_transmat_prior)

    def _set_value(self, param, value):
        if param == 'm':
            self.mu_ = value
        elif param == 'p':
            self.precision_ = value
        else:
            raise ValueError('unknown parameter')


# Helper classes

class ConvergenceMonitor(object):
    """Monitors and reports convergence to :data:`sys.stderr`.

    Parameters
    ----------
    tol : double
        Convergence threshold. EM has converged either if the maximum
        number of iterations is reached or the log probability
        improvement between the two consecutive iterations is less
        than threshold.
    n_iter : int
        Maximum number of iterations to perform.
    verbose : bool
        If ``True`` then per-iteration convergence reports are printed,
        otherwise the monitor is mute.

    Attributes
    ----------
    history : deque
        The log probability of the data for the last two training
        iterations. If the values are not strictly increasing, the
        model did not converge.
    iter : int
        Number of iterations performed while training the model.

    Note
    ----
    The convergence monitor is adapted from hmmlearn.base.
    """
    fmt = "{iter:>10d} {logprob:>16.4f} {delta:>+16.4f}"

    def __init__(self, tol, n_iter, n_iter_min, verbose):
        self.tol = tol
        self.n_iter = n_iter
        self.n_iter_min = n_iter_min
        self.verbose = verbose
        self.history = deque(maxlen=2)
        self.iter = 1

    def __repr__(self):
        class_name = self.__class__.__name__
        params = dict(vars(self), history=list(self.history))
        return "{0}({1})".format(
            class_name, _pprint(params, offset=len(class_name)))

    def report(self, logprob):
        """Reports the log probability of the next iteration."""
        if self.history and self.verbose:
            delta = logprob - self.history[-1]
            message = self.fmt.format(
                iter=self.iter, logprob=logprob, delta=delta)
            print(message, file=sys.stderr)

        self.history.append(logprob)
        self.iter += 1

    @property
    def converged(self):
        """``True`` if the EM-algorithm converged and ``False`` otherwise."""
        has_converged = False
        if self.iter < self.n_iter_min:
            return has_converged
        if len(self.history) == 2:
            diff = self.history[1] - self.history[0]
            absdiff = abs(diff)
            if diff < 0:
                if self.verbose:
                    print('Warning: LL did decrease', file=sys.stderr)
                    has_converged = True
            if absdiff < self.tol:
                if self.verbose:
                    print('Converged, |difference| is: {}'.format(absdiff))
                has_converged = True
        if self.iter == self.n_iter:
            if self.verbose:
                print('Warning: Maximum iterations reached', file=sys.stderr)
            has_converged = True
        return has_converged

class Timer(object):
    """Helper class to time performance"""
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tenter = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('{}: '.format(self.name))
        print('Elapsed: {}'.format((time.time() - self.tenter)))


# Helper functions

def precision_prior_params(mean_gamma_prior, var_gamma_prior):
    """Returns ``weight`` and ``prior`` for a gamma prior with mean and var"""
    # mean: alpha / beta, var: alpha / beta**2
    beta = mean_gamma_prior / var_gamma_prior
    alpha = mean_gamma_prior * beta
    weight = alpha
    prior = np.array((beta,beta))
    return weight, prior

def sequence_to_rects(seq=None, y=-5, height=10,
                      colors = ['0.2','0.4', '0.6', '0.7']):
    """Transforms a state sequence to rects for plotting with matplotlib.

    Parameters
    ----------
    seq : array
        state sequence
    y : int
        lower left corner
    height: int
        height
    colors : array
        array of label colors

    Returns
    -------
    rects : dict
         .xy : tuple
             (x,y) tuple specifying the lower left
         .width: int
             width of rect
         .height : int
             height of rect
         .label : int
             state label
         .color : string
             color string
    """
    y_ = y
    height_ = height
    label_ = seq[0]
    x_ = -0.5
    width_ = 1.0
    rects = []
    for s in range(1,len(seq)):
        if seq[s] != seq[s-1] or s == len(seq)-1:
            rects.append({'xy': (x_, y_),
                          'width': width_,
                          'height': height_,
                          'label': int(label_),
                          'color': colors[int(label_)]})
            x_ = s-0.5
            width_ = 1.0
            label_ = seq[s]
        else:
            if s == len(seq)-2:
                width_ += 2.0
            else:
                width_ += 1.0
    return rects
