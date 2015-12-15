from __future__ import division, print_function, absolute_import

import string
import warnings

import numpy as np

from scipy import stats
from scipy.stats import norm
from scipy.linalg import toeplitz, pinv

from sklearn import cluster
from sklearn.utils import check_random_state

import theano.tensor as tt
from theano.tensor import addbroadcast as bc

import statsmodels.api as smapi
from statsmodels.tsa.tsatools import lagmat

from .tm import THMM

__all__ = ['ARTHMM']

decoder_algorithms = frozenset(("viterbi", "map"))

class ARTHMM(THMM):
    """Hidden Markov Model with tied states and autoregressive observations

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

    n_lags : int
        Number of lags (order of AR).

    tied_alpha : bool
        If set to true, alpha is shared across states.

    alpha_init : array, shape (``n_components``, ``n_lags``)
        Initial alpha parameter per state.

    mu_bounds : array, shape (``2``)
        Upper and lower bound for mu [lower bound, upper bound].

    precision_bounds : array, shape (``2``)
        Upper and lower bound for precision [lower bound, upper bound].

    alpha_bounds : array, shape(``2``)
        Upper and lower bound for alpha [lower bound, upper bound].

    Attributes
    ----------
    n_components : int
        Number of total components

    mu_ : array, shape (``n_unique``)

    precision_ : array, shape (``n_unique``)

    transmat_ :  array, shape (``n_unique``, ``n_unique``)

    startprob_ :  array, shape (``n_unique``, ``n_unique``)

    n_lags : int

    n_inputs : int

    alpha_ : array, shape (``n_components``, ``n_lags``)
    """
    def __init__(self, n_unique=2, n_lags=0, n_tied=0,
                 startprob_init=None, transmat_init=None, startprob_prior=1.0,
                 transmat_prior=1.0, algorithm="viterbi", random_state=None,
                 n_iter=25, n_iter_min=2, tol=1e-4,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters, alpha_init=None,
                 mu_init=None, precision_init=None,
                 precision_prior=None, precision_weight=0.0, mu_prior=None,
                 mu_weight=0.0, tied_alpha=True,
                 tied_precision=False, n_iter_update=1, verbose=False,
                 mu_bounds=np.array([-50.0, 50.0]),
                 precision_bounds=np.array([0.001, 10000.0]),
                 alpha_bounds=np.array([-10.0, 10.0])):
        super(ARTHMM, self).__init__(n_unique=n_unique, n_tied=n_tied,
                                     tied_precision=tied_precision,
                                     algorithm=algorithm,
                                     params=params, init_params=init_params,
                                     startprob_init=startprob_init,
                                     startprob_prior=startprob_prior,
                                     transmat_init=transmat_init,
                                     transmat_prior=transmat_prior,
                                     mu_init=mu_init, mu_weight=mu_weight,
                                     mu_prior=mu_prior,
                                     precision_init=precision_init,
                                     precision_weight=precision_weight,
                                     precision_prior=precision_prior, tol=tol,
                                     n_iter=n_iter, n_iter_min=n_iter_min,
                                     n_iter_update=n_iter_update,
                                     random_state=random_state,
                                     verbose=verbose,
                                     mu_bounds=mu_bounds,
                                     precision_bounds=precision_bounds)
        self.alpha_ = alpha_init
        self.tied_alpha = tied_alpha

        self.n_lags = n_lags

        self.alpha_bounds = alpha_bounds

        if self.n_lags > 0:
            self.xln = tt.dmatrix('xln')  # N x n_lags
            self.a = tt.dmatrix('a')

            self.inputs_hmm_ll.extend([self.xln, self.a])
            self.inputs_neg_ll.extend([self.xln, self.a])
            self.wrt.extend([self.a])
            if not self.tied_alpha:
                self.wrt_dims.update({'a': (self.n_unique, self.n_lags)})
            else:
                self.wrt_dims.update({'a': (1, self.n_lags)})
            self.wrt_bounds.update({'a': (self.alpha_bounds[0], self.alpha_bounds[1])})

            if not self.tied_alpha:
                self.hmm_mean = self.hmm_mean + tt.dot(self.xln, self.a.T)
            else:
                self.hmm_mean = self.hmm_mean + bc(tt.dot(self.xln, self.a.T),1)

    # hmm_ll_, hmm_ell_, and neg_ll_ as in THMM

    def _compute_log_likelihood(self, data, from_=0, to_=-1):
        if self.compiled == False:  # check if Theano functions are compiled
            self._compile()

        values = {'m': self.mu_,
                  'p': self.precision_}
        values.update({'xn': data['obs'][from_:to_]})

        if self.tied_precision:
            precision = self.precision_[0]
            values.update({'p': precision})

        if self.n_lags > 0:
            if not self.tied_alpha:
                alpha = self.alpha_
            else:
                alpha = self.alpha_[0,:]
            values.update({'a': alpha,
                           'xln': data['lagged'][from_:to_]})

        ll_eval = self._eval_hmm_ll(values)
        rep = self.n_tied+1
        return np.repeat(ll_eval, rep).reshape(-1, self.n_unique*rep)

    def _do_mstep_grad(self, puc, data):
        wrt = [str(p) for p in self.wrt if str(p) in \
               self.params]
        for update_idx in range(self.n_iter_update):
            for p in wrt:
                values = {'m': self.mu_,
                          'p': self.precision_,
                          'mw': self.mu_weight_,
                          'mp': self.mu_prior_,
                          'pw': self.precision_weight_,
                          'pp': self.precision_prior_,
                          'xn': data['obs'],
                          'gn': puc
                         }

                if self.tied_precision:
                    precision = self.precision_[0]
                    values.update({'p': precision})

                if self.n_lags > 0:
                    if not self.tied_alpha:
                        alpha = self.alpha_
                    else:
                        alpha = self.alpha_[0,:]
                    values.update({'a': alpha,
                                   'xln': data['lagged']})

                result = self._optim(p, values)

                if p == 'm':
                    self.mu_ = result
                elif p == 'p':
                    self.precision_ = result
                elif p == 'a':
                    self.alpha_ = result
                else:
                    raise ValueError('unknown parameter')

    def _init_params(self, data, lengths=None, params='stmpaw'):
        X = data['obs']

        if self.n_lags == 0:
            super(ARTHMM, self)._init_params(data, lengths, params)
        else:
            if 's' in params:
                super(ARTHMM, self)._init_params(data, lengths, 's')

            if 't' in params:
                super(ARTHMM, self)._init_params(data, lengths, 't')

            if 'm' in params or 'a' in params or 'p' in params:
                kmmod = cluster.KMeans(
                    n_clusters=self.n_unique,
                    random_state=self.random_state).fit(X)
                kmeans = kmmod.cluster_centers_
                ar_mod = []
                ar_alpha = []
                ar_resid = []
                if not self.tied_alpha:
                    for u in range(self.n_unique):
                        ar_mod.append(smapi.tsa.AR(X[kmmod.labels_ == \
                                                u]).fit(self.n_lags))
                        ar_alpha.append(ar_mod[u].params[1:])
                        ar_resid.append(ar_mod[u].resid)
                else:
                    # run one AR model on most part of time series
                    # that has most points assigned after clustering
                    mf = np.argmax(np.bincount(kmmod.labels_))
                    ar_mod.append(smapi.tsa.AR(X[kmmod.labels_ == \
                                              mf]).fit(self.n_lags))
                    ar_alpha.append(ar_mod[0].params[1:])
                    ar_resid.append(ar_mod[0].resid)

            if 'm' in params:
                self._mu_ = np.zeros(self.n_components)
                for u in range(self.n_unique):
                    for t in range(1+self.n_tied):
                        ar_idx = u
                        if self.tied_alpha:
                            ar_idx = 0
                        self._mu_[u*(1+self.n_tied)+t] = kmeans[u, 0] - np.dot(
                            np.repeat(kmeans[u, 0], self.n_lags),
                            ar_alpha[ar_idx])

            if 'p' in params:
                self._precision_ = np.zeros(self.n_components)
                for u in range(self.n_unique):
                    for t in range(1+self.n_tied):
                        if not self.tied_alpha:
                            maxVar = np.max([np.var(ar_resid[i]) for i in
                                            range(self.n_unique)])
                        else:
                            maxVar = np.var(ar_resid[0])
                        self._precision_[u*(1+self.n_tied)+t] = 1.0 / maxVar

            if 'a' in params:
                self._alpha_ = np.zeros((self.n_components, self.n_lags))
                for u in range(self.n_unique):
                    for t in range(1+self.n_tied):
                        ar_idx = u
                        if self.tied_alpha:
                            ar_idx = 0
                        self._alpha_[u*(1+self.n_tied)+t, :] = \
                            ar_alpha[ar_idx]

    def _process_inputs(self, X, E=None, lengths=None):
        # Makes sure inputs have correct shape, generates features
        lagged = None
        if lengths is None:
            lagged = lagmat(X, maxlag=self.n_lags, trim='forward',
                            original='ex')
        else:
            lagged = np.zeros((len(X), self.n_lags))
            for i, j in iter_from_X_lengths(X, lengths):
                lagged[i:j, :] = lagmat(X[i:j], maxlag=self.n_lags,
                                        trim='forward', original='ex')

        inputs = {'obs': X.reshape(-1,1),
                  'lagged': lagged}
        return inputs

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
            Returns model.
        """
        data = self._process_inputs(X)
        self._do_fit(data, lengths)
        return self

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
        logprob, posteriors =  self._do_score_samples(data)
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

        state_sequence : array_like, shape (n,)
            Index of the most likely states for each observation (accounting
            for tied states by giving them the same index)
        """
        data = self._process_inputs(X)
        logprob, state_sequence = self._do_decode(data, algorithm)
        return logprob, self._process_sequence(state_sequence)

    def sample(self, n_samples=2000, observed_states=None,
               init_samples=None, init_state=None, random_state=None):
        """Generate random samples from the self.

        Parameters
        ----------
        n : int
            Number of samples to generate.

        observed_states : array
            If provided, states are not sampled.

        random_state: RandomState or an int seed
            A random number generator instance. If None is given, the
            object's random_state is used

        init_state : int
            If provided, initial state is not sampled.

        init_samples : array, default: None
            If provided, initial samples (for AR) are not sampled.

        E : array-like, shape (n_samples, n_inputs)
            Feature matrix of individual inputs.

        Returns
        -------
        samples : array_like, length (``n_samples``)
                  List of samples

        states : array_like, shape (``n_samples``)
                 List of hidden states (accounting for tied states by giving
                 them the same index)
        """
        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        samples = np.zeros(n_samples)
        states = np.zeros(n_samples)

        order = self.n_lags

        if init_state is None:
            startprob_pdf = np.exp(np.copy(self._log_startprob))
            start_dist = stats.rv_discrete(name='custm',
                                      values=(np.arange(startprob_pdf.shape[0]),
                                                        startprob_pdf),
                                      seed=random_state)
            start_state = start_dist.rvs(size=1)[0]
        else:
            start_state = init_state

        if self.n_lags > 0:
            if init_samples is None:
                n_init_samples = order + 10
                noise = np.sqrt(1.0/self._precision_[start_state]) * \
                        random_state.randn(n_init_samples)

                pad_after = n_init_samples - order - 1
                col = np.pad(1*self._alpha_[start_state, :], (1, pad_after),
                             mode='constant')
                row = np.zeros(n_init_samples)
                col[0] = row[0] = 1

                A = toeplitz(col, row)
                init_samples = np.dot(pinv(A), noise + self._mu_[start_state])
                # TODO: fix bug with n_lags > 1, blows up
                init_samples = 0.01*np.ones((len(init_samples)))  # temporary fix

        if observed_states is None:
            transmat_pdf = np.exp(np.copy(self._log_transmat))
            transmat_cdf = np.cumsum(transmat_pdf, 1)

            states[0] = (transmat_cdf[start_state] >
                         random_state.rand()).argmax()

            transmat_pdf = np.exp(self._log_transmat)
            transmat_cdf = np.cumsum(transmat_pdf, 1)

            nrand = random_state.rand(n_samples)
            for idx in range(1,n_samples):
                newstate = (transmat_cdf[states[idx-1]] > nrand[idx-1]).argmax()
                states[idx] = newstate
        else:
            states = observed_states

        precision = np.copy(self._precision_)
        for idx in range(n_samples):
            state_ = states[idx]
            var_ = np.sqrt(1/precision[state_])

            if self.n_lags == 0:
                mean_ = np.copy(self._mu_[state_])
            else:
                mean_ = np.copy(self._mu_[state_])

                for lag in range(1, order+1):
                    if idx < lag:
                        prev_ = init_samples[len(init_samples)-lag]
                    else:
                        prev_ = samples[idx-lag]
                    mean_ += np.copy(self._alpha_[state_, lag-1])*prev_

            samples[idx] = norm.rvs(loc=mean_, scale=var_, size=1,
                                    random_state=random_state)

        states = self._process_sequence(states)
        return samples, states

    def _get_alpha(self):
        if self.n_tied == 0:
            return self._alpha_
        else:
            return self._alpha_[
                [u*(1+self.n_tied) for u in range(self.n_unique)]]

    def _set_alpha(self, alpha_val):
        if alpha_val is None or self.n_lags == 0:
            self._alpha_ = np.zeros((self.n_components, 1))  # TODO: check
        else:
            alpha_val = np.atleast_2d(np.asarray(alpha_val))
            if alpha_val.shape[1] != self.n_lags:
                raise ValueError("shape does not match n_lags")
            if self.tied_alpha == True:
                if not (alpha_val == alpha_val[0]).all():
                    raise ValueError("rows are not identical (tied_alpha)")
            if alpha_val.shape[0] == 1:
                self._alpha_ = np.zeros((self.n_components, self.n_lags))
                for k in range(self.n_components):
                    self._alpha_[k,:] = alpha_val
            elif alpha_val.shape[0] == self.n_components:
                self._alpha_ = alpha_val.copy()
            elif alpha_val.shape[0] == self.n_unique:
                self._alpha_ = np.zeros((self.n_components, self.n_lags))
                for u in range(self.n_unique):
                    for t in range(1+self.n_tied):
                        self._alpha_[u*(1+self.n_tied)+t] = alpha_val[u, :].copy()
            else:
                raise ValueError("cannot match shape of alpha")

    alpha_ = property(_get_alpha, _set_alpha)

    def _get_ar_mean(self):
        # Calculates AR mean
        if self.n_lags == 0:
            raise NotImplementedError('not an autoregressive model')
        if self.n_inputs > 0:
            warnings.warn("additional dependency on input vector",
                          RuntimeWarning)
        means = np.zeros(self.n_unique)
        for u in range(self.n_unique):
            num = self.mu_[u]
            denom = 1
            denom -= np.sum(self.alpha_[u])
            means[u] = num / denom
        return means

    ar_mean_ = property(_get_ar_mean)
