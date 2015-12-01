from __future__ import division, print_function, absolute_import

import string
import warnings

import numpy as np

from scipy import stats
from scipy.stats import norm
from scipy.linalg import toeplitz, pinv

from sklearn import cluster
from sklearn.utils import check_random_state

import statsmodels.api as smapi
from statsmodels.tsa.tsatools import lagmat

from . import auto
from .tm import THMM

__all__ = ['ARTHMM']

decoder_algorithms = frozenset(("viterbi", "map"))

class ARTHMM(THMM):
    """Hidden Markov Model with Autoregressive Observations and External Inputs

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

    tied_omega : bool
        If set to true, omega is shared across states.

    alpha_init : array, shape (``n_components``, ``n_lags``)
        Initial alpha parameter per state.

    omega_init : array, shape (``n_components``, ``n_inputs``)
        Initial omega parameter per state.

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

    omega_ : array, shape (``n_components``, ``n_inputs``)
    """
    def __init__(self, n_unique=2, n_lags=0, n_tied=0,
                 startprob_init=None, transmat_init=None, startprob_prior=1.0,
                 transmat_prior=1.0, algorithm="viterbi", random_state=None,
                 n_iter=25, n_iter_min=2, tol=1e-4,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters, alpha_init=None,
                 omega_init=None, mu_init=None, precision_init=None,
                 precision_prior=None, precision_weight=0.0, mu_prior=None,
                 mu_weight=0.0, tied_alpha=True, tied_omega=False,
                 tied_precision=False, n_iter_update=1, verbose=False):
        THMM.__init__(self, n_unique=n_unique, n_tied=n_tied,
                      tied_precision=tied_precision,
                      algorithm=algorithm,
                      params=params, init_params=init_params,
                      startprob_init=startprob_init,
                      startprob_prior=startprob_prior,
                      transmat_init=transmat_init,
                      transmat_prior=transmat_prior,
                      mu_init=mu_init, mu_weight=mu_weight,
                      mu_prior=mu_prior, precision_init=precision_init,
                      precision_weight=precision_weight,
                      precision_prior=precision_prior, tol=tol,
                      n_iter=n_iter, n_iter_min=n_iter_min,
                      n_iter_update=n_iter_update,
                      random_state=random_state, verbose=verbose)
        self.alpha_ = alpha_init
        self.omega_ = omega_init

        self.tied_alpha = tied_alpha
        self.tied_omega = tied_omega

        if tied_omega:
            raise ValueError('not implemented')  # TODO: implement

        self.n_lags = n_lags
        self.n_inputs = 0  # will be reset once data is passed

    def _init(self, data, lengths=None, params='stmpaw'):
        X = data['obs']
        E = data['external']

        # TODO: subtract external input before estimating parameters
        # TODO: come up with how to initialize omegas

        if self.n_lags == 0:
            super(ARTHMM, self)._init(data, lengths, params)
        else:
            if 's' in params:
                super(ARTHMM, self)._init(data, lengths, 's')

            if 't' in params:
                super(ARTHMM, self)._init(data, lengths, 't')

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

            """
            if 'w' in params:
                # TODO: test omega init on toy problem
                self._omega_ = np.zeros((self.n_components, self.n_inputs))
                if self.n_inputs > 0:
                    for i in range(self.n_inputs):
                        for u in range(self.n_unique):
                            for t in range(1+self.n_tied):
                                self._omega_[u*(1+self.n_tied)+t, i] = \
                                    inputs_concat[kmmod.labels_ == u, i] - \
                                    obs_concat[kmmod.labels_ == u]
            """

    def _compute_log_likelihood(self, data, from_=0, to_=-1):
        if not hasattr(self, 'auto'):
            # if auto class is not instantiated yet, it will be done from here
            self.auto = auto.ARTHMM(model=self)
            self.auto.init()

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

        if self.n_inputs > 0:
            if not self.tied_omega:
                omega = self.omega_
            else:
                omega = self.omega_[0,:]
            values.update({'w': omega,
                           'fn': data['external'][from_:to_]})

        ll_eval = self.auto.hmm_ll(values)
        rep = self.n_tied+1
        return np.repeat(ll_eval, rep).reshape(-1, self.n_unique*rep)

    def _do_mstep_grad(self, puc, data):
        wrt = [str(p) for p in self.auto.wrt if str(p) in \
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

                if self.n_inputs > 0:
                    if not self.tied_omega:
                        omega = self.omega_
                    else:
                        omega = self.omega_[0,:]
                    values.update({'w': omega,
                                   'fn': data['external']})

                result = self.auto.optim(p, values)
                self._set_value(p, result)

    def _process_inputs(self, X, E=None, lengths=None):
        lagged = None
        if lengths is None:
            lagged = lagmat(X, maxlag=self.n_lags, trim='forward',
                            original='ex')
        else:
            lagged = np.zeros((len(X), self.n_lags))
            for i, j in iter_from_X_lengths(X, lengths):
                lagged[i:j, :] = lagmat(X[i:j], maxlag=self.n_lags,
                                        trim='forward', original='ex')

        external = None
        if E is not None:
            raise ValueError("handling of external inputs not yet implemented")
            # TODO: implement
            # TODO: set self.n_inputs

        inputs = {'obs': X.reshape(-1,1),
                  'lagged': lagged,
                  'external': external}
        return inputs

    def fit(self, X, E=None, lengths=None):
        """Estimate model parameters.

        An initialization step is performed before entering the
        EM-algorithm. If you want to avoid this step for a subset of
        the parameters, pass proper ``init_params`` keyword argument
        to estimator's constructor.


        Parameters
        ----------
        X : array-like, shape (n_samples, 1)
            Feature matrix of individual samples.

        E : array-like, shape (n_samples, n_inputs)
            Feature matrix of individual inputs.

        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """
        # TODO: account for external input
        data = self._process_inputs(X, E)
        self._do_fit(data, lengths)
        return self

    def score(self, X, E=None):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array_like, shape (n)
            Each row corresponds to a single data point.

        E : array-like, shape (n_samples, n_inputs)
            Feature matrix of individual inputs.

        Returns
        -------
        logprob : float
            Log likelihood of the ``X``.
        """        # TODO: account for external input
        data = self._process_inputs(X, E)
        return self._do_score(data)

    def score_samples(self, X, E=None):
        """Compute the log probability under the model and compute posteriors.

        Parameters
        ----------
        X : array_like, shape (n)
            Each row corresponds to a single point in the sequence.

        E : array-like, shape (n_samples, n_inputs)
            Feature matrix of individual inputs.

        Returns
        -------
        logprob : float
            Log likelihood of the sequence ``X``

        posteriors : array_like, shape (n, n_components)
            Posterior probabilities of each state for each observation
        """
        # TODO: account for external input
        data = self._process_inputs(X, E)
        return self._do_score_samples(data)

    def decode(self, X, E=None, algorithm="viterbi"):
        """Find most likely state sequence corresponding to ``obs``.
        Uses the selected algorithm for decoding.

        Parameters
        ----------
        X : array_like, shape (n)
            Each row corresponds to a single point in the sequence.

        E : array-like, shape (n_samples, n_inputs)
            Feature matrix of individual inputs.

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
        # TODO: account for external input
        data = self._process_inputs(X, E)
        return self._do_decode(data, algorithm)

    def sample(self, n_samples=2000, observed_states=None,
               init_samples=None, init_state=None, E=None, random_state=None):
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

        ustates : array_like, shape (``n_samples``)
                 List of hidden states
        """
        # TODO: account for external input
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
                #import pdb; pdb.set_trace()

                for lag in range(1, order+1):
                    if idx < lag:
                        prev_ = init_samples[len(init_samples)-lag]
                    else:
                        prev_ = samples[idx-lag]
                    mean_ += np.copy(self._alpha_[state_, lag-1])*prev_

            samples[idx] = norm.rvs(loc=mean_, scale=var_, size=1,
                                    random_state=random_state)

        ustates = self._process_sequence(states)
        return samples, ustates

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

    def _get_omega(self):
        if self.n_tied == 0:
            return self._omega_
        else:
            return self._omega_[
                [u*(1+self.n_tied) for u in range(self.n_unique)]]

    def _set_omega(self, omega):
        if omega is None or self.n_inputs == 0:
            self._omega_ = np.zeros((self.n_components, 1))  # TODO: check
        else:
            omega = np.asarray(omega)  # TODO: tied_omega
            if len(omega) == self.n_components:
                self._omega_ = omega.copy()
            else:
                raise ValueError("cannot match shape of omega")

    omega_ = property(_get_omega, _set_omega)

    def _set_value(self, param, value):
        if param == 'm':
            self.mu_ = value
        elif param == 'p':
            self.precision_ = value
        elif param == 'a':
            self.alpha_ = value
        elif param == 'w':
            self.omega_ = value
        else:
            raise ValueError('unknown parameter')

    def _get_ar_mean(self):
        """Calculate AR mean
        """
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
