from __future__ import division, print_function, absolute_import

import string
import warnings

from scipy import stats
from scipy.stats import norm
from scipy.linalg import toeplitz, pinv

from sklearn import cluster
from sklearn.utils import check_random_state

import autograd.numpy as np
from autograd import grad, value_and_grad
from scipy.optimize import minimize

import statsmodels.api as smapi
from statsmodels.tsa.tsatools import lagmat

from .tm import THMM

__all__ = ['ARTHMM']

ZEROLOGPROB = -1e200
EPS = np.finfo(float).eps
NEGINF = -np.inf

decoder_algorithms = frozenset(("viterbi", "map"))

class ARTHMM(THMM):
    """Hidden Markov Model with tied states and autoregressive observations

    Parameters
    ----------
    n_unique : int
        Number of unique components.

    n_tied : int
        Number of tied states for each component.

    n_features : int
        Number of features.

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

    mu_init : array, shape (``n_unique``, ``n_features``)
        Initial mean parameters for each state.

    mu_weight : int
        Weight of mu prior, shared across components.

    mu_prior : array, shape (``n_unique``, ``n_features``)
        Prior on mu.

    precision_init : array, shape (``n_unique``, ``n_features``, ``n_features``)
        Initial precision (inverse variance) parameters for each state.

    precision_weight : int
        Weight of precision (inverse variance) prior.

    precision_prior : array, shape (``n_unique``, ``n_features``, ``n_features``)
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

    shared_alpha : bool
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

    mu_ : array, shape (``n_unique``, ``n_features``)

    precision_ : array, shape (``n_unique``, ``n_features``, ``n_features``)

    transmat_ :  array, shape (``n_unique``, ``n_unique``)

    startprob_ :  array, shape (``n_unique``, ``n_unique``)

    n_lags : int

    n_inputs : int

    alpha_ : array, shape (``n_components``, ``n_lags``)
    """
    def __init__(self, n_unique=2, n_lags=1, n_tied=0, n_features=1,
                 startprob_init=None, transmat_init=None, startprob_prior=1.0,
                 transmat_prior=None, algorithm="viterbi", random_state=None,
                 n_iter=25, n_iter_min=2, tol=1e-4,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters, alpha_init=None,
                 mu_init=None, precision_init=None,
                 precision_prior=None, precision_weight=0.0, mu_prior=None,
                 mu_weight=0.0, shared_alpha=True,
                 n_iter_update=1, verbose=False,
                 mu_bounds=np.array([-1.0e5, 1.0e5]),
                 precision_bounds=np.array([-1.0e5, 1.0e5]),
                 alpha_bounds=np.array([-1.0e5, 1.0e5])):
        super(ARTHMM, self).__init__(n_unique=n_unique, n_tied=n_tied,
                                     n_features=n_features,
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
        self.n_lags = n_lags
        if self.n_lags < 1:
            raise ValueError("n_lags needs to be greater than 0")

        self.shared_alpha = shared_alpha
        self.alpha_ = alpha_init
        self.alpha_bounds = alpha_bounds

        self.wrt.extend(['a'])
        if not self.shared_alpha:
            self.wrt_dims.update({'a': (self.n_unique, self.n_lags)})
        else:
            self.wrt_dims.update({'a': (1, self.n_lags)})
        self.wrt_bounds.update({'a': (self.alpha_bounds[0],
                                      self.alpha_bounds[1])})

    def _compute_log_likelihood(self, data, from_=0, to_=-1):
        ll = self._ll(self.mu_, self.precision_, self.alpha_,
                      data['obs'][from_:to_],
                      data['lagged'][from_:to_])
        return ll

    def _ll(self, m, p, a, xn, xln, **kwargs):
        """Computation of log likelihood

        Dimensions
        ----------
        m :  n_unique x n_features
        p :  n_unique x n_features x n_features
        a :  n_unique x n_lags (shared_alpha=F)
             OR     1 x n_lags (shared_alpha=T)
        xn:  N x n_features
        xln: N x n_features x n_lags
        """

        n_samples = xn.shape[0]
        xn = xn.reshape(n_samples, self.n_features, 1)
        m = m.reshape(1, self.n_features, self.n_unique)

        det = np.linalg.det(np.linalg.inv(p))
        det = det.reshape(1, self.n_unique)

        xm = xn-(np.dot(xln, a.T) + m)
        tem = np.einsum('NFU,UFX,NXU->NU', xm, p, xm)

        res = (-self.n_features/2.0)*np.log(2*np.pi) - 0.5*tem - 0.5*np.log(det)

        return res

    def _obj(self, m, p, a, xn, xln, gn, **kwargs):
        ll = self._ll(m, p, a, xn, xln)

        mw = self.mu_weight_
        mp = self.mu_prior_
        pw = self.precision_weight_
        pp = self.precision_prior_
        prior = (pw-0.5) * np.log(p) - 0.5*p*(mw*(m-mp)**2 + 2*pp)

        res = -1*(np.sum(gn * ll) + np.sum(prior))
        return res

    def _obj_grad(self, wrt, m, p, a, xn, xln, gn, **kwargs):
        res = grad(self._obj, wrt)(m, p, a, xn, xln, gn)
        res = np.array([res])
        return res

    def _do_mstep_grad(self, puc, data):
        wrt = [str(p) for p in self.wrt if str(p) in self.params]
        for update_idx in range(self.n_iter_update):
            for p in wrt:
                if p == 'm':
                    optim_x0 = self.mu_
                    wrt_arg = 0
                elif p == 'p':
                    optim_x0 = self.precision_
                    wrt_arg = 1
                elif p == 'a':
                    optim_x0 = self.alpha_
                    wrt_arg = 2
                else:
                    raise ValueError('unknown parameter')

                optim_bounds = [self.wrt_bounds[p] for k in
                                range(np.prod(self.wrt_dims[p]))]
                result = minimize(fun=self._optim_wrap, jac=True,
                                  x0=np.array(optim_x0).reshape(-1),
                                  args=(p,
                                        {'wrt': wrt_arg,
                                         'p': self.precision_,
                                         'm': self.mu_,
                                         'a': self.alpha_,
                                         'xn': data['obs'],
                                         'xln': data['lagged'],
                                         'gn': puc  # post. uni. concat.
                                        }),
                                  bounds=optim_bounds,
                                  method='TNC')

                newv = result.x.reshape(self.wrt_dims[p])
                if p == 'm':
                    self.mu_ = newv
                elif p == 'p':
                    self.precision_ = newv
                elif p == 'a':
                    self.alpha_ = newv
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

            if 'm' in params or 'a' in params or 'p' in params:  # TODO: init for n_features > 1
                kmmod = cluster.KMeans(
                    n_clusters=self.n_unique,
                    random_state=self.random_state).fit(X)
                kmeans = kmmod.cluster_centers_
                ar_mod = []
                ar_alpha = []
                ar_resid = []
                if not self.shared_alpha:
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

            if 'm' in params:  # TODO: init for n_features > 1
                mu_init = np.zeros((self.n_unique, self.n_features))
                for u in range(self.n_unique):
                    ar_idx = u
                    if self.shared_alpha:
                        ar_idx = 0
                    mu_init[u] = kmeans[u, 0] - np.dot(
                            np.repeat(kmeans[u, 0], self.n_lags),
                            ar_alpha[ar_idx])
                self.mu_ = np.copy(mu_init)

            if 'p' in params:  # TODO: init for n_features > 1
                precision_init = np.zeros((self.n_unique, self.n_features))
                for u in range(self.n_unique):
                    if not self.shared_alpha:
                        maxVar = np.max([np.var(ar_resid[i]) for i in
                                        range(self.n_unique)])
                    else:
                        maxVar = np.var(ar_resid[0])
                    precision_init[u] = 1.0 / maxVar
                self.precision_ = np.copy(precision_init)

            if 'a' in params:  # TODO: init for n_features > 1
                alpha_init = np.zeros((self.n_unique, self.n_lags))
                for u in range(self.n_unique):
                    ar_idx = u
                    if self.shared_alpha:
                        ar_idx = 0
                    alpha_init[u, :] = ar_alpha[ar_idx]
                self.alpha_ = alpha_init

    def _process_inputs(self, X, E=None, lengths=None):
        if self.n_features == 1:
            lagged = None
            if lengths is None:
                lagged = lagmat(X, maxlag=self.n_lags, trim='forward',
                                original='ex')
            else:
                lagged = np.zeros((len(X), self.n_lags))
                for i, j in iter_from_X_lengths(X, lengths):
                    lagged[i:j, :] = lagmat(X[i:j], maxlag=self.n_lags,
                                            trim='forward', original='ex')

            return {'obs': X.reshape(-1,1),
                    'lagged': lagged.reshape(-1, self.n_features, self.n_lags)}
        else:  # TODO: implement
            """
            ...
            """
            raise ValueError('not implemented')

    def fit(self, X, lengths=None):
        """Estimate model parameters.

        An initialization step is performed before entering the
        EM-algorithm. If you want to avoid this step for a sub of
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
        if random_state is None:  # TODO: generating samples for n_features > 1
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
                init_samples = 0.01*np.ones((self.n_lags, self.n_features))  # TODO: better init

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
            state_ = int(states[idx])
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
        # returns alpha as n_unique x n_lags, if shared_alpha = False
        # returns alpha as 1 x n_lags, if shared_alpha = True
        if self.shared_alpha == False:
            return self._alpha_[
                [u*(1+self.n_tied) for u in range(self.n_unique)], :]
        else:
            return self._alpha_[0,:].reshape(1, self.n_lags)

    def _set_alpha(self, alpha_val):
        # new val needs to have a 1st dim of length n_unique x n_lags
        # if shared_alpha is true, a shape of 1 x n_lags is possible, too
        # internally, n_components x n_lags
        alpha_new = np.zeros((self.n_components, self.n_lags))

        if alpha_val is not None:
            if alpha_val.ndim == 1:
                alpha_val = alpha_val.reshape(-1, 1)  # make sure 2nd dim exists

            if alpha_val.shape[1] != self.n_lags:
                raise ValueError("shape[1] does not match n_lags")

            if self.shared_alpha == False:
                # alpha is not shared
                if alpha_val.shape[0] != self.n_unique:
                    raise ValueError("shape[0] does not match n_unique")
                for u in range(self.n_unique):
                    for t in range(1+self.n_tied):
                        alpha_new[u*(1+self.n_tied)+t, :] = alpha_val[u, :].copy()
            else:
                # alpha is shared ...
                if alpha_val.shape[0] != self.n_unique and \
                  alpha_val.shape[0] != 1:
                    # ... the shape should either be 1 x L or U x L
                    raise ValueError("shape[0] is neither 1 nor does it match n_unique")
                if alpha_val.shape[0] == self.n_unique and \
                  not (alpha_val == alpha_val[0,:]).all():
                    # .. in case of U x L the rows need to be identical
                    raise ValueError("rows not identical (shared_alpha)")
                for u in range(self.n_unique):
                    for t in range(1+self.n_tied):
                        alpha_new[u*(1+self.n_tied)+t, :] = alpha_val[0, :].copy()

        self._alpha_ = alpha_new

    alpha_ = property(_get_alpha, _set_alpha)

    def _get_ar_mean(self):
        # Calculates AR mean
        if self.n_inputs > 0:
            warnings.warn("additional dependency on input vector",
                          RuntimeWarning)
        means = np.zeros(self.n_unique)
        ualphas = self._alpha_[
                  [u*(1+self.n_tied) for u in range(self.n_unique)], :]
        for u in range(self.n_unique):
            num = self.mu_[u]
            denom = 1
            denom -= np.sum(ualphas[u])
            means[u] = num / denom
        return means

    ar_mean_ = property(_get_ar_mean)
