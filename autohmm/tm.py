from __future__ import division, print_function, absolute_import

import string
import warnings

from scipy import linalg
from scipy import stats
from scipy.stats import norm, multivariate_normal
import scipy.sparse as sp

from sklearn import cluster
from sklearn.utils import check_random_state
from sklearn.mixture import sample_gaussian

import autograd.numpy as np
from autograd import grad, value_and_grad
from scipy.optimize import minimize

from hmmlearn.utils import normalize, iter_from_X_lengths

from .base import _BaseAUTOHMM
from .utils import ConvergenceMonitor

__all__ = ['THMM']

ZEROLOGPROB = -1e200
EPS = np.finfo(float).eps
NEGINF = -np.inf

decoder_algorithms = frozenset(("viterbi", "map"))

class THMM(_BaseAUTOHMM):
    """Hidden Markov Model with tied states

    Parameters
    ----------
    n_unique : int
        Number of unique components.

    n_tied : int
        Number of tied states for each component.

    n_features: int
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

    mu_bounds : array, shape (``2``)
        Upper and lower bound for mu [lower bound, upper bound].

    precision_bounds : array, shape (``2``)
        Upper and lower bound for precision [lower bound, upper bound].

    Attributes
    ----------
    n_components : int
        Number of total components

    mu_ : array, shape (``n_unique``, ``n_features``)

    precision_ : array, shape (``n_unique``, ``n_features``, ``n_features``)

    transmat_ :  array, shape (``n_unique``, ``n_unique``)

    startprob_ :  array, shape (``n_unique``, ``n_unique``)
    """
    def __init__(self, n_unique=2, n_tied=0, n_features=1,
                 algorithm="viterbi",
                 params=string.ascii_letters, init_params=string.ascii_letters,
                 startprob_init=None, startprob_prior=1.0,
                 transmat_init=None, transmat_prior=None,
                 mu_init=None, mu_weight=0.0, mu_prior=None,
                 precision_init=None, precision_weight=0.0,
                 precision_prior=None, tol=1e-4,
                 n_iter=25, n_iter_min=2, n_iter_update=1,
                 random_state=None, verbose=False,
                 mu_bounds=np.array([-1e5, 1e5]),
                 precision_bounds=np.array([-1e5, 1e5])):
        super(THMM, self).__init__(algorithm=algorithm, params=params,
                                   init_params=init_params, tol=tol,
                                   n_iter=n_iter, n_iter_min=n_iter_min,
                                   n_iter_update=n_iter_update,
                                   random_state=random_state, verbose=verbose)

        self.n_tied = n_tied
        self.n_chain = n_tied+1
        self.n_unique = n_unique
        self.n_components = n_unique * self.n_chain
        self.n_features = n_features

        self.mu_bounds = mu_bounds
        self.precision_bounds = precision_bounds

        # parameters are passed to a setters
        # setters check for shape and set default values
        self.startprob_ = startprob_init
        self.startprob_prior_ = startprob_prior
        self.transmat_ = transmat_init
        self.transmat_prior_ = transmat_prior
        self.mu_ = mu_init
        self.mu_weight_ = mu_weight
        self.mu_prior_ = mu_prior
        self.precision_ = precision_init
        self.precision_weight_ = precision_weight
        self.precision_prior_ = precision_prior

        self.wrt.extend(['m', 'p'])
        self.wrt_dims.update({'m': (self.n_unique, self.n_features)})
        self.wrt_dims.update({'p': (self.n_unique, self.n_features, self.n_features)})
        self.wrt_bounds.update({'m': (self.mu_bounds[0], self.mu_bounds[1])})
        self.wrt_bounds.update({'p': (self.precision_bounds[0],
                                      self.precision_bounds[1])})

    def _compute_log_likelihood(self, data, from_=0, to_=-1):
        ll = self._ll(self.mu_, self.precision_, data['obs'][from_:to_])
        return ll

    def _ll(self, m, p, xn, **kwargs):
        """Computation of log likelihood

        Dimensions
        ----------
        m : n_unique x n_features
        p : n_unique x n_features x n_features
        xn: N x n_features
        """

        samples = xn.shape[0]
        xn = xn.reshape(samples, 1, self.n_features)
        m = m.reshape(1, self.n_unique, self.n_features)

        det = np.linalg.det(np.linalg.inv(p))
        det = det.reshape(1, self.n_unique)
        tem = np.einsum('NUF,UFX,NUX->NU', (xn - m), p, (xn - m))
        res = (-self.n_features/2.0)*np.log(2*np.pi) - 0.5*tem - 0.5*np.log(det)

        return res  # N x n_unique

    def _obj(self, m, p, xn, gn, **kwargs):
        ll = self._ll(m, p, xn)

        mw = self.mu_weight_
        mp = self.mu_prior_
        pw = self.precision_weight_
        pp = self.precision_prior_

        m = m.reshape(self.n_unique, self.n_features, 1)
        mp = mp.reshape(self.n_unique, self.n_features, 1)

        prior = (pw-0.5) * np.log(p) - 0.5*p*(mw*(m-mp)**2 + 2*pp)

        res = -1*(np.sum(gn * ll) ) + np.sum(prior)
        return res  # scalar

    def _obj_grad(self, wrt, m, p, xn, gn, **kwargs):
        m = m.reshape(self.n_unique, self.n_features, 1)
        res = grad(self._obj, wrt)(m, p, xn, gn)
        res = np.array([res])
        return res  # scalar

    def _do_mstep_grad(self, gn, data):
        wrt = [str(p) for p in self.wrt if str(p) in self.params]
        for update_idx in range(self.n_iter_update):
            for p in wrt:
                if p == 'm':
                    optim_x0 = self.mu_
                    wrt_arg = 0
                elif p == 'p':
                    optim_x0 = self.precision_
                    wrt_arg = 1
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
                                         'xn': data['obs'],
                                         'gn': gn  # post. uni. concat.
                                        }),
                                  bounds=optim_bounds,
                                  method='TNC')
                newv = result.x.reshape(self.wrt_dims[p])
                if p == 'm':
                    self.mu_ = newv
                elif p == 'p':
                    # ensure that precision matrix is symmetric
                    for u in range(self.n_unique):
                        newv[u,:,:] = (newv[u,:,:] + newv[u,:,:].T)/2.0
                    self.precision_ = newv
                else:
                    raise ValueError('unknown parameter')

    def _init_params(self, data, lengths=None, params='stmp'):
        X = data['obs']

        if 's' in params:
            self.startprob_.fill(1.0 / self.n_components)

        if 't' in params or 'm' in params or 'p' in params:

            kmmod = cluster.KMeans(n_clusters=self.n_unique,
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
            mu_init = np.zeros((self.n_unique, self.n_features))
            for u in range(self.n_unique):
                for f in range(self.n_features):
                    mu_init[u][f] = kmeans[u, f]

            self.mu_ = np.copy(mu_init)

        if 'p' in params:
            precision_init = np.zeros((self.n_unique, self.n_features, self.n_features))
            for u in range(self.n_unique):
                if self.n_features == 1:
                    precision_init[u] = np.linalg.inv(np.cov(X[kmmod.labels_ == u], bias = 1))
                else:
                    precision_init[u] = np.linalg.inv(np.cov(np.transpose(X[kmmod.labels_ == u])))

            self.precision_ = np.copy(precision_init)

    def _do_mstep(self, stats, params):  # M-Step for startprob and transmat
        if 's' in params:
            startprob_ = self.startprob_prior + stats['start']
            normalize(startprob_)
            self.startprob_ = np.where(self.startprob_ <= np.finfo(float).eps,
                                       self.startprob_, startprob_)
        if 't' in params:

            if self.n_tied == 0:
                transmat_ = self.transmat_prior + stats['trans']
                normalize(transmat_, axis=1)
                self.transmat_ = np.where(self.transmat_ <= np.finfo(float).eps,
                                          self.transmat_, transmat_)
            else:
                transmat_ = np.zeros((self.n_components, self.n_components))
                transitionCnts = stats['trans'] + self.transmat_prior
                transition_index = [i * self.n_chain for i in range(self.n_unique)]

                for b in range(self.n_unique):
                    block = \
                    transitionCnts[self.n_chain * b : self.n_chain * (b + 1)][:] + 0.

                    denominator_diagonal = np.sum(block)
                    diagonal = 0.0

                    index_line = range(0, self.n_chain)
                    index_row = range(self.n_chain * b, self.n_chain * (b + 1))

                    for l, r in zip(index_line, index_row):
                        diagonal += (block[l][r])

                    for l, r in zip(index_line, index_row):
                        block[l][r] = diagonal / denominator_diagonal

                    self_transition = block[0][self.n_chain * b]
                    denominator_off_diagonal = \
                    (np.sum(block[self.n_chain-1])) - self_transition
                    template = block[self.n_chain - 1] + 0.

                    for entry in range(len(template)):
                        template[entry] = (template[entry] * (1 - self_transition)) \
                        / float(denominator_off_diagonal)

                    template[(self.n_chain * (b + 1)) - 1] = 0.
                    line_value = 1 - self_transition

                    for entry in range(len(template)):
                        line_value = line_value - template[entry]

                    for index in transition_index:
                        if index != (b * self.n_chain):
                            block[self.n_chain - 1][index] = \
                            line_value + template[index]

                    line = range(self.n_chain - 1)
                    row = [b * self.n_chain + i for i in range(1, self.n_chain)]

                    for x, y in zip(line, row):
                        block[x][y] = 1 - self_transition

                    transmat_[self.n_chain * b : self.n_chain * (b + 1)][:] = block
                self.transmat_ = np.copy(transmat_)

    def _process_inputs(self, X):
        if self.n_features == 1:
            return {'obs': X.reshape(-1,1)}
        else:
            return {'obs': X}

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

        return reduced_sequence.astype(int)

    def fit(self, X, lengths=None):
        """Estimate model parameters.

        An initialization step is performed before entering the
        EM-algorithm. If you want to avoid this step for a subset of
        the parameters, pass proper ``init_params`` keyword argument
        to estimator's constructor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
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
        self._init_params(data, lengths=lengths, params=self.init_params)

        X = data['obs']
        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter,
                                           self.n_iter_min, self.verbose)
        for iter in range(self.n_iter):
            stats = self._initialize_sufficient_statistics()
            gn = np.zeros((X.shape[0], self.n_unique))
            curr_logprob = 0
            for i, j in iter_from_X_lengths(X, lengths):
                flp = self._compute_log_likelihood(data, from_=i, to_=j) # n_samples, n_unique
                flp_rep = np.zeros((flp.shape[0], self.n_components))
                for u in range(self.n_unique):
                    for c in range(self.n_chain):
                        flp_rep[:, u*self.n_chain+c] = flp[:, u]

                # n_samples, n_components below
                logprob, fwdlattice = self._do_forward_pass(flp_rep)
                curr_logprob += logprob
                bwdlattice = self._do_backward_pass(flp_rep)
                posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
                self._accumulate_sufficient_statistics(
                    stats, X[i:j], flp_rep, posteriors, fwdlattice, bwdlattice)
                # sum responsibilities across chain if tied states exist
                if self.n_tied == 0:
                    gn[i:j, :] = posteriors
                elif self.n_tied > 0:
                    for u in range(self.n_unique):
                        cols = range(u*(self.n_chain),
                                     u*(self.n_chain)+(self.n_chain))
                        gn[i:j, u] = (np.sum(posteriors[:, cols], axis=1)).reshape(X.shape[0])

            self.monitor_.report(curr_logprob)
            if self.monitor_.converged:
                break

            self._do_mstep(stats, self.params)
            self._do_mstep_grad(gn, data)

        return self

    def sample(self, n_samples=2000, observed_states=None, random_state=None):
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

        Returns
        -------
        samples : array_like, length (``n_samples``, ``n_features``)
                  List of samples

        states : array_like, shape (``n_samples``)
                 List of hidden states (accounting for tied states by giving
                 them the same index)
        """

        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        samples = np.zeros((n_samples, self.n_features))
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
                newstate = (transmat_cdf[int(states[idx-1])] > nrand[idx-1]).argmax()
                states[idx] = newstate

        else:
            states = observed_states

        mu = np.copy(self._mu_)
        precision = np.copy(self._precision_)
        for idx in range(n_samples):
            mean_ = mu[states[idx]]

            covar_ = np.linalg.inv(precision[states[idx]])
            samples[idx] = multivariate_normal.rvs(mean=mean_, cov=covar_,
                                                   random_state=random_state)
        states = self._process_sequence(states)
        return samples, states

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
        logprob, posteriors = self._do_score_samples(data)
        return logprob, posteriors

    def decode(self, X, algorithm="viterbi"):
        """Find most likely state sequence corresponding to ``X``.
        Uses the selected algorithm for decoding.

        Parameters
        ----------
        X : array_like, shape (n)
            Each row corresponds to a single point in the sequence.

        algorithm : string, one of the ``decoder_algorithms``
            decoder algorithm to be used.

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

    def _get_mu(self):
        if self.n_tied == 0:
            return self._mu_
        else:
            return self._mu_[[u*(self.n_chain) for u in range(self.n_unique)]]

    def _set_mu(self, mu_val):
        # new val needs to be of shape (n_uniqe, n_features)
        # internally, (n_components x n_features)
        mu_new = np.zeros((self.n_components, self.n_features))
        if mu_val is not None:
            mu_val = mu_val.reshape(self.n_unique, self.n_features)
            if mu_val.shape == (self.n_unique, self.n_features):
                for u in range(self.n_unique):
                    for t in range(self.n_chain):
                        mu_new[u*(self.n_chain)+t] = mu_val[u].copy()
            else:
                raise ValueError("cannot match shape of mu")
        self._mu_ = mu_new

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
            self._mu_prior_ = np.zeros((self.n_components, self.n_features))
        else:
            mu_prior = np.asarray(mu_prior)
            mu_prior = mu_prior.reshape(self.n_unique, self.n_features)
            if mu_prior.shape == (self.n_unique, self.n_features):
                for u in range(self.n_unique):
                    for t in range(self.n_chain):
                        self._mu_prior[u*(self.n_chain)+t] = mu_prior[u].copy()

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
        # new val needs to have dimension (n_unique, n_features, n_features)
        # internally, n_components x 1
        precision_new = \
        np.zeros((self.n_components, self.n_features, self.n_features))
        if precision_val is not None:
            precision_val = \
            precision_val.reshape(self.n_unique, self.n_features, self.n_features)
            if precision_val.shape == \
            (self.n_unique, self.n_features, self.n_features):
                for u in range(self.n_unique):
                    for t in range(self.n_chain):
                        precision_new[u*(self.n_chain)+t] = precision_val[u].copy()
            else:
                raise ValueError("cannot match shape of precision")
        self._precision_ = precision_new

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
            self._precision_prior_ = \
            np.zeros((self.n_components, self.n_features, self.n_features))
        else:
            precision_prior = np.asarray(precision_prior)
            if len(precision_prior) == 1:
                self._precision_prior_ = np.tile(precision_prior,
                (self.n_components, self.n_features, self.n_features))
            elif \
            (precision_prior.reshape(self.n_unique, self.n_features, self.n_features)).shape \
            == (self.n_unique, self.n_features, self.n_features):
                self._precision_prior_ = \
                np.zeros((self.n_components, self.n_features, self.n_features))
                for u in range(self.n_unique):
                    for t in range(self.n_chain):
                        self._precision_prior_[u*(self.n_chain)+t] = precision_prior[u].copy()
            else:
                raise ValueError("cannot match shape of precision_prior")

    precision_prior_ = property(_get_precision_prior, _set_precision_prior)

    def _get_var(self):
        if self.n_features == 1:
            return 1.0 / self._get_precision()
        else:
            return np.linalg.inv(self._get_precision())

    def _set_var(self, var_val):
        if self.n_features == 1:
            return self._set_precision(1.0 / var_val)
        else:
            return self._set_precision(np.linalg.inv(var_val))

    var_ = property(_get_var, _set_var)

    def _get_var_weight(self):
        return self._get_precision_weight()

    def _set_var_weight(self, var_weight):
        self._set_precision_weight(var_weight)

    var_weight_ = property(_get_var_weight, _set_var_weight)

    def _get_var_prior(self):
        if self.n_features == 1:
            return 1.0 / self._get_precision_prior()
        else:
            return np.linalg.inv(self._get_precision_prior())

    def _set_var_prior(self, var_prior):
        var_prior = np.asarray(var_prior)
        if self.n_features == 1:
            self._set_precision_prior(1.0 / var_prior)
        else:
            self._set_precision_prior(np.linalg.inv(var_prior))

    var_prior_ = property(_get_var_prior, _set_var_prior)

    def _get_startprob(self):  # TODO: decide upon shape
        return np.exp(self._log_startprob)

    def _set_startprob(self, startprob):
        if startprob is None:
            startprob = np.tile(1.0 / self.n_components, self.n_components)
        else:
            startprob = np.asarray(startprob, dtype=np.float)

            normalize(startprob)

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
        if startprob_prior is None or startprob_prior == 1.0:
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
        #                        +-----------------+
        #                        |a|1|0|0|0|0|0|0|0|
        #                        +-----------------+
        #                        |0|a|1|0|0|0|0|0|0|
        #                        +-----------------+
        #   +---+---+---+        |0|0|a|b|0|0|c|0|0|
        #   | a | b | c |        +-----------------+
        #   +-----------+        |0|0|0|e|1|0|0|0|0|
        #   | d | e | f | +----> +-----------------+
        #   +-----------+        |0|0|0|0|e|1|0|0|0|
        #   | g | h | i |        +-----------------+
        #   +---+---+---+        |d|0|0|0|0|e|f|0|0|
        #                        +-----------------+
        #                        |0|0|0|0|0|0|i|1|0|
        #                        +-----------------+
        #                        |0|0|0|0|0|0|0|i|1|
        #                        +-----------------+
        #                        |g|0|0|h|0|0|0|0|i|
        #                        +-----------------+
        # for a model with n_unique = 3 and n_tied = 2
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

    def _get_transmat(self):  # TODO: decide upon shape
            return np.exp(self._log_transmat)

    def _set_transmat(self, transmat_val):
        if transmat_val is None:
            transmat = np.tile(1.0 / self.n_components,
                               (self.n_components, self.n_components))
        else:
            transmat_val[np.isnan(transmat_val)] = 0.0
            normalize(transmat_val, axis=1)

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

    def _set_transmat_prior(self, transmat_prior_val):
        # new val needs be n_unique x n_unique
        # internally, n_components x n_components
        # _ntied_transmat_prior is
        # called to get n_components x n_components
        transmat_prior_new = np.zeros((self.n_components, self.n_components))
        if transmat_prior_val is not None:

            if transmat_prior_val.shape == (self.n_unique, self.n_unique):
                transmat_prior_new = \
                np.copy(self._ntied_transmat_prior(transmat_prior_val))

            else:
                raise ValueError("cannot match shape of transmat_prior")


        self.transmat_prior = transmat_prior_new

    transmat_prior_ = property(_get_transmat_prior, _set_transmat_prior)
