from __future__ import division, print_function, absolute_import

import string
import warnings

from scipy import stats
from scipy.stats import norm, multivariate_normal
from scipy.linalg import toeplitz, pinv

from sklearn import cluster
from sklearn.utils import check_random_state

import autograd.numpy as np
from autograd import grad, value_and_grad
from scipy.optimize import minimize
from scipy.special import gamma
import statsmodels.api as smapi
from statsmodels.tsa.tsatools import lagmat


from .ar import ARTHMM

__all__ = ['STUDENT']

ZEROLOGPROB = -1e200
EPS = np.finfo(float).eps
NEGINF = -np.inf

decoder_algorithms = frozenset(("viterbi", "map"))

class STUDENT(ARTHMM):
        """Hidden Markov Model with tied states and autoregressive observations
           drawn from a student t distribution

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
            This is the final precision, will NOT be multiplied by scale factor

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

        degree_freedom : int
            Degrees of freedom

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

        degree_freedom : int, degrees of freedom
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
                    alpha_bounds=np.array([-1.0e5, 1.0e5]),
                    degree_freedom=5):
            super(STUDENT, self).__init__(n_unique=n_unique, n_tied=n_tied,
                                         n_lags=n_lags,
                                         n_features=n_features,
                                         algorithm=algorithm,
                                         params=params, init_params=init_params,
                                         startprob_init=startprob_init,
                                         startprob_prior=startprob_prior,
                                         transmat_init=transmat_init,
                                         transmat_prior=transmat_prior,
                                         mu_init=mu_init, mu_weight=mu_weight,
                                         mu_prior=mu_prior,
                                         shared_alpha=shared_alpha,
                                         precision_init=precision_init,
                                         precision_weight=precision_weight,
                                         precision_prior=precision_prior,
                                         tol=tol, n_iter=n_iter,
                                         n_iter_min=n_iter_min,
                                         n_iter_update=n_iter_update,
                                         random_state=random_state,
                                         verbose=verbose, mu_bounds=mu_bounds,
                                         precision_bounds=precision_bounds,
                                         alpha_bounds=alpha_bounds)

            if degree_freedom <= 2:
                raise ValueError('Degrees of freedom has to be > 2')
            else:
                self.degree_freedom = degree_freedom


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

            samples = xn.shape[0]
            xn = xn.reshape(samples, 1, self.n_features)
            m = m.reshape(1, self.n_unique, self.n_features)
            det = np.linalg.det(np.linalg.inv(p))
            det = det.reshape(1, self.n_unique)

            lagged = np.dot(xln, a.T)  # NFU
            lagged = np.swapaxes(lagged, 1, 2)  # NUF
            xm = xn-(lagged + m)
            tem = np.einsum('NUF,UFX,NUX->NU', xm, p, xm)

            # TODO division in gamma function
            res = np.log(gamma((self.degree_freedom + self.n_features)/2)) - \
                  np.log(gamma(self.degree_freedom/2)) - (self.n_features/2.0) * \
                  np.log(self.degree_freedom) - \
                  (self.n_features/2.0) * np.log(np.pi) - 0.5 * np.log(det) - \
                  ((self.degree_freedom + self.n_features) / 2.0) * \
                  np.log(1 + (1/self.degree_freedom) * tem)

            return res

        def _init_params(self, data, lengths=None, params='stmpaw'):

            super(STUDENT, self)._init_params(data, lengths, params)

            if 'p' in params:
                self.precision_ = self.precision_ * \
                (self.degree_freedom/(self.degree_freedom - 2))

        # Adapted from: https://github.com/statsmodels/
        # statsmodels/blob/master/statsmodels/sandbox/distributions/multivariate.py
        #written by Enzo Michelangeli, style changes by josef-pktd
        # Student's T random variable
        def multivariate_t_rvs(self, m, S, random_state = None):
            '''generate random variables of multivariate t distribution
            Parameters
            ----------
            m : array_like
                mean of random variable, length determines dimension of random variable
            S : array_like
                square array of covariance  matrix
            df : int or float
                degrees of freedom
            n : int
                number of observations, return random array will be (n, len(m))
            random_state : int
                           seed
            Returns
            -------
            rvs : ndarray, (n, len(m))
                each row is an independent draw of a multivariate t distributed
                random variable
            '''
            np.random.rand(9)
            m = np.asarray(m)
            d = self.n_features
            df = self.degree_freedom
            n = 1
            if df == np.inf:
                x = 1.
            else:
                x = random_state.chisquare(df, n)/df
            np.random.rand(90)

            z = random_state.multivariate_normal(np.zeros(d),S,(n,))
            return m + z/np.sqrt(x)[:,None]
            # same output format as random.multivariate_normal


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


                covar_ = np.linalg.inv(precision[state_])

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


                samples[idx] = self.multivariate_t_rvs(mean_, covar_,
                                                       random_state)

            states = self._process_sequence(states)

            return samples, states
