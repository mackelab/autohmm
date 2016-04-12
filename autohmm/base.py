from __future__ import division, print_function, absolute_import

import string
import warnings

from hmmlearn import _hmmc
from hmmlearn.base import _BaseHMM, logsumexp

import autograd.numpy as np

decoder_algorithms = frozenset(("viterbi", "map"))

class _BaseAUTOHMM(_BaseHMM):
    """autohmm base class

    Parameters
    ----------
    algorithm : string
        Decoding algorithm.

    params : string
        Controls which parameters are updated in the training
        process. Defaults to all parameters.

    init_params : string
        Controls which parameters are initialized prior to
        training. Defaults to all parameters.

    n_iter : int
        Number of iterations to perform maximally.

    n_iter_min : int
        Number of iterations to perform minimally.

    n_iter_update : int
        Number of iterations per M-Step.

    random_state : int
        Sets seed.

    tol : float
        Convergence threshold, below which EM will stop.

    verbose : bool
        When ``True`` convergence reports are printed.

    Attributes
    ----------
    compiled : bool
        Set to `True` when Theano functions are compiled.
    """
    def __init__(self, algorithm="viterbi", params=string.ascii_letters,
                 init_params=string.ascii_letters, tol=1e-4, n_iter=25,
                 n_iter_min=2, n_iter_update=1, random_state=None,
                 verbose=False):
        # TODO: optim verbosity, optim iter
        self.algorithm = algorithm
        self.params = params
        self.init_params = init_params
        self.tol = tol
        self.n_iter = n_iter
        self.n_iter_min = n_iter_min
        self.n_iter_update = n_iter_update
        self.random_state = random_state
        self.verbose = verbose

        self.wrt = []
        self.wrt_dims = {}
        self.wrt_bounds = {}

    def _optim_wrap(self, current_value, param, values = {}):
        values[param] = np.array(current_value).reshape(self.wrt_dims[param])
        return (self._obj(**values).reshape(-1),
                self._obj_grad(**values).reshape(-1))

    def _do_score_samples(self, data, lengths=None):  # adapted hmmlearn
        # TODO: Support lengths arguement
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

    def _do_decode(self, data, algorithm=None, lengths=None):  # adapt. hmmlearn
        # TODO: Support lengths arguement
        if algorithm in decoder_algorithms:
            algorithm = algorithm
        elif self.algorithm in decoder_algorithms:
            algorithm = self.algorithm
        decoder = {"viterbi": self._decode_viterbi,
                   "map": self._decode_map}
        logprob, state_sequence = decoder[algorithm](data)
        return logprob, state_sequence

    def _decode_viterbi(self, data):  # adapted hmmlearn
        framelogprob = self._compute_log_likelihood(data)
        viterbi_logprob, state_sequence = self._do_viterbi_pass(framelogprob)
        return viterbi_logprob, state_sequence

    def _decode_map(self, data):  # adapted hmmlearn
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
        return map_logprob, state_sequence

    def _do_viterbi_pass(self, framelogprob):
        # Based on hmmlearn's _BaseHMM
        safe_startmat = self.startprob_ + np.finfo(float).eps
        safe_transmat = self.transmat_ + np.finfo(float).eps
        n_samples, n_components = framelogprob.shape
        state_sequence, logprob = _hmmc._viterbi(
            n_samples, n_components, np.log(safe_startmat),
            np.log(safe_transmat), framelogprob)
        return logprob, state_sequence

    def _do_forward_pass(self, framelogprob):
        # Based on hmmlearn's _BaseHMM
        safe_startmat = self.startprob_ + np.finfo(float).eps
        safe_transmat = self.transmat_ + np.finfo(float).eps
        n_samples, n_components = framelogprob.shape
        fwdlattice = np.zeros((n_samples, n_components))
        _hmmc._forward(n_samples, n_components,
                       np.log(safe_startmat),
                       np.log(safe_transmat),
                       framelogprob, fwdlattice)
        return logsumexp(fwdlattice[-1]), fwdlattice

    def _do_backward_pass(self, framelogprob):
        # Based on hmmlearn's _BaseHMM
        safe_startmat = self.startprob_ + np.finfo(float).eps
        safe_transmat = self.transmat_ + np.finfo(float).eps
        n_samples, n_components = framelogprob.shape
        bwdlattice = np.zeros((n_samples, n_components))
        _hmmc._backward(n_samples, n_components,
                        np.log(safe_startmat),
                        np.log(safe_transmat),
                        framelogprob, bwdlattice)
        return bwdlattice


    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        """Updates sufficient statistics from a given sample.
        Parameters
        ----------
        stats : dict
            Sufficient statistics as returned by
            :meth:`~base._BaseHMM._initialize_sufficient_statistics`.
        X : array, shape (n_samples, n_features)
            Sample sequence.
        framelogprob : array, shape (n_samples, n_components)
            Log-probabilities of each sample under each of the model states.
        posteriors : array, shape (n_samples, n_components)
            Posterior probabilities of each sample being generated by each
            of the model states.
        fwdlattice, bwdlattice : array, shape (n_samples, n_components)
            Log-forward and log-backward probabilities.
        """

        # Based on hmmlearn's _BaseHMM
        safe_transmat = self.transmat_ + np.finfo(float).eps
        stats['nobs'] += 1
        if 's' in self.params:
            stats['start'] += posteriors[0]
        if 't' in self.params:
            n_samples, n_components = framelogprob.shape
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            if n_samples <= 1:
                return

            lneta = np.zeros((n_samples - 1, n_components, n_components))
            _hmmc._compute_lneta(n_samples, n_components, fwdlattice,
                                 np.log(safe_transmat),
                                 bwdlattice, framelogprob, lneta)

            stats['trans'] += np.exp(logsumexp(lneta, axis=0))
            # stats['trans'] = np.round(stats['trans'])
            # if np.sum(stats['trans']) != X.shape[0]-1:
            #     warnings.warn("transmat counts != n_samples", RuntimeWarning)
            #     import pdb; pdb.set_trace()
            stats['trans'][np.where(stats['trans'] < 0.01)] = 0.0
