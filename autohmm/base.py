from __future__ import division, print_function, absolute_import

import string
import warnings

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

    # hmmlearn implements _do_viterbi_pass, _do_forward_pass,
    # _do_backward_pass, _compute_posteriors, _accumulate_sufficient_statistics

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
