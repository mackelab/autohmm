from __future__ import division, print_function, absolute_import

import string
import warnings

import theano
import theano.tensor as tt
from theano.tensor import addbroadcast as bc
from theano import function, scan, shared, pp

from hmmlearn.base import _BaseHMM, logsumexp

import numpy as np
import scipy.optimize


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

        # initialize variables for Theano
        self.inputs_hmm_ll = []
        self.inputs_neg_ll = []
        self.wrt = []
        self.wrt_dims = {}
        self.wrt_bounds = {}
        self.compiled = False

    def _compile(self, on_unused_input='ignore'):
        self._tfun_hmm_ll_ = theano.function(inputs=self.inputs_hmm_ll,
                                             outputs=self.hmm_ll_,
                                             on_unused_input=on_unused_input)
        self._tfun_obj_ = theano.function(inputs=self.inputs_neg_ll,
                                          outputs=self.neg_ll_,
                                          on_unused_input=on_unused_input)
        params = [str(el) for el in self.wrt]
        grad = theano.gradient.jacobian(self.neg_ll_, self.wrt)
        self._tfun_grad_ = {param: theano.function(inputs=self.inputs_neg_ll,
                                                   outputs=grad[param_idx],
                                                   on_unused_input=\
                                                       on_unused_input)
                            for param_idx, param in enumerate(params)}
        self.compiled = True

    def _convert_shape(self, values):
        for param in values:
            if param in self.wrt_dims:
                if self.wrt_dims[param] != (1) and \
                   type(values[param]) != float:
                    values[param] = values[param].reshape(self.wrt_dims[param])
                else:
                    if type(values[param]) == np.ndarray:
                        values[param] = float(values[param][0])
                    else:
                        values[param] = float(values[param])
        return values

    def _eval_hmm_ll(self, values):
        values = self._convert_shape(values)
        return self._tfun_hmm_ll_(**values)

    def _eval_nll(self, x, param, values, derivative=0, reshape=True):
        if derivative == 0:
            fn_handle = self._tfun_obj_
        elif derivative == 1:
            fn_handle = self._tfun_grad_[param]
        else:
            raise ValueError('not implemented')

        values[param] = x
        values = self._convert_shape(values)

        result = fn_handle(**values)

        if reshape is True and self.wrt_dims[param] != (1,):
            return result.reshape(-1)
        else:
            return result

    def _eval_grad_nll(self, *args, **kwargs):
        kwargs.update({'derivative': 1})
        return self._eval_nll(*args, **kwargs)

    def _optim(self, param, values, optim_maxiter=50, disp=0):
        if type(values[param]) != float:
            optim_x0 = values[param].reshape(-1)
        else:
            optim_x0 = np.float64(values[param])

        optim_bounds = [self.wrt_bounds[param] for k in
                        range(np.prod(self.wrt_dims[param]))]

        best_x, nfeval, rc = scipy.optimize.fmin_tnc(x0=optim_x0,
                                         func=self._eval_nll,
                                         fprime=self._eval_grad_nll,
                                         args=(param, values),
                                         bounds=optim_bounds,
                                         disp = 0)  # optim_maxiter
                                         # TODO: disp according to verbose arg
        result = best_x.reshape(self.wrt_dims[param])
        return result

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

    def _do_decode(self, data, algorithm=None, lengths=None):  # adapted hmmlearn
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
