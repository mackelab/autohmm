from __future__ import division, print_function, absolute_import

import time
import sys
from collections import deque

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

def gamma_prior_params(mean_gamma_prior, var_gamma_prior):
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
