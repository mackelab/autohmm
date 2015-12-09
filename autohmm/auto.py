from __future__ import division, print_function, absolute_import

import theano
import theano.tensor as tt
from theano.tensor import addbroadcast as bc
from theano import function, scan, shared, pp

import numpy as np
import scipy.optimize

import warnings

class _AUTO(object):
    def __init__(self, model):
        self.inputs_hmm_ll = []
        self.inputs_neg_ll = []
        self.wrt = []
        self.wrt_dims = {}
        self.wrt_bounds = {}

        self.init_current = True
        self.verbose = model.verbose

    def init(self, on_unused_input='ignore'):
        self._hmm_ll_ = theano.function(inputs=self.inputs_hmm_ll,
                                        outputs=self.hmm_ll_,
                                        on_unused_input=on_unused_input)

        self._tfn_obj_ = theano.function(inputs=self.inputs_neg_ll,
                                         outputs=self.neg_ll_,
                                         on_unused_input=on_unused_input)

        params = [str(el) for el in self.wrt]
        grad = theano.gradient.jacobian(self.neg_ll_, self.wrt)
        self._tfn_grad_ = {param: theano.function(inputs=self.inputs_neg_ll,
                                           outputs=grad[param_idx],
                                           on_unused_input=on_unused_input)
                           for param_idx, param in enumerate(params)}

    def optim(self, param, values, optim_maxiter=50, disp=0):

        if self.init_current:
            if type(values[param]) != float:
                optim_x0 = values[param].reshape(-1)
            else:
                optim_x0 = np.float64(values[param])
        else:
            if self.wrt_dims[param] != (1):
                optim_x0 = 0.001*np.ones(self.wrt_dims[param], dtype='float64')
            else:
                optim_x0 = np.float64(0.001)

        optim_bounds = [self.wrt_bounds[param] for k in
                        range(np.prod(self.wrt_dims[param]))]

        #if param == 'c':
        #    import pdb; pdb.set_trace()

        best_x, nfeval, rc = scipy.optimize.fmin_tnc(x0=optim_x0,
                                         func=self._eval_nll,
                                         fprime=self._eval_grad_nll,
                                         args=(param, values),
                                         bounds=optim_bounds,
                                         disp = 0)  # optim_maxiter
                                         # TODO: disp according to verbose arg

        # if(rc == 4):
        #     # TODO: verbosity parameter, logging
        #     #print('line search failed for param {}, nfeval is '.format(param, nfeval))
        #     if param == 'c':
        #         maxv = self.hmm_obs_max_.eval({self.xn:values['xn'], self.snb:values['snb'], self.b:values['b'], self.o:values['o']})
        #         maxc = np.log(1/values['r'])/maxv*0.999
        #         optim_bounds = [(0.01, maxc)]
        #
        #         best_x, nfeval, rc = scipy.optimize.fmin_tnc(x0=optim_x0,
        #                                  func=self._eval_nll,
        #                                  fprime=self._eval_grad_nll,
        #                                  args=(param, values),
        #                                  bounds=optim_bounds,
        #                                  disp = 5)  # optim_maxiter
        #
        #         print('redid optimization, upper {}, found {}'.format(maxc, best_x))

        result = best_x.reshape(self.wrt_dims[param])
        return result

    def hmm_ll(self, values):
        values = self._convert_shape(values)
        return self._hmm_ll_(**values)

    def _eval_nll(self, x, param, values, derivative=0, reshape=True):
        if derivative == 0:
            fn_handle = self._tfn_obj_
        elif derivative == 1:
            fn_handle = self._tfn_grad_[param]
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

class THMM(_AUTO):
    def __init__(self, model):
        _AUTO.__init__(self, model)

        self.xn = tt.dmatrix('xn')    # N x 1
        self.gn = tt.dmatrix('gn')    # N x n_unique
        self.mw = tt.dscalar('mw')
        self.mp = tt.dvector('mp')    # n_unique
        self.pw = tt.dscalar('pw')
        self.pp = tt.dvector('pp')    # n_unique
        self.m = tt.dvector('m')
        if not model.tied_precision:
            self.p = tt.dvector('p')
        else:
            self.p = tt.dscalar('p')

        self.inputs_hmm_ll.extend([self.xn, self.m, self.p])
        self.inputs_neg_ll.extend([self.xn, self.m, self.p, self.gn, self.mw,
                                   self.pw, self.mp, self.pp])

        self.wrt.extend([self.m, self.p])
        self.wrt_dims.update({'m': (model.n_unique,)})
        if not model.tied_precision:
            self.wrt_dims.update({'p': (model.n_unique,)})
        else:
            self.wrt_dims.update({'p': (1)})
        self.wrt_bounds.update({'m': (-50.0, 50.0)})
        self.wrt_bounds.update({'p': (0.001, 10000.0)})

        self.hmm_obs   = bc(self.xn, 1)
        self.hmm_mean  = self.m
        self.hmm_prior = (self.pw-0.5) * tt.log(self.p) - \
                         0.5*self.p*(self.mw*(self.m-self.mp)**2 + 2*self.pp)

    @property
    def hmm_ll_(self):
        return -0.5*tt.log(2*np.pi) + 0.5*tt.log(self.p) - \
               0.5*self.p*(self.hmm_obs-self.hmm_mean)**2
               # N x n_unique

    @property
    def hmm_ell_(self):
        return tt.sum(self.hmm_prior) + tt.sum(self.gn * self.hmm_ll_)
               # (1,)

    @property
    def neg_ll_(self):
        return -1*self.hmm_ell_

class ARTHMM(THMM):
    def __init__(self, model):
        THMM.__init__(self, model)

        if model.n_lags > 0:
            self.xln = tt.dmatrix('xln')  # N x n_lags
            self.a = tt.dmatrix('a')

            self.inputs_hmm_ll.extend([self.xln, self.a])
            self.inputs_neg_ll.extend([self.xln, self.a])
            self.wrt.extend([self.a])
            if not model.tied_alpha:
                self.wrt_dims.update({'a': (model.n_unique, model.n_lags)})
            else:
                self.wrt_dims.update({'a': (1, model.n_lags)})
            self.wrt_bounds.update({'a': (-10.0, 10.0)})

            if not model.tied_alpha:
                self.hmm_mean = self.hmm_mean + tt.dot(self.xln, self.a.T)
            else:
                self.hmm_mean = self.hmm_mean + bc(tt.dot(self.xln, self.a.T),1)

        if model.n_inputs > 0:
            self.fn = tt.dmatrix('fn')  # N x n_inputs
            self.w = tt.dmatrix('w')

            self.inputs_hmm_ll.extend([self.fn, self.w])
            self.inputs_neg_ll.extend([self.fn, self.w])
            self.wrt.extend([self.w])
            if not model.tied_omega:
                self.wrt_dims.update({'w': (model.n_unique,
                                            model.n_inputs)})
            else:
                self.wrt_dims.update({'w': (1, model.n_inputs)})
            self.wrt_bounds.update({'w': (-10.0, 10.0)})

            if not model.tied_omega:
                self.hmm_mean = self.hmm_mean + tt.dot(self.fn, self.w.T)
            else:
                self.hmm_mean = self.hmm_mean + bc(tt.dot(self.fn, self.w.T), 1)
