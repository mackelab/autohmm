# autohmm [![Build Status](https://travis-ci.org/mackelab/autohmm.svg?branch=master)](https://travis-ci.org/mackelab/autohmm) [![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](http://www.mackelab.org/autohmm/)

This packages provides an implementation of Hidden Markov Models (HMMs) with tied states and autoregressive observations, written in Python. For HMM recursions, the C implementations of the [hmmlearn package](https://github.com/hmmlearn/hmmlearn) are used. Evaluation of the likelihood function and the maximization of the expected complete data log-likelihood is implemented in [Theano](https://github.com/Theano/Theano), to allow quick development of novel models.

**Important: The code in this repository is still experimental, and APIs are subject to change without warning.**

## Installation

As of now, this package is not installable via pip. Instead, clone the current version from git:

```bash
$ git clone https://github.com/mackelab/autohmm.git
```

The dependencies are listed in `requirements.txt`. If you are using pip, you can install the dependencies through:

```bash
$ pip install -r requirements.txt
```

To install autohmm, call `make` or:

```bash
$ python setup.py install
```

## Quick Example

```python
import numpy as np
from autohmm import ar

model = ar.ARTHMM(n_unique=2)
model.mu_ = np.array([2.0, -2.0])
model.var_ = np.array([0.25, 0.25])

samples, states = model.sample(n_samples=500)
```


## Documentation

**An early stage version of the documentation is available at: [http://www.mackelab.org/autohmm/](http://www.mackelab.org/autohmm/)**
