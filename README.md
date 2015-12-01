# autohmm

Description ..

## Installation

Install the dependencies:

```bash
$ pip install hmmlearn numpy scipy scikit-learn statsmodels theano
```

As of now, this package is not installable via pip.
Instead, clone the current version from git using:

```bash
$ git clone https://github.com/mackelab/arhmm.git
```

And run the installation script in the source directory:

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

The documentation is available at:

[http://mackelab.github.io/autohmm/](http://mackelab.github.io/autohmm/)


## License

...
