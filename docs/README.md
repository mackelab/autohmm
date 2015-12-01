# Documentation

An online version of the documentation is hosted at: [http://mackelab.github.io/autohmm/](http://mackelab.github.io/autohmm/).


## Building

To build the documentation found in the wiki, execute `make html` in this directory.

For this to work, a number of dependencies need to be met. Python dependencies
 can be installed with:
```bash
$ pip install sphinx sphinxcontrib-napoleon sphinx_rtd_theme
```

Sphinx is used to extract [Docstrings]() from the package, for the API
 reference. The contributed Sphinx extension napoleon is used to allow writing
 Docstrings in NumPy style.
