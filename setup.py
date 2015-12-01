from __future__ import absolute_import
from setuptools import setup

setup(name='autohmm',
      version='0.1.0',
      description='Autoregressive Hidden Markov Models and extensions',
      author='Jan-Matthis Lueckmann',
      author_email='jan-matthis.lueckmann@caesar.de',
      packages=['autohmm'],
      requires=['scipy', 'numpy', 'sklearn', 'hmmlearn', 'statsmodels',
                'theano'],
      install_requires=['scipy >=0.16', 'numpy', 'sklearn', 'hmmlearn >=0.2.0',
                        'statsmodels>=0.6.0', 'theano'],
      keywords=['Autoression', 'Autoregressive Hidden-Markov Model', 'HMM',
                'machine learning', 'Python', 'Numpy', 'Scipy', 'Theano'],
      url='https://github.com/mackelab/autohmm',
      license='new BSD',
      classifiers=['Development Status :: 3 - Alpha',
                   'License :: OSI Approved',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.4'])
