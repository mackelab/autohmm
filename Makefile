# Based on `hmmlearn` project Makefile.
# Testing will not work on Windows.

PYTHON ?= python
NOSETESTS ?= nosetests

.PHONY: all clean docs test code-analysis

all: clean inplace test

clean:
	$(PYTHON) setup.py clean
	rm -rf dist

docs:
	$(MAKE) -C docs

in: inplace  # shortcut

inplace:
	$(PYTHON) setup.py install

test-code: in
	$(NOSETESTS) -s -v tests

test: test-code

code-analysis:
	flake8 autohmm | grep -v __init__ | grep -v external
	pylint -E -i y autohmm/ -d E1103,E0611,E1101
