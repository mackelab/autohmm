# Based on `hmmlearn` project Makefile.
# Testing will not work on Windows.

PYTHON ?= python
NOSETESTS ?= nosetests

GH_PAGES_DIRS = _sources _static api
GH_PAGES_BUILD = autohmm docs

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

gh-pages:
	git checkout gh-pages
	rm -rf $(GH_PAGES_DIRS)
	git checkout master $(GH_PAGES_BUILD)
	git reset HEAD
	cd docs
	make html
	mv -fv _build/html/* ../
	cd ..
	rm -rf $(GH_PAGES_BUILD)
	git add -A
	git commit -m "Generated gh-pages for `git log master -1 --pretty=short --abbrev-commit`" && git push origin gh-pages ; git checkout master
