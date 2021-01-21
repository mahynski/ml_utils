ml_utils
========
Additional python machine learning utilities and functions not found in scikit-learn, etc.

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![codecov](https://codecov.io/gh/mahynski/ml_utils/branch/main/graph/badge.svg?token=YSLBQ33C7F)](https://codecov.io/gh/mahynski/ml_utils)
[![Build Status](https://travis-ci.com/mahynski/ml_utils.svg?branch=main)](https://travis-ci.com/mahynski/ml_utils)

## Installation

~~~ bash
$ git clone https://github.com/mahynski/ml_utils.git
$ pip install -r requirements.txt
~~~

Simply add this directory to your PYTHONPATH, or locally in each instance (i.e., sys.path.append()) and import the model as usual.

~~~ bash
$ echo 'export PYTHONPATH=$PYTHONPATH:/path/to/module/' >> ~/.bashrc
$ source ~/.bashrc
~~~

~~~ python
import ml_utils
~~~

## Unittests
~~~ bash
$ python -m unittest discover tests/
~~~
