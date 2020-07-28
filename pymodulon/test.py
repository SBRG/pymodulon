# -*- coding: utf-8 -*-
"""
Module containing functions for testing various :mod:`pymodulon` methods.
There is the :func:`test_all` function to run all tests. Note that the testing
requirements must be installed (e.g. :mod:`pytest`) for this function to work.
"""

from os import listdir
from os.path import abspath, dirname, join

try:
    import pytest
    import pytest_benchmark
except ImportError:
    pytest = None