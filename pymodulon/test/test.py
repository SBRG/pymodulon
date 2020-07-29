# -*- coding: utf-8 -*-
"""
Module containing functions for testing various :mod:`pymodulon` methods. Note that the testing
requirements must be installed (e.g. :mod:`pytest`) for this function to work.
"""

from os.path import abspath, dirname, join
import pandas as pd
from pymodulon.core import IcaData

try:
    import pytest
    import pytest_benchmark
except ImportError:
    pytest = None

PYMOD_DIR = abspath(join(dirname(abspath(__file__)), ".."))
"""str: The directory location of where :mod:`pymodulon` is installed."""

DATA_DIR = join(PYMOD_DIR, "test", "data", "")
"""str: THe directory location of the test data."""

# Get test data
s_file = abspath(join(DATA_DIR, 'test_S_matrix.csv'))
a_file = abspath(join(DATA_DIR, 'test_A_matrix.csv'))
x_file = abspath(join(DATA_DIR, 'test_X_matrix.csv'))

# Load test data
s = pd.read_csv(s_file, index_col=0)
a = pd.read_csv(a_file, index_col=0)
x = pd.read_csv(x_file, index_col=0)


def test_core():
    # Test loading in just S and A matrix
    ica_data = IcaData(s, a)
    # TODO: assert that specific functions fail

    # Test loading S, A, and X
    ica_data = IcaData(s, a, x_matrix=x)

    # Test loading all tables
    # TODO: create test data

    return ica_data


def test_enrichment():
    pass


def test_io():
    pass


def test_util():
    pass
