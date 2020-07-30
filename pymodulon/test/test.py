# -*- coding: utf-8 -*-
"""
Module containing functions for testing various :mod:`pymodulon` methods. Note that the testing
requirements must be installed (e.g. :mod:`pytest`) for this function to work.
"""

from os.path import abspath, dirname, join
from pymodulon.core import IcaData
from pymodulon.enrichment import *
import pytest

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


# Test enrichment module
set1 = {'gene0', 'gene1', 'gene2'}

all_genes = set(['gene' + str(i) for i in range(10)])


def test_enrichment():
    test_compute_enrichment0()
    test_compute_enrichment1()
    test_compute_enrichment2()
    test_compute_enrichment3()


def test_compute_enrichment0():
    # Make sure function runs
    set2 = {'gene2', 'gene3', 'gene4'}
    result = compute_enrichment(set1, set2, all_genes, label='test')
    assert (0 < result.pvalue < 1 and
            0 < result.precision < 1 and
            0 < result.recall < 1 and
            0 < result.f1score < 1 and
            result.TP == 1)


def test_compute_enrichment1():
    # Edge case 1: No overlap
    set2 = {'gene3', 'gene4', 'gene5'}
    result = compute_enrichment(set1, set2, all_genes, label='test')
    assert (result.pvalue == 1 and
            result.precision == 0 and
            result.recall == 0 and
            result.f1score == 0 and
            result.TP == 0)


def test_compute_enrichment2():
    # Edge case 2: Complete overlap
    set2 = set1
    result = compute_enrichment(set1, set2, all_genes, label='test')
    assert (result.pvalue == 0 and
            result.precision == 1 and
            result.recall == 1 and
            result.f1score == 1 and
            result.TP == 3)


def test_compute_enrichment3():
    # Edge case 3: Genes outside of all_genes
    set2 = {'gene100'}
    with pytest.raises(ValueError):
        compute_enrichment(set1, set2, all_genes, label='test')


def test_io():
    pass


def test_util():
    pass
