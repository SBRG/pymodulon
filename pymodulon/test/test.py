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
s_file = abspath(join(DATA_DIR, 'ecoli_M.csv'))
a_file = abspath(join(DATA_DIR, 'ecoli_A.csv'))
x_file = abspath(join(DATA_DIR, 'ecoli_X.csv'))
gene_file = abspath(join(DATA_DIR, 'ecoli_gene_table.csv'))
sample_file = abspath(join(DATA_DIR, 'ecoli_sample_table.csv'))
imodulon_file = abspath(join(DATA_DIR, 'ecoli_imodulon_table.csv'))
trn_file = abspath(join(DATA_DIR, 'ecoli_trn.csv'))

# Load test data
s = pd.read_csv(s_file, index_col=0)
a = pd.read_csv(a_file, index_col=0)
x = pd.read_csv(x_file, index_col=0)
gene_table = pd.read_csv(gene_file, index_col=0)
sample_table = pd.read_csv(sample_file, index_col=0)
imodulon_table = pd.read_csv(imodulon_file, index_col=0)
trn = pd.read_csv(trn_file)


def test_core():
    # Test loading in just S and A matrix
    IcaData(s, a)

    # Test loading everything
    ica_data = IcaData(s, a, x_matrix=x, gene_table=gene_table, sample_table=sample_table,
                       imodulon_table=imodulon_table, trn=trn)

    # Ensure that gene names are consistent
    gene_list = ica_data.gene_names
    assert (gene_list == ica_data.X.index.tolist() and
            gene_list == ica_data.S.index.tolist() and
            gene_list == ica_data.gene_table.index.tolist())

    # Ensure that sample names are consistent
    sample_list = ica_data.sample_names
    assert (sample_list == ica_data.X.columns.tolist() and
            sample_list == ica_data.A.columns.tolist() and
            sample_list == ica_data.sample_table.index.tolist())

    # Ensure that iModulon names are consistent
    imodulon_list = ica_data.imodulon_names
    assert (imodulon_list == ica_data.S.columns.tolist() and
            imodulon_list == ica_data.A.index.tolist() and
            imodulon_list == ica_data.imodulon_table.index.tolist())

    # Check if gene names are used
    assert(ica_data.gene_names[0] == 'thrA')

    # Check if renaming works for iModulons
    ica_data.rename_imodulons({2:'YgbI'})
    assert(ica_data.imodulon_names[2] == 'YgbI' and
           ica_data.A.index[2] == 'YgbI' and
           ica_data.S.columns[2] == 'YgbI')

    # TODO: Test loading all tables


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
    assert (0 <= result.pvalue <= 1 and
            0 <= result.precision <= 1 and
            0 <= result.recall <= 1 and
            0 <= result.f1score <= 1 and
            result.TP <= 0)


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
