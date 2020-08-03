# -*- coding: utf-8 -*-
"""
Module containing functions for testing various :mod:`pymodulon` methods. Note that the testing
requirements must be installed (e.g. :mod:`pytest`) for this function to work.
"""

from os.path import abspath, dirname, join
from pymodulon.core import IcaData
from pymodulon.enrichment import *

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
    test_simple_ica_data()
    ica_data = IcaData(s, a, x_matrix=x, gene_table=gene_table, sample_table=sample_table,
                       imodulon_table=imodulon_table, trn=trn, dagostino_cutoff=750)
    test_ica_data_consistency(ica_data)
    test_compute_regulon_enrichment(ica_data)
    test_compute_trn_enrichment(ica_data)


def test_simple_ica_data():
    # Test loading in just S and A matrix
    IcaData(s, a)


def test_ica_data_consistency(ica_data):
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
    assert (ica_data.gene_names[0] == 'b0002')

    # Check if renaming works for iModulons
    ica_data.rename_imodulons({0: 'YieP'})
    print(ica_data.view_imodulon('YieP'))


def test_compute_regulon_enrichment(ica_data):
    print('Testing single enrichment')
    enrich = ica_data.compute_regulon_enrichment(1, 'glpR')
    print(enrich)

    print('Testing multiple enrichment')
    enrich = ica_data.compute_regulon_enrichment(1, 'glpR+crp')
    print(enrich)

    print('Original iModulon table')
    print(ica_data.imodulon_table)

    print('GlpR iModulon table')
    ica_data.compute_regulon_enrichment(1, 'glpR', save=True)



def test_compute_trn_enrichment(ica_data):
    print('Testing full TRN enrichment')
    enrich = ica_data.compute_trn_enrichment()
    print(enrich)

    print('Original iModulon table')
    print(ica_data.imodulon_table)
    print('Full iModulon table')
    ica_data.compute_trn_enrichment(save=True)
    print(ica_data.imodulon_table)


def test_io():
    pass


def test_util():
    pass
