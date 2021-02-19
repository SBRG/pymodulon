# -*- coding: utf-8 -*-
"""Module containing functions for testing various :mod:`pymodulon` methods.
Note that the testing requirements must be installed (e.g. :mod:`pytest`) for
this function to work. """

from os.path import abspath, dirname, join

import pandas as pd
import pytest

from pymodulon.core import IcaData


PYMOD_DIR = abspath(join(dirname(abspath(__file__)), ".."))
"""str: The directory location of where :mod:`pymodulon` is installed."""

DATA_DIR = join(PYMOD_DIR, "tests", "data", "ecoli_data")
"""str: THe directory location of the test data."""

# Get test data
m_file = abspath(join(DATA_DIR, "M.csv"))
a_file = abspath(join(DATA_DIR, "A.csv"))
x_file = abspath(join(DATA_DIR, "X.csv"))
gene_file = abspath(join(DATA_DIR, "gene_table.csv"))
sample_file = abspath(join(DATA_DIR, "sample_table.csv"))
imodulon_file = abspath(join(DATA_DIR, "imodulon_table.csv"))
trn_file = abspath(join(DATA_DIR, "trn.csv"))

# Load test data
m = pd.read_csv(m_file, index_col=0)
a = pd.read_csv(a_file, index_col=0)
x = pd.read_csv(x_file, index_col=0)
gene_table = pd.read_csv(gene_file, index_col=0)
sample_table = pd.read_csv(sample_file, index_col=0)
imodulon_table = pd.read_csv(imodulon_file, index_col=0)
trn = pd.read_csv(trn_file)

# TODO: Remove excess genes from TRN file

ecoli_data = IcaData(
    m,
    a,
    X=x,
    gene_table=gene_table,
    sample_table=sample_table,
    imodulon_table=imodulon_table,
    trn=trn,
    dagostino_cutoff=750,
    optimize_cutoff=False,
)

# Smaller version of Ecoli data with 10 iModulons
mini_data = IcaData(
    ecoli_data.M.iloc[:, :10],
    ecoli_data.A.iloc[:10, :],
    gene_table=ecoli_data.gene_table,
    imodulon_table=ecoli_data.imodulon_table[:10],
    trn=ecoli_data.trn,
    optimize_cutoff=False,
    dagostino_cutoff=2000,
)

# Capture expected UserWarning here
with pytest.warns(UserWarning) as record:
    mini_data_opt = IcaData(
        ecoli_data.M.iloc[:, :10],
        ecoli_data.A.iloc[:10, :],
        gene_table=ecoli_data.gene_table,
        imodulon_table=ecoli_data.imodulon_table[:10],
        trn=ecoli_data.trn,
        optimize_cutoff=True,
    )


@pytest.fixture()
def ecoli_obj():
    return ecoli_data.copy()


@pytest.fixture()
def mini_obj():
    return mini_data.copy()


@pytest.fixture()
def mini_obj_opt():
    return mini_data_opt.copy()
