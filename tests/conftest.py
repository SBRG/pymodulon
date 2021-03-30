# -*- coding: utf-8 -*-
"""Module containing functions for testing various :mod:`pymodulon` methods.
Note that the testing requirements must be installed (e.g. :mod:`pytest`) for
this function to work. """

import pytest

from pymodulon import example_data
from pymodulon.core import IcaData


ecoli_data = example_data.load_ecoli_data()

# Add an iModulon with a number as its name
ecoli_data.rename_imodulons({"CdaR": 2})

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
