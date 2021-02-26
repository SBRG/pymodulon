"""
Pre-loaded example dataset for PyModulon tutorials. Dataset from Sastry et al. 2019.
Nature Communications.
"""

from os import path

import pandas as pd

from pymodulon.io import load_json_model


_data_dir = path.join(path.abspath(path.dirname(__file__)), "data")

_ecoli_dir = path.join(_data_dir, "ecoli")

# E coli datasets
M = pd.read_csv(path.join(_ecoli_dir, "M.csv"), index_col=0)
A = pd.read_csv(path.join(_ecoli_dir, "A.csv"), index_col=0)
X = pd.read_csv(path.join(_ecoli_dir, "X.csv"), index_col=0)
gene_table = pd.read_csv(path.join(_ecoli_dir, "gene_table.csv"), index_col=0)
sample_table = pd.read_csv(path.join(_ecoli_dir, "sample_table.csv"), index_col=0)
imodulon_table = pd.read_csv(path.join(_ecoli_dir, "imodulon_table.csv"), index_col=0)
trn = pd.read_csv(path.join(_ecoli_dir, "trn.csv"), index_col=None)

# E coli genome annotations
ecoli_fasta = path.join(_ecoli_dir, "genome.fasta")
ecoli_gff = path.join(_ecoli_dir, "genome.gff3")
ecoli_eggnog = path.join(_ecoli_dir, "eggNOG.annotations")


# Load E coli IcaData Object
def load_ecoli_data():
    return load_json_model(path.join(_data_dir, "objects", "ecoli_data.json"))


def load_staph_data():
    return load_json_model(path.join(_data_dir, "objects", "staph_data.json"))


def load_example_bbh():
    return pd.read_csv(path.join(_data_dir, "bbh", "example_bbh.csv"), index_col=0)


def load_example_tpm():
    return pd.read_csv(path.join(_data_dir, "ecoli", "example_tpm.csv"), index_col=0)
