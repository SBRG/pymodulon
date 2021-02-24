"""
Pre-loaded example dataset for PyModulon tutorials. Dataset from Sastry et al. 2019.
Nature Communications.
"""

from os import path
import pandas as pd
from pymodulon.io import load_json_model

_data_dir = path.join(path.abspath(path.dirname(__file__)))

ECOLI_DIR = path.join(path.abspath(path.dirname(__file__)), 'ecoli')

# E coli datasets
M = pd.read_csv(path.join(ECOLI_DIR, 'M.csv'), index_col=0)
A = pd.read_csv(path.join(ECOLI_DIR, 'A.csv'), index_col=0)
X = pd.read_csv(path.join(ECOLI_DIR, 'X.csv'), index_col=0)
gene_table = pd.read_csv(path.join(ECOLI_DIR, 'gene_table.csv'), index_col=0)
sample_table = pd.read_csv(path.join(ECOLI_DIR, 'sample_table.csv'), index_col=0)
imodulon_table = pd.read_csv(path.join(ECOLI_DIR, 'imodulon_table.csv'), index_col=0)
trn = pd.read_csv(path.join(ECOLI_DIR, 'trn.csv'), index_col=None)

# E coli genome annotations
ecoli_fasta = path.join(ECOLI_DIR, 'genome.fasta')


# Load E coli IcaData Object
def load_ecoli_data():
    return load_json_model(path.join(ECOLI_DIR, 'ecoli_data.json'))


def load_staph_data():
    pass


def load_example_bbh():
    return pd.read_csv(path.join(_data_dir, 'example_bbh.csv'))


def load_example_tpm():
    return pd.read_csv(path.join(_data_dir, 'example_tpm.csv'), index_col=0)