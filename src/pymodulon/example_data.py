"""
Pre-loaded example dataset for PyModulon tutorials.
"""

from os import path

import pandas as pd

from pymodulon.io import load_json_model


_data_dir = path.join(path.abspath(path.dirname(__file__)), "data")

_ecoli_dir = path.join(_data_dir, "ecoli")

# E. coli datasets
M = pd.read_csv(path.join(_ecoli_dir, "M.csv"), index_col=0)
A = pd.read_csv(path.join(_ecoli_dir, "A.csv"), index_col=0)
X = pd.read_csv(path.join(_ecoli_dir, "X.csv"), index_col=0)
gene_table = pd.read_csv(path.join(_ecoli_dir, "gene_table.csv"), index_col=0)
sample_table = pd.read_csv(path.join(_ecoli_dir, "sample_table.csv"), index_col=0)
imodulon_table = pd.read_csv(path.join(_ecoli_dir, "imodulon_table.csv"), index_col=0)
trn = pd.read_csv(path.join(_ecoli_dir, "trn.csv"), index_col=None)

# E. coli genome annotations
ecoli_fasta = path.join(_ecoli_dir, "genome.fasta")
ecoli_gff = path.join(_ecoli_dir, "genome.gff3")
ecoli_eggnog = path.join(_ecoli_dir, "eggNOG_annotations.txt")
ecoli_biocyc = path.join(_ecoli_dir, "biocyc_operon_annotations.txt")
ecoli_go_example = path.join(_ecoli_dir, "GO_example_annotations.txt")


# Load E coli IcaData Object
def load_ecoli_data():
    """
    Load *Escherichia coli* :class:`~pymodulon.core.IcaData` object from
    :cite:`Sastry2019`

    Returns
    -------
    ecoli_data: ~pymodulon.core.IcaData
        *E. coli* :class:`~pymodulon.core.IcaData` object
    """
    return load_json_model(path.join(_data_dir, "objects", "ecoli_data.json.gz"))


def load_staph_data():
    """
    Load *Staphylococcus aureus* :class:`~pymodulon.core.IcaData` object from
    :cite:`Poudel2020`

    Returns
    -------
    staph_data: ~pymodulon.core.IcaData
        *S. aureus* :class:`~pymodulon.core.IcaData` object
    """
    return load_json_model(path.join(_data_dir, "objects", "staph_data.json.gz"))


def load_bsub_data():
    """
    Load *Bacillus subtilis* :class:`~pymodulon.core.IcaData` object from
    :cite:`Rychel2020a`

    Returns
    -------
    bsub_data: ~pymodulon.core.IcaData
        *B. subtilis* :class:`~pymodulon.core.IcaData` object
    """
    return load_json_model(path.join(_data_dir, "objects", "bsub_data.json.gz"))


def load_example_bbh():
    """
    Load an example bi-directional blast best hit (BBH) file

    Returns
    -------
    example_bbh: ~pandas.DataFrame
        Example BBH file
    """
    return pd.read_csv(path.join(_data_dir, "bbh", "example_bbh.csv"), index_col=0)


def load_example_log_tpm():
    """
    Load an example expression dataset in units log-TPM

    Returns
    -------
    example_tpm: ~pandas.DataFrame
        Example expression dataset
    """
    return pd.read_csv(path.join(_data_dir, "ecoli", "example_tpm.csv"), index_col=0)
