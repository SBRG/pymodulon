"""
General utility functions for the pymodulon package
"""

from matplotlib.axes import Axes
from typing import *
import numpy as np
import pandas as pd
from re import split
from typing import List
import os
from scipy import stats
import tqdm
from graphviz import Digraph
import warnings

################
# Type Aliases #
################
Ax = TypeVar("Ax", Axes, object)
Data = Union[pd.DataFrame, os.PathLike]
SeqSetStr = Union[Sequence[str], Set[str], str]
ImodName = Union[str, int]
ImodNameList = Union[ImodName, List[ImodName]]


def _check_table(table: Data, name: str, index: Optional[Collection] = None,
                 index_col=0):
    # Set as empty dataframe if not input given
    if table is None:
        return pd.DataFrame(index=index)

    # Load table if necessary
    elif isinstance(table, str):
        try:
            table = pd.read_json(table)
        except ValueError:
            sep = '\t' if table.endswith('.tsv') else ','
            table = pd.read_csv(table, index_col=index_col, sep=sep)

    if isinstance(table, pd.DataFrame):
        # dont run _check_table_helper if no index is passed
        return table if index is None else _check_table_helper(table, index,
                                                               name)
    else:
        raise TypeError('{}_table must be a pandas DataFrame '
                        'filename or a valid JSON string'.format(name))


def _check_table_helper(table: pd.DataFrame, index: Optional[Collection],
                        name: ImodName):
    if table.shape == (0, 0):
        return pd.DataFrame(index=index)
    # Check if all indices are in table
    missing_index = list(set(index) - set(table.index))
    if len(missing_index) > 0:
        warnings.warn('Some {} are missing from the {} table: {}'
                      .format(name, name, missing_index))

    # Remove extra indices from table
    table = table.loc[index]
    return table


def compute_threshold(ic: pd.Series, dagostino_cutoff: float):
    """
    Computes D'agostino-test-based threshold for a component of an M matrix
    :param ic: Pandas Series containing an independent component
    :param dagostino_cutoff: Minimum D'agostino test statistic value
        to determine threshold
    :return: iModulon threshold
    """
    i = 0

    # Sort genes based on absolute value
    ordered_genes = abs(ic).sort_values()

    # Compute k2-statistic
    k_square, p = stats.normaltest(ic)

    # Iteratively remove gene w/ largest weight until k2-statistic < cutoff
    while k_square > dagostino_cutoff:
        i -= 1
        k_square, p = stats.normaltest(ic.loc[ordered_genes.index[:i]])

    # Select genes in iModulon
    comp_genes = ordered_genes.iloc[i:]

    # Slightly modify threshold to improve plotting visibility
    if len(comp_genes) == len(ic.index):
        return max(comp_genes) + .05
    else:
        return np.mean([ordered_genes.iloc[i], ordered_genes.iloc[i - 1]])


def name2num(ica_data, gene: Union[Iterable, str]) -> Union[Iterable, str]:
    """
    Convert a gene name to the locus tag
    Args:
        ica_data: IcaData object
        gene: Gene name or list of gene names

    Returns: Locus tag or list of locus tags

    """
    gene_table = ica_data.gene_table
    if 'gene_name' not in gene_table.columns:
        raise ValueError('Gene table does not contain "gene_name" column.')

    if isinstance(gene, str):
        gene_list = [gene]
    else:
        gene_list = gene

    final_list = []
    for g in gene_list:
        loci = gene_table[gene_table.gene_name == g].index

        # Ensure only one locus maps to this gene
        if len(loci) == 0:
            raise ValueError('Gene does not exist: {}'.format(g))
        elif len(loci) > 1:
            warnings.warn('Found multiple genes named {}. Only '
                          'reporting first locus tag'.format(g))

        final_list.append(loci[0])

    # Return string if string was given as input
    if isinstance(gene, str):
        return final_list[0]
    else:
        return final_list


def num2name(ica_data, gene: Union[Iterable, str]) -> Union[Iterable, str]:
    """
    Convert a locus tag to the gene name
    Args:
        ica_data: IcaData object
        gene: Locus tag or list of locus tags

    Returns: Gene name or list of gene names

    """
    result = ica_data.gene_table.loc[gene].gene_name
    if isinstance(gene, list):
        return result.tolist()
    else:
        return result
