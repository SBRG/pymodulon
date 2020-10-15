"""
General utility functions for the pymodulon package
"""
from itertools import combinations

from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import os
from scipy import stats
import warnings
from typing import *
import re

from pymodulon.enrichment import FDR

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


def dima(ica_data, sample1: Union[Collection, str],
         sample2: Union[Collection, str], threshold: float = 5,
         fdr: float = 0.1):
    """

    Args:
        ica_data: IcaData object
        sample1: List of sample IDs or name of "project:condition"
        sample2: List of sample IDs or name of "project:condition"
        threshold: Minimum activity difference to determine DiMAs
        fdr: False Detection Rate

    Returns:

    """
    _diff = pd.DataFrame()

    sample1_list = _parse_sample(ica_data, sample1)
    sample2_list = _parse_sample(ica_data, sample2)

    for name, group in ica_data.sample_table.groupby(['project', 'condition']):
        for i1, i2 in combinations(group.index, 2):
            _diff[':'.join(name)] = abs(ica_data.A[i1] - ica_data.A[i2])
    dist = {}

    for k in ica_data.A.index:
        dist[k] = stats.lognorm(*stats.lognorm.fit(_diff.loc[k].values)).cdf

    res = pd.DataFrame(index=ica_data.A.index)
    for k in res.index:
        a1 = ica_data.A.loc[k, sample1_list].mean()
        a2 = ica_data.A.loc[k, sample2_list].mean()
        res.loc[k, 'difference'] = a2 - a1
        res.loc[k, 'pvalue'] = 1 - dist[k](abs(a1 - a2))
    result = FDR(res, fdr)
    return result[(abs(result.difference) > threshold)].sort_values(
        'difference', ascending=False)


def _parse_sample(ica_data, sample: Union[Collection, str]):
    """
    Parses sample inputs into a list of sample IDs
    Args:
        ica_data: IcaData object
        sample: List of sample IDs or "project:condition"

    Returns: A list of samples

    """
    sample_table = ica_data.sample_table
    if isinstance(sample, str):
        proj, cond = re.search('(.*):(.*)', sample).groups()
        samples = sample_table[(sample_table.project == proj) &
                               (sample_table.condition == cond)].index
        if len(samples) == 0:
            raise ValueError(f'No samples exist for project={proj} condition='
                             f'{cond}')
        else:
            return samples
    else:
        return sample
