import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, List
from warnings import warn
import os

ImodName = Union[str, int]
Data = Union[pd.DataFrame, os.PathLike]


def _check_table(table: Data, index: List, name: str):
    # Set as empty dataframe if not input given
    if table is None:
        return pd.DataFrame(index=index)

    # Load table if necessary
    elif isinstance(table, str):
        df = pd.read_csv(table, index_col=0)
        return _check_table_helper(df, index, name)
    elif isinstance(table, pd.DataFrame):
        return _check_table_helper(table, index, name)
    else:
        raise TypeError('{}_table must be a pandas DataFrame or filename'.format(name))


def _check_table_helper(table: pd.DataFrame, index: List, name: str):
    # Check if all indices are in table
    missing_index = list(set(index) - set(table.index))
    if len(missing_index) > 0:
        warn('Some {} are missing from the {} table: {}'.format(name, name, ', '.join(missing_index)))

    # Remove extra indices from table
    table = table.loc[index]
    return table


def rename_imodulon(ica_data, old_name: ImodName, new_name: ImodName) -> None:
    """
    Rename an iModulon
    :param ica_data: The IcaData object
    :param old_name: iModulon name to be replaced
    :param new_name: New iModulon name
    """
    # Check that new names is not already in use
    old_list = ica_data.imodulon_names
    if new_name in old_list:
        raise ValueError('iModulon name ({:s}) already in use. Please choose a different name.'.format(new_name))
    if old_name not in old_list:
        raise ValueError('No iModulon named {:s}'.format(old_name))
    name_list = [name if name != old_name else new_name for name in old_list]
    ica_data.imodulon_names = name_list


def compute_threshold(ic: pd.Series, dagostino_cutoff: float):
    """
    Computes D'agostino-test-based threshold for a component of an S matrix
    :param ic: Pandas Series containing an independent component
    :param dagostino_cutoff: Minimum D'agostino test statistic value to determine threshold
    :return: iModulon threshold
    """
    i = 0

    # Sort genes based on absolute value
    ordered_genes = abs(ic).sort_values()

    # Compute k2-statistic
    k_square, p = stats.normaltest(ic)

    # Iteratively remove gene with largest weight until k2-statistic is below cutoff
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
