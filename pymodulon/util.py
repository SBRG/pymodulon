import numpy as np
import pandas as pd
from scipy import stats
from typing import *
from warnings import warn
import os

ImodName = Union[str, int]
Data = Union[pd.DataFrame, os.PathLike]


def _check_table(table: Data, name: str, index: List = None):
    # Set as empty dataframe if not input given
    if table is None:
        return pd.DataFrame(index=index)

    # Load table if necessary
    elif isinstance(table, str):
        try:
            table = pd.read_json(table)
        except ValueError:
            try:
                table = pd.read_csv(table, index_col=0)
            except FileNotFoundError:
                raise TypeError('{}_table must be a pandas DataFrame filename or a valid JSON string'.format(name))

    if isinstance(table, pd.DataFrame):
        # dont run _check_table_helper if no index is passed
        return table if index is None else _check_table_helper(table, index, name)
    else:
        raise TypeError('{}_table must be a pandas DataFrame filename or a valid JSON string'.format(name))


def _check_table_helper(table: pd.DataFrame, index: List, name: ImodName):
    if table.shape == (0, 0):
        return pd.DataFrame(index=index)
    # Check if all indices are in table
    missing_index = list(set(index) - set(table.index))
    if len(missing_index) > 0:
        warn('Some {} are missing from the {} table: {}'.format(name, name, missing_index))

    # Remove extra indices from table
    table = table.loc[index]
    return table


def compute_threshold(ic: pd.Series, dagostino_cutoff: float):
    """
    Computes D'agostino-test-based threshold for a component of an M matrix
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


######################
## Compare ICA runs ##
######################

from graphviz import Digraph
from scipy.cluster.hierarchy import linkage, dendrogram
from tqdm import tqdm_notebook as tqdm


def _make_DF_corr(S1: pd.DataFrame, S2: pd.DataFrame, metric: str):
    """

    :param S1:
    :param S2:
    :param metric:
    :return:
    """
    # Only keep genes found in both S matrices
    common = set(S1.index) & set(S2.index)
    s1 = S1.reindex(common)
    s2 = S2.reindex(common)

    # Ensure column names are strings
    s1.columns = s1.columns.astype(str)
    s2.columns = s2.columns.astype(str)

    # Split names in half if necessary for comp1
    cols = {}
    for x in s1.columns:
        if len(x) > 10:
            cols[x] = x[:len(x) // 2] + '-\n' + x[len(x) // 2:]
        else:
            cols[x] = x
    s1.columns = [cols[x] for x in s1.columns]

    # Calculate correlation matrix
    corr = np.zeros((len(s1.columns), len(s2.columns)))

    for i, k1 in tqdm(enumerate(s1.columns), total=len(s1.columns)):
        for j, k2 in enumerate(s2.columns):
            if metric == 'pearson':
                corr[i, j] = abs(stats.pearsonr(s1[k1], s2[k2])[0])

    DF_corr = pd.DataFrame(corr, index=s1.columns, columns=s2.columns)

    return DF_corr


def _make_dot_graph(DF_corr: pd.DataFrame, cutoff: float):
    """

    :param DF_corr:
    :param cutoff:
    :return:
    """
    # Initialize Graph
    dot = Digraph(engine='dot', graph_attr={'ranksep': '0.3', 'nodesep': '0', 'packmode': 'array_u', 'size': '7,7'},
                  node_attr={'fontsize': '14', 'shape': 'none'},
                  edge_attr={'arrowsize': '0.5'}, format='png')

    # Set up linkage and designate terminal nodes
    loc1, loc2 = np.where(DF_corr > cutoff)
    links = list(zip(s1.columns[loc1], s2.columns[loc2]))

    if len(links) == 0:
        warnings.warn('No components shared across runs')
        return None, None

    # Initialize Nodes
    for k in sorted(s2.columns):
        if k in s2.columns[loc2]:
            color = 'black'
            font = 'helvetica'
        else:
            color = 'red'
            font = 'helvetica-bold'
        dot.node('data2_' + str(k), label=k, _attributes={'fontcolor': color, 'fontname': font})

    for k in s1.columns:
        if k in s1.columns[loc1]:
            color = 'black'
            font = 'helvetica'
        else:
            color = 'red'
            font = 'helvetica-bold'
        dot.node('data1_' + str(k), label=k, _attributes={'fontcolor': color, 'fontname': font})

    # Add links between related components
    for k1, k2 in links:
        width = DF_corr.loc[k1, k2] * 5
        dot.edge('data1_' + str(k1), 'data2_' + str(k2), _attributes={'penwidth': '{:.2f}'.format(width)})

    # Reformat names back to normal
    name1, name2 = list(zip(*links))
    inv_cols = {v: k for k, v in cols.items()}
    name_links = list(zip([inv_cols[x] for x in name1], name2))
    return dot, name_links


def compare_ica(S1, S2, metric='pearson', cutoff=0.2,ortho_dir=None):
    """

    :param S1:
    :param S2:
    :param metric:
    :param cutoff:
    :return:
    """
    if ortho_dir is None:
        DF_corr = _make_DF_corr(S1, S2, metric)
        dot, name_links = _make_dot_graph(DF_corr, cutoff)
        return dot, name_links

