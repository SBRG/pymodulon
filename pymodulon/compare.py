import os
import tqdm
import warnings
import numpy as np
import pandas as pd
from re import split
from typing import *
from graphviz import Digraph
from scipy import stats


def _make_dot_graph(M1: pd.DataFrame, M2: pd.DataFrame, metric: str,
                    cutoff: float, show_all: bool):
    """
    Given two M matrices, returns the dot graph and name links of the various
    connected ICA components
    Args:
        M1: M matrix from the first organism
        M2: M matrix from the second organism
        metric: Statistical test to use (default:'pearson')
        cutoff: Float cut off value for pearson statistical test
        show_all:

    Returns: Dot graph and name links of connected ICA components

    """

    # Only keep genes found in both S matrices
    common = set(M1.index) & set(M2.index)

    if len(common) == 0:
        raise KeyError("No common genes")

    s1 = M1.reindex(common)
    s2 = M2.reindex(common)

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

    for i, k1 in tqdm.tqdm(enumerate(s1.columns), total=len(s1.columns)):
        for j, k2 in enumerate(s2.columns):
            if metric == 'pearson':
                corr[i, j] = abs(stats.pearsonr(s1[k1], s2[k2])[0])
            elif metric == 'spearman':
                corr[i, j] = abs(stats.spearmanr(s1[k1], s2[k2])[0])

    # Only keep genes found in both S matrices
    DF_corr = pd.DataFrame(corr, index=s1.columns, columns=s2.columns)

    # Initialize Graph
    dot = Digraph(engine='dot', graph_attr={'ranksep': '0.3', 'nodesep': '0',
                                            'packmode': 'array_u',
                                            'size': '10,10'},
                  node_attr={'fontsize': '14', 'shape': 'none'},
                  edge_attr={'arrowsize': '0.5'}, format='png')

    # Set up linkage and designate terminal nodes
    loc1, loc2 = np.where(DF_corr > cutoff)
    links = list(zip(s1.columns[loc1], s2.columns[loc2]))

    if len(links) == 0:
        warnings.warn('No components shared across runs')
        return None, None
    if show_all is True:
        # Initialize Nodes
        for k in sorted(s2.columns):
            if k in s2.columns[loc2]:
                color = 'black'
                font = 'helvetica'
            else:
                color = 'red'
                font = 'helvetica-bold'
            dot.node('data2_' + str(k), label=k,
                     _attributes={'fontcolor': color, 'fontname': font})

        for k in s1.columns:
            if k in s1.columns[loc1]:
                color = 'black'
                font = 'helvetica'
            else:
                color = 'red'
                font = 'helvetica-bold'
            dot.node('data1_' + str(k), label=k,
                     _attributes={'fontcolor': color, 'fontname': font})

        # Add links between related components
        for k1, k2 in links:
            width = DF_corr.loc[k1, k2] * 5
            dot.edge('data1_' + str(k1), 'data2_' + str(k2),
                     _attributes={'penwidth': '{:.2f}'.format(width)})
    else:
        # Initialize Nodes
        for k in sorted(s2.columns):
            if k in s2.columns[loc2]:
                color = 'black'
                font = 'helvetica'
                dot.node('data2_' + str(k), label=k,
                         _attributes={'fontcolor': color, 'fontname': font})

        for k in s1.columns:
            if k in s1.columns[loc1]:
                color = 'black'
                font = 'helvetica'
                dot.node('data1_' + str(k), label=k,
                         _attributes={'fontcolor': color, 'fontname': font})

        # Add links between related components
        for k1, k2 in links:
            if k1 in s1.columns[loc1] and k2 in s2.columns[loc2]:
                width = DF_corr.loc[k1, k2] * 5
                dot.edge('data1_' + str(k1), 'data2_' + str(k2),
                         _attributes={'penwidth': '{:.2f}'.format(width)})

    # Reformat names back to normal
    name1, name2 = list(zip(*links))
    inv_cols = {v: k for k, v in cols.items()}
    name_links = list(zip([inv_cols[x] for x in name1], name2))
    return dot, name_links


def _pull_bbh_csv(ortho_file: str, S1: pd.DataFrame):
    """
    Receives an the S matrix for an organism and returns the same S matrix
    with index genes translated into the orthologs in organism 2
    Args:
        ortho_file: String path to the bbh CSV file in the
        "modulome_compare_data" repository.
        Ex. "../../modulome_compare_data/bbh_csv/
        bSubtilis_full_protein_vs_sAureus_full_protein_parsed.csv"
        S1: Pandas DataFrame of the S matrix for organism 1

    Returns:
        Pandas DataFrame of the S matrix for organism 1 with indexes
        translated into orthologs

    """

    bbh_DF = pd.read_csv(ortho_file, index_col="gene")

    S1_copy = S1.copy()
    S1_index = list(S1_copy.index)

    for i in range(0, len(S1_index)):
        try:
            S1_index[i] = bbh_DF.loc[S1_index[i]]["subject"]
        except KeyError:
            continue
    S1_copy.index = S1_index

    return S1_copy


def compare_ica(S1: pd.DataFrame, S2: pd.DataFrame,
                ortho_file: Optional[str] = None, cutoff: float = 0.2,
                metric='pearson', show_all: bool = False):
    """
    Compares two S matrices between a single organism or across organisms and
    returns the connected ICA components
    Args:
        S1: Pandas Dataframe of S matrix 1
        S2: Pandas Dataframe of S Matrix 2
        ortho_file: String of the location where organism data can be found
            (can be found under modulome/data)
        cutoff: Float cut off value for pearson statistical test
        metric: A string of what statistical test to use (standard is 'pearson')
        show_all: True will show all nodes of the digraph matrix

    Returns: Dot graph and name links of connected ICA components between the
    two runs or organisms.

    """
    if ortho_file is None:
        dot, name_links = _make_dot_graph(S1, S2, metric=metric, cutoff=cutoff,
                                          show_all=show_all)
        return dot, name_links

    else:
        warnings.warn("Please ensure that the order of S1 and S2 match the "
                      "order of the BBH CSV file")
        translated_S = _pull_bbh_csv(ortho_file, S1)
        dot, name_links = _make_dot_graph(translated_S, S2, metric, cutoff,
                                          show_all=show_all)
        return dot, name_links
