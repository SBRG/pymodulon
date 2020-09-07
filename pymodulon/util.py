import os

import numpy as np
import pandas as pd

from matplotlib.axes import Axes
from scipy import stats
from typing import *
import warnings
from graphviz import Digraph
from tqdm import tqdm_notebook as tqdm
from scipy import sparse


################
# Type Aliases #
################
Ax = TypeVar("Ax", Axes, object)
Data = Union[pd.DataFrame, os.PathLike]
SeqSetStr = Union[Sequence[str], Set[str], str]
ImodName = Union[str, int]
ImodNameList = Union[ImodName, List[ImodName]]


def _check_table(table: Data, name: str, index: Optional[Collection] = None):
    # Set as empty dataframe if not input given
    if table is None:
        return pd.DataFrame(index=index)

    # Load table if necessary
    elif isinstance(table, str):
        try:
            table = pd.read_json(table)
        except ValueError:
            sep = '\t' if table.endswith('.tsv') else ','
            table = pd.read_csv(table, index_col=0, sep=sep)

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


####################
# Compare ICA runs #
####################

from scipy import sparse
import numpy as np
import pandas as pd
from re import split
from typing import List, Dict
import os
from scipy import stats, special
import tqdm
from graphviz import Digraph
import warnings


def _make_dot_graph(S1: pd.DataFrame, S2: pd.DataFrame, metric: str,
                    cutoff: float, show_all: bool):
    """
    Given two S matrices, returns the dot graph and name links of the various
    connected ICA components
    Args:
        S1: S matrix from the first organism
        S2: S matrix from the second organism
        metric: Statistical test to use (default:'pearson')
        cutoff: Float cut off value for pearson statistical test
        show_all:

    Returns: Dot graph and name links of connected ICA components

    """

    # Only keep genes found in both S matrices
    common = set(S1.index) & set(S2.index)

    if len(common) == 0:
        raise KeyError("No common genes")

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


def _gene_dictionary(gene_list: List):
    """
    Given a list of genes, will return a string for what organism best matches the gene prefixes
    Args
        :param gene_list: List of genes, usually from an S matrix
        :return: String of the organism name specific to the _pull_bbh_csv function
    """
    gene_to_org_dict = {"A1S": "aBaumannii", "BSU": "bSubtilis", "MPN": "mPneumoniae", "PP": "pPutida",
                        "PSPTO": "pSyringae", "STMMW": "sEnterica_D23580", "SEN": "sEnterica_enteritidis",
                        "SL1344": "sEnterica_SL1344", "STM474": "sEnterica_ST4_74", "USA300HOU": "sAureus",
                        "SACI": "sAcidocaldarius", "SYNPCC7942": "sElongatus", "b": "eColi", "Rv": "mTuberculosis",
                        "PA": "pAeruginosa", "STM": "sEnterica_full", "SCO": "sCoelicolor"}

    org_counts = {}
    for gene in gene_list:
        try:
            curr_org = gene_to_org_dict[gene.split("_")[0]]
            if curr_org not in org_counts.keys():
                org_counts.update({curr_org: 1})
            else:
                org_counts.update({curr_org: org_counts[curr_org] + 1})
        except KeyError:
            try:
                curr_org = gene_to_org_dict[split("[0-9]", gene, maxsplit=1)[0]]
                if curr_org not in org_counts.keys():
                    org_counts.update({curr_org: 1})
                else:
                    org_counts.update({curr_org: org_counts[curr_org] + 1})
            except KeyError:
                continue
    if (org_counts[max(org_counts)] / len(gene_list)) >= .7:
        return max(org_counts)
    else:
        print("One of your org files contains too many different genes "
              +str((org_counts[max(org_counts)] / len(gene_list))))
        raise KeyError


def _pull_bbh_csv(org_1: str, org_2: str, ortho_dir: str, S1: pd.DataFrame):
    """
    Receives an the S matrix for an organism and returns the same S matrix with index genes translated into the
    orthologs in organism 2
    :param org_1: Name of organism 1 based on the following format: Ex. "eColi" "mTuberculosis". Can use output of
    _gene_dictionary function
    :param org_2: Name of organism 2 based on the following format: Ex. "eColi" "mTuberculosis". Can use output of
    _gene_dictionary function
    :param ortho_dir: String path to the "bbh_csv" directory in the "modulome_compare_data" repository.
    Ex. "../../modulome_compare_data/bbh_csv"
    :param S1: Pandas DataFrame of the S matrix for organism 1
    :return: Pandas DataFrame of the S matrix for organism 1 with indexes translated into orthologs
    """
    for dirpath, dirname, file_arr in os.walk(ortho_dir):
        for file in file_arr:
            file_split = file.split("_vs_")
            if org_1 in file_split[0] and org_2 in file_split[1]:
                bbh_csv = os.path.join(dirpath, file)
                bbh_DF = pd.read_csv(bbh_csv, index_col="gene")
                bbh_DF.drop(columns="Unnamed: 0", inplace=True)
                bbh_DF = bbh_DF.loc[:]["subject"]

                S1_copy = S1.copy()
                S1_index = list(S1_copy.index)
                for i in range(0, len(S1_index)):
                    try:
                        S1_index[i] = bbh_DF.loc[S1_index[i]]
                    except KeyError:
                        continue
                S1_copy.index = S1_index
                return S1_copy

            elif org_1 in file_split[1] and org_2 in file_split[0]:
                bbh_csv = os.path.join(dirpath, file)
                bbh_DF = pd.read_csv(bbh_csv, index_col="subject")
                bbh_DF.drop(columns="Unnamed: 0", inplace=True)
                bbh_DF = bbh_DF.loc[:]["gene"]

                S1_copy = S1.copy()
                S1_index = list(S1_copy.index)
                for i in range(0, len(S1_index)):
                    try:
                        S1_index[i] = bbh_DF.loc[S1_index[i]]
                    except KeyError:
                        continue
                S1_copy.index = S1_index
                return S1_copy


def compare_ica(S1: pd.DataFrame, S2: pd.DataFrame, ortho_dir, auto_find: bool = True, org_1_name: str = None,
                org_2_name: str = None,
                metric='pearson', cutoff=0.2, show_all=False):
    """
    Compares two S matrices between a single organism or across organisms and
    returns the connected ICA components
    Args:
        S1: Pandas Dataframe of S matrix 1
        S2: Pandas Dataframe of S Matrix 2
        ortho_dir: String of the location where organism data can be found
            (can be found under modulome/data)
        metric: A string of what statistical test to use (standard is 'pearson')
        cutoff: Float cut off value for pearson statistical test
        show_all: True will show all nodes of the digraph matrix

    Returns: Dot graph and name links of connected ICA components between the
    two runs or organisms.

    """
    if ortho_dir is None:
        dot, name_links = _make_dot_graph(S1, S2, metric=metric, cutoff=cutoff,
                                          show_all=show_all)
        return dot, name_links
    else:
        if auto_find is False and org_1_name is not None and org_2_name is not None:
            translated_S = _pull_bbh_csv(org_1_name, org_2_name, ortho_dir, S1)
            dot, name_links = _make_dot_graph(translated_S, S2, metric, cutoff,
                                              show_all=show_all)
        else:
            org_1_name = _gene_dictionary(S1.index)
            org_2_name = _gene_dictionary(S2.index)
            translated_S = _pull_bbh_csv(org_1_name, org_2_name, ortho_dir, S1)
            dot, name_links = _make_dot_graph(translated_S, S2, metric, cutoff,
                                              show_all=show_all)

        return dot, name_links
