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


####################
# Compare ICA runs #
####################

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

    for i, k1 in tqdm(enumerate(s1.columns), total=len(s1.columns)):
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


def _load_ortho_matrix(ortho_dir: str):
    """
    Load the .npz file and organism labels and compiles them into one
    Args:
        ortho_dir: String of the location where organism data can be found
        (can be found under modulome/data)

    Returns: Pandas Dataframe of the full organism compare matrix

    """

    filename = os.path.join(ortho_dir, "org_compare.npz")
    label_file = os.path.join(ortho_dir, "org_compare_label.txt")
    ortho_DF = pd.DataFrame(sparse.load_npz(filename).toarray())
    labels = list(pd.read_csv(label_file, header=None, nrows=1).loc[0][:])

    ortho_DF.index = labels
    ortho_DF.columns = labels

    # Filter out different strains of Salmonella, only leaves LT2 strain
    filter_strain = []
    for i in ortho_DF.index:
        if "STM" in i and "_" in i:
            filter_strain.append(i)

    ortho_DF.drop(filter_strain, inplace=True)
    ortho_DF.drop(columns=filter_strain, inplace=True)

    return ortho_DF


def _extract_genes(gene_set_1: List, gene_set_2: List, ortho_DF):
    """
    Returns a Panda Series that contains a cut down version of the complete
    Comparison DF, where the rows are only genes
    Args:
        gene_set_1: List of genes from organism 1
        gene_set_2: list of genes from organism 2
        ortho_DF: Full ortholog comparison Dataframe

    Returns: A reduced version of the ortho_DF Dataframe that only contains
    the genes from gene_set_1 and gene_set_2

    """

    rows = ortho_DF.loc[ortho_DF.index.isin(list(gene_set_1))][:].index
    columns = ortho_DF.loc[:][ortho_DF.columns.isin(list(gene_set_2))].index

    reduced_DF = ortho_DF.loc[rows][columns]

    reduced_DF = reduced_DF.replace(0, np.nan)
    reduced_DF = reduced_DF.dropna(axis=1, how="all")
    reduced_DF = reduced_DF.replace(np.nan, 0)

    return reduced_DF


def _translate_genes(gene_list: List, reduced_DF: pd.DataFrame):
    """
    Converts genes from one list to their corresponding ortholog of another
    organism

    Args:
        gene_list: List of genes to be translated
        reduced_DF: Pandas Dataframe of the orthologs between your two target
            organisms

    Returns: List of genes that have been translated

    """
    gene_list_copy = gene_list
    for i in range(0, len(gene_list_copy)):
        try:
            convert_column = reduced_DF[gene_list_copy[i]]
            gene_list_copy[i] = str(
                convert_column.loc[convert_column == 1][:].index[0])
        except KeyError:
            continue
    return gene_list_copy


def compare_ica(S1: pd.DataFrame, S2: pd.DataFrame, ortho_dir, metric='pearson',
                cutoff=0.2, show_all=False):
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
        ortho_DF = _load_ortho_matrix(ortho_dir)
        ortho_reduced_DF = _extract_genes(list(S1.index), list(S2.index),
                                          ortho_DF)
        translated_genes = _translate_genes(list(S2.index), ortho_reduced_DF)
        S2.index = translated_genes
        dot, name_links = _make_dot_graph(S1, S2, metric, cutoff,
                                          show_all=show_all)
        return dot, name_links
