import numpy as np
import pandas as pd
from scipy import stats
from typing import *
from warnings import warn
import os

ImodName = Union[str, int]
ImodNameList = Union[ImodName, List[ImodName]]
Data = Union[pd.DataFrame, os.PathLike]


def _check_table(table: Data, name: str, index: Optional[Collection] = None):
    # Set as empty dataframe if not input given
    if table is None:
        return pd.DataFrame(index=index)

    # Load table if necessary
    elif isinstance(table, str):
        try:
            table = pd.read_json(table)
        except ValueError:
            try:
                sep = '\t' if table.endswith('.tsv') else ','
                table = pd.read_csv(table, index_col=0, sep=sep)
            except FileNotFoundError:
                raise TypeError('{}_table must be a pandas DataFrame '
                                'filename or a valid JSON string'.format(name))

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
        warn('Some {} are missing from the {} table: {}'.format(name, name,
                                                                missing_index))

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


######################
## Compare ICA runs ##
######################

from graphviz import Digraph
from scipy.cluster.hierarchy import linkage, dendrogram
from tqdm import notebook as tqdm
from scipy import sparse


def _make_dot_graph(S1: pd.DataFrame, S2: pd.DataFrame, metric: str, cutoff: float):
    """
    Given two S matrices, returns the dot graph and name links of the various connected ICA components
    :param S1: S matrix from the first organism
    :param S2: S matrix from the second organism
    :param metric: A string of what statistical test to use (standard is 'pearson')
    :param cutoff: Float cut off value for pearson statistical test
    :return: Dot graph and name links of connected ICA components
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

    for i, k1 in tqdm.tqdm(enumerate(s1.columns), total=len(s1.columns)):
        for j, k2 in enumerate(s2.columns):
            if metric == 'pearson':
                corr[i, j] = abs(stats.pearsonr(s1[k1], s2[k2])[0])

    DF_corr = pd.DataFrame(corr, index=s1.columns, columns=s2.columns)  # Only keep genes found in both S matrices

    # Initialize Graph
    dot = Digraph(engine='dot', graph_attr={'ranksep': '0.3', 'nodesep': '0', 'packmode': 'array_u', 'size': '7,7'},
                  node_attr={'fontsize': '14', 'shape': 'none'},
                  edge_attr={'arrowsize': '0.5'}, format='png')

    # Set up linkage and designate terminal nodes
    loc1, loc2 = np.where(DF_corr > cutoff)
    links = list(zip(s1.columns[loc1], s2.columns[loc2]))

    if len(links) == 0:
        warn('No components shared across runs')
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


def _load_ortho_matrix(ortho_dir: str):
    """
    Load the .npz file and organism labels and compiles them into one
    :param ortho_dir: String of the location where organism data can be found (can be found under modulome/data)
    :return: Pandas Dataframe of the full organism compare matrix
    """
    filename = os.path.join(ortho_dir, "org_compare.npz")
    label_file = os.path.join(ortho_dir, "org_compare_label.txt")
    ortho_DF = pd.DataFrame(sparse.load_npz(filename).toarray())
    labels = list(pd.read_csv(label_file, header=None, nrows=1).loc[0][:])

    ortho_DF.index = labels
    ortho_DF.columns = labels

    # Filter out different strains of Salmonella, only leaves LT2 strain
    filter = []
    for i in ortho_DF.index:
        if "STM" in i and "_" in i:
            filter.append(i)

    ortho_DF.drop(filter, inplace=True)
    ortho_DF.drop(columns=filter, inplace=True)

    return ortho_DF


def _extract_genes(gene_set_1: List, gene_set_2: List, ortho_DF):
    """
    Returns a Panda Series that contains a cut down version of the complete Comparison DF, where the rows are only genes
    found in the I-modulon component
    :param gene_set_1: list of genes from organism 1
    :param gene_set_2: list of genes from organism 2
    :param ortho_DF: Full ortholog comparison Dataframe
    :return: A reduced version of the ortho_DF Dataframe that only contains the genes from gene_set_1 and gene_set_2
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
    Converts genes from one list to their corresponding ortholog of another organism
    :param gene_list: List of genes to be translated
    :param reduced_DF: Pandas Dataframe of the orthologs between your two target organisms
    :return: List of genes that have been translated
    """
    gene_list_copy = gene_list
    for i in range(0, len(gene_list_copy)):
        try:
            convert_column = reduced_DF[gene_list_copy[i]]
            gene_list_copy[i] = str(convert_column.loc[convert_column == 1][:].index[0])
        except KeyError as e:
            continue
    return gene_list_copy


def compare_ica(S1: pd.DataFrame, S2: pd.DataFrame, metric='pearson', cutoff=0.2, ortho_dir=None):
    """
    Compares two S matrices between a single organism or across organisms and returns the connected ICA components
    :param S1: Pandas Dataframe of S matrix 1
    :param S2: Pandas Dataframe of S Matrix 2
    :param metric: A string of what statistical test to use (standard is 'pearson')
    :param cutoff: Float cut off value for pearson statistical test
    :param ortho_dir: String of the location where organism data can be found (can be found under modulome/data)
    :return: Dot graph and name links of connected ICA components between the two runs or organisms.
    """
    if ortho_dir is None:
        dot, name_links = _make_dot_graph(S1, S2, DF_corr, cutoff)
        return dot, name_links
    else:
        ortho_DF = _load_ortho_matrix(ortho_dir)
        ortho_reduced_DF = _extract_genes(list(S1.index), list(S2.index), ortho_DF)
        translated_genes = _translate_genes(list(S2.index), ortho_reduced_DF)
        S2.index = translated_genes
        dot, name_links = _make_dot_graph(S1, S2, metric, cutoff)
        return dot, name_links
