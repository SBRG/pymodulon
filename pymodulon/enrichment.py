import itertools
import warnings
from typing import Union, Iterable
import numpy as np
import pandas as pd
from scipy import stats, special
from statsmodels.stats.multitest import fdrcorrection

ImodName = Union[str, int]


def contingency(set1: Iterable, set2: Iterable, all_genes: Iterable):
    """
    Creates contingency table for gene enrichment
    Args:
        set1: Set of genes (e.g. regulon)
        set2: Set of genes (e.g. i-modulon)
        all_genes: Set of all genes

    Returns: Contingency table

    """

    set1 = set(set1)
    set2 = set(set2)
    all_genes = set(all_genes)
    if len(set1 - all_genes) > 0 or len(set2 - all_genes) > 0:
        raise ValueError('Gene sets contain genes not in all_genes')

    tp = len(set1 & set2)
    fp = len(set2 - set1)
    tn = len(all_genes - set1 - set2)
    fn = len(set1 - set2)
    return [[tp, fp], [fn, tn]]


def compute_enrichment(gene_set, target_genes, all_genes, label=None):
    """
    Computes enrichment statistic for gene_set in target_genes.
    Args:
        gene_set: Gene set for enrichment (e.g. genes in iModulon)
        target_genes: Genes to be enriched against (e.g. genes in regulon or
            GO term)
        all_genes: List of all genes
        label: Label for target_genes (e.g. regulator name or GO term)

    Returns: Pandas Series containing enrichment statistics

    """

    # Create contingency table
    ((tp, fp), (fn, tn)) = contingency(gene_set, target_genes, all_genes)

    # Handle edge cases
    if tp == 0:
        res = [1, 0, 0, 0, 0, len(target_genes), len(gene_set)]
    elif fp == 0 and fn == 0:
        res = [0, 1, 1, 1, len(gene_set), len(target_genes), len(gene_set)]
    else:
        odds, pval = stats.fisher_exact([[tp, fp], [fn, tn]],
                                        alternative='greater')
        recall = np.true_divide(tp, tp + fn)
        precision = np.true_divide(tp, tp + fp)
        f1score = (2 * precision * recall) / (precision + recall)
        res = [pval, precision, recall, f1score, tp, len(target_genes),
               len(gene_set)]

    return pd.Series(res, index=['pvalue', 'precision', 'recall', 'f1score',
                                 'TP', 'target_set_size', 'gene_set_size'],
                     name=label)


def FDR(p_values: pd.DataFrame, fdr: float, total: int = None):
    """
    Runs false detection correction over a pandas Dataframe
    Args:
        p_values: Pandas Dataframe with 'pvalue' column
        fdr: False detection rate
        total: Total number of tests (for multi-enrichment)

    Returns: Pandas DataFrame containing entries that passed multiple
    hypothesis correction

    """

    if total is not None:
        pvals = p_values.pvalue.values.tolist() + [1] * (total - len(p_values))
    else:
        pvals = p_values.pvalue.values

    keep, qvals = fdrcorrection(pvals, alpha=fdr)

    result = p_values.copy()
    result['qvalue'] = qvals[:len(p_values)]
    result = result[keep[:len(p_values)]]

    return result.sort_values('qvalue')


def parse_regulon_str(regulon_str: str, trn: pd.DataFrame) -> set:
    """
    Converts a complex regulon (regulon_str) into a list of genes
    Args:
        regulon_str: Complex regulon, where "/" uses genes in any regulon and
            "+" uses genes in all regulons
        trn: Pandas dataframe containing transcriptional regulatory network

    Returns: Set of genes regulated by regulon_str

    """

    if regulon_str == '':
        return set()
    if '+' in regulon_str and '/' in regulon_str:
        raise NotImplementedError('Complex regulons cannot contain both "+" '
                                  '(AND) and "/" (OR) operators')
    elif '+' in regulon_str:
        join = set.intersection
        regs = regulon_str.split('+')
    elif '/' in regulon_str:
        join = set.union
        regs = regulon_str.split('/')
    else:
        join = set.union
        regs = [regulon_str]

    # Combine regulon
    reg_genes = join(*[set(trn[trn.regulator == reg].gene_id) for reg in regs])
    return reg_genes


def compute_regulon_enrichment(gene_set: Iterable, regulon_str: str,
                               all_genes: Iterable, trn: pd.DataFrame):
    """
    Computes enrichment statistics for a gene_set in a regulon
    Args:
        gene_set: Gene set for enrichment (e.g. genes in iModulon)
        regulon_str: Complex regulon, where "/" uses genes in any regulon and
            "+" uses genes in all regulons
        all_genes: List of all genes
        trn: Pandas dataframe containing transcriptional regulatory network

    Returns: Pandas dataframe containing enrichment statistics

    """

    regulon = parse_regulon_str(regulon_str, trn)
    # Remove genes in regulon that are not in all_genes
    if len(regulon - set(all_genes)) > 0:
        warnings.warn('Some genes are in the regulon but not in all_genes. '
                      'These genes are removed before enrichment analysis.',
                      category=UserWarning)
        regulon = regulon & set(all_genes)
    result = compute_enrichment(gene_set, regulon, all_genes, regulon_str)
    result.rename({'target_set_size': 'regulon_size'}, inplace=True)
    n_regs = 1 + regulon_str.count('+') + regulon_str.count('/')
    result['n_regs'] = n_regs
    return result


def compute_trn_enrichment(gene_set: Iterable, all_genes: Iterable,
                           trn: pd.DataFrame,
                           max_regs: int = 1, fdr: float = 0.01,
                           method: str = 'both', force: bool = False):
    """
    Compare a gene set against an entire TRN
    Args:
        gene_set: Gene set for enrichment (e.g. genes in iModulon)
        all_genes: List of all genes
        trn: Pandas dataframe containing transcriptional regulatory network
        max_regs: Maximum number of regulators to include in complex regulon
            (default: 1)
        fdr: False detection rate (default: 0.01)
        method: How to combine complex regulons. 'or' computes enrichment
            against union of regulons, 'and' computes enrichment against
            intersection of regulons, and 'both' performs both tests
            (default: 'both')
        force: Allows computation of >2 regulators

    Returns: Pandas dataframe containing statistically significant enrichments

    """

    # Warning if max_regs is too high
    if max_regs > 2 and not force:
        ValueError(
            'Using >2 maximum regulators may take time to compute. '
            'To perform analysis, use force=True')

    # Only search for regulators known to regulate a gene in gene_set
    # This reduces the total runtime by skipping unnecessary tests
    # However, this needs to be taken into account for FDR
    imod_regs = trn[trn.gene_id.isin(gene_set)].regulator.unique()

    # # Account for imodulons with no regulators
    # if len(imod_regs) == 0:
    #     return compute_regulon_enrichment(gene_set, '', all_genes, trn)

    # Prepare complex regulon names
    enrich_list = []
    total = 0

    for n_regs in range(1, max_regs + 1):
        group = itertools.combinations(imod_regs, n_regs)
        num_tests = int(special.comb(len(trn.regulator.unique()), n_regs))

        if method == 'and':
            reg_list = ['+'.join(regs) for regs in group]
            total += num_tests
        elif method == 'or':
            reg_list = ['/'.join(regs) for regs in group]
            total += num_tests
        elif method == 'both':
            reg_list = ['+'.join(regs) for regs in group] + ['/'.join(regs) for
                                                             regs in group]
            total += 2 * num_tests
        else:
            raise ValueError("'method' must be either 'and', 'or', or 'both'")

        # Perform enrichments
        for reg in reg_list:
            enrich_list.append(
                compute_regulon_enrichment(gene_set, reg, all_genes, trn))

    if len(enrich_list) == 0:
        return pd.DataFrame()
    df_enrich = pd.concat(enrich_list, axis=1).T
    return FDR(df_enrich, fdr=fdr, total=total)


def compute_annotation_enrichment(gene_set: Iterable, all_genes: Iterable,
                                  annotation: pd.DataFrame,
                                  column='annotation',
                                  fdr: float = 0.01):
    """
    Compare a gene set against a dataframe of gene annotations
    Args:
        gene_set: Gene set for enrichment (e.g. genes in iModulon)
        all_genes: List of all genes
        annotation: Table containing gene annotations
        column: Name of column in the annotation DataFrame (default =
        'annotation')
        fdr: False detection rate (default = 0.01)

    Returns: Pandas dataframe containing statistically significant enrichments
    """
    # TODO: Create test functions
    enrich_list = []
    for name, group in annotation.groupby(column):
        target_genes = group['gene_id']
        enrich_list.append(compute_enrichment(gene_set, target_genes,
                                              all_genes, label=name))

    df_enrich = pd.concat(enrich_list, axis=1).T
    return FDR(df_enrich, fdr=fdr)
