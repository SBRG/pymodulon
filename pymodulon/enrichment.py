import sys

from statsmodels.stats.multitest import fdrcorrection
from scipy import stats, special
import numpy as np
import pandas as pd
from typing import Union, List
import itertools
from warnings import warn

ImodName = Union[str, int]


def contingency(set1: List, set2: List, all_genes: List):
    """
    Creates contingency table for gene enrichment
    :param set1: Set of genes (e.g. regulon)
    :param set2: Set of genes (e.g. i-modulon)
    :param all_genes: Set of all genes
    :return: Contingency table
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
    :param gene_set: Gene set for enrichment (e.g. genes in iModulon)
    :param target_genes: Genes to be enriched against (e.g. genes in regulon or GO term)
    :param all_genes: List of all genes
    :param label: Label for target_genes (e.g. regulator name or GO term)
    :return: Pandas Series containing enrichment statistics
    """

    # Create contingency table
    ((tp, fp), (fn, tn)) = contingency(gene_set, target_genes, all_genes)

    # Handle edge cases
    if tp == 0:
        res = [1, 0, 0, 0, 0, len(target_genes), len(gene_set)]
    elif fp == 0 and fn == 0:
        res = [0, 1, 1, 1, len(gene_set), len(target_genes), len(gene_set)]
    else:
        odds, pval = stats.fisher_exact([[tp, fp], [fn, tn]], alternative='greater')
        recall = np.true_divide(tp, tp + fn)
        precision = np.true_divide(tp, tp + fp)
        f1score = (2 * precision * recall) / (precision + recall)
        res = [pval, precision, recall, f1score, tp, len(target_genes), len(gene_set)]

    return pd.Series(res, index=['pvalue', 'precision', 'recall', 'f1score', 'TP', 'target_set_size', 'gene_set_size'],
                     name=label)


def FDR(p_values: pd.DataFrame, fdr: float, total: int = None):
    """Runs false detection correction over a pandas Dataframe
    :param p_values: Pandas Dataframe with 'pvalue' column
    :param fdr: False detection rate
    :param total: Total number of tests (for multi-enrichment)
    :return: Pandas DataFrame containing entries that passed multiple hypothesis correction
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
    :param regulon_str: Complex regulon, where "/" uses genes in any regulon and "+" uses genes in all regulons
    :param trn: Pandas dataframe containing transcriptional regulatory network
    :return: Set of genes regulated by regulon_str
    """

    if regulon_str == '':
        return set()
    if '+' in regulon_str and '/' in regulon_str:
        raise NotImplementedError('Complex regulons cannot contain both "+" (AND) and "/" (OR) operators')
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


def compute_regulon_enrichment(gene_set: List, regulon_str: str, all_genes: List, trn: pd.DataFrame):
    """
    Computes enrichment statistics for a gene_set in a regulon
    :param gene_set: Gene set for enrichment (e.g. genes in iModulon)
    :param regulon_str: Complex regulon, where "/" uses genes in any regulon and "+" uses genes in all regulons
    :param all_genes: List of all genes
    :param trn: Pandas dataframe containing transcriptional regulatory network
    :return: Pandas dataframe containing enrichment statistics
    """
    regulon = parse_regulon_str(regulon_str, trn)
    # Remove genes in regulon that are not in all_genes
    if len(regulon - set(all_genes)) > 0:
        warn('Some genes are in the regulon but not in all_genes. These genes are removed before enrichment '
             'analysis.', category=UserWarning)
        regulon = regulon & set(all_genes)
    result = compute_enrichment(gene_set, regulon, all_genes, regulon_str)
    result.rename({'target_set_size': 'regulon_size'}, inplace=True)
    n_regs = 1 + regulon_str.count('+') + regulon_str.count('/')
    result['n_regs'] = n_regs
    return result


def compute_trn_enrichment(gene_set: List, all_genes: List, trn: pd.DataFrame, max_regs: int = 1, fdr: float = 0.01,
                           method: str = 'both', force: bool = False):
    """
    Compare a gene set against an entire TRN
    :param gene_set: Gene set for enrichment (e.g. genes in iModulon)
    :param all_genes: List of all genes
    :param trn: Pandas dataframe containing transcriptional regulatory network
    :param max_regs: Maximum number of regulators to include in complex regulon (default: 1)
    :param method: How to combine complex regulons. 'or' computes enrichment against union of regulons, \
    'and' computes enrichment against intersection of regulons, and 'both' performs both tests (default: 'both')
    :param fdr: False detection rate
    :param force: Allows computation of >2 regulators
    :return: Pandas dataframe containing statistically significant enrichments
    """

    # Warning if max_regs is too high
    if max_regs > 2:
        warn('Using >2 maximum regulators may take time to compute. To perform analysis, use force=True',
             category=RuntimeWarning)
        if not force:
            return

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
            reg_list = ['+'.join(regs) for regs in group] + ['/'.join(regs) for regs in group]
            total += 2 * num_tests
        else:
            raise ValueError("'method' must be either 'and', 'or', or 'both'")

        # Perform enrichments
        for reg in reg_list:
            enrich_list.append(compute_regulon_enrichment(gene_set, reg, all_genes, trn))

    if len(enrich_list) == 0:
        return pd.DataFrame()
    df_enrich = pd.concat(enrich_list, axis=1).T
    return FDR(df_enrich, fdr=fdr, total=total)
