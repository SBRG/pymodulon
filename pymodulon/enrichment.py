from statsmodels.stats.multitest import fdrcorrection
from scipy import stats
import numpy as np
import pandas as pd


def contingency(set1, set2, all_genes):
    """Creates contingency table for gene enrichment
        set1: Set of genes (e.g. regulon)
        set2: Set of genes (e.g. i-modulon)
        all_genes: Set of all genes
    """

    tp = len(set1 & set2)
    fp = len(set2 - set1)
    tn = len(all_genes - set1 - set2)
    fn = len(set1 - set2)
    return [[tp, fp], [fn, tn]]


def FDR(p_values, fdr_rate, total=None):
    """Runs false detection correction over a pandas Dataframe
        p_values: Pandas Dataframe with 'pvalue' column
        fdr_rate: False detection rate
        total: Total number of tests (for multi-enrichment)
    """

    if total is not None:
        pvals = p_values.pvalue.values.tolist() + [1] * (total - len(p_values))
    else:
        pvals = p_values.pvalue.values

    keep, qvals = fdrcorrection(pvals, alpha=fdr_rate)

    result = p_values.copy()
    result['qvalue'] = qvals[:len(p_values)]
    result = result[keep[:len(p_values)]]

    return result.sort_values('qvalue')


def compute_enrichment(imodulon_genes, gene_set, all_genes, label=None):
    """ Calculates the enrichment of an iModulon against a set of genes
        imodulon_genes: Genes in iModulon
        gene_set: Genes in comparison set
        all_genes: List of all genes in organism
        label: Name of comparison set (e.g. TF name, GO term)
    """

    # Create contingency table
    ((tp, fp), (fn, tn)) = contingency(imodulon_genes, gene_set, all_genes)

    # Handle edge cases
    if tp == 0:
        res = [0, 1, 0, 0, 0]
    elif fp == 0 and fn == 0:
        res = [np.inf, 0, 1, 1, len(imodulon_genes)]
    else:
        odds, pval = stats.fisher_exact([[tp, fp], [fn, tn]], alternative='greater')
        recall = np.true_divide(tp, tp + fn)
        precision = np.true_divide(tp, tp + fp)
        res = [np.log(odds), pval, recall, precision, tp]

    return pd.Series(res, index=['log_odds', 'pvalue', 'recall', 'precision', 'TP'],
                     name=label)


def compute_simple_regulon_enrichment(ica_data, imodulon, regulator):
    pass


def compute_complex_regulon_enrichment(ica_data, imodulon, regulator_str):
    pass
