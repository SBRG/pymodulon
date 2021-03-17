import pandas as pd

from pymodulon.enrichment import (
    compute_annotation_enrichment,
    compute_enrichment,
    compute_trn_enrichment,
)


def test_compute_enrichment():
    """
    Test for compute_enrichment() that computes statistic for gene_set in target_genes in a pd.Series.

    """
    # creating test datasets
    test_all_genes = ["gene_" + str(i) for i in range(1, 11)]  # complete set
    test_target_set = test_all_genes[0:3]  # target set]
    test_gene_set1 = test_all_genes[0:3]  # complete overlap to target set
    test_gene_set2 = test_all_genes[0:5]  # genes set > target set
    test_gene_set3 = test_all_genes[0:1]  # genes set < target set
    test_gene_set4 = ["gene_9", "gene_10"]  # no overlapping genes to target set

    # calling original method
    res1 = compute_enrichment(test_gene_set1, test_target_set, test_all_genes)
    res2 = compute_enrichment(test_gene_set2, test_target_set, test_all_genes)
    res3 = compute_enrichment(test_gene_set3, test_target_set, test_all_genes)
    res4 = compute_enrichment(test_gene_set4, test_target_set, test_all_genes)

    # assert statements
    assert (
        res1.pvalue == 0
        and res1.precision == 1
        and res1.recall == 1
        and res1.f1score == 1
        and res1.TP == 3
        and res1.target_set_size == 3
        and res1.gene_set_size == 3
    )

    assert (
        0 <= res2.pvalue <= 1
        and 0 <= res2.precision <= 1
        and res2.recall == 1
        and 0 <= res2.f1score <= 1
        and res2.TP == 3
        and res2.target_set_size == 3
        and res2.gene_set_size == 5
    )

    assert (
        0 <= res3.pvalue <= 1
        and res3.precision == 1
        and 0 <= res3.recall <= 1
        and 0 <= res3.f1score <= 1
        and res3.TP == 1
        and res3.target_set_size == 3
        and res3.gene_set_size == 1
    )

    assert (
        res4.pvalue == 1
        and res4.precision == 0
        and res4.recall == 0
        and res4.f1score == 0
        and res4.TP == 0
        and res4.target_set_size == 3
        and res4.gene_set_size == 2
    )


def test_compute_trn_enrichment():
    """
    Test for compute_trn_enrichment() that Compare a gene set against an entire TRN.
    """
    # creating test datasets
    test_all_genes_list = ["gene_" + str(i) for i in range(1, 16)]
    test_all_genes = set(test_all_genes_list)
    test_gene_set1 = set(test_all_genes_list[0:5])  # complete overlap with reg_1
    test_gene_set2 = set(test_all_genes_list[10:13])  # no overlap/matches with TRN
    test_gene_set3 = set(test_all_genes_list[1:5])  # some overlap with reg_1
    test_gene_set4 = set(test_all_genes_list[0:1])  # gene1 with max_regs=2
    test_gene_set5 = set(test_all_genes_list[5:6])  # gene6 & gene7 with max_regs>2

    test_trn = pd.DataFrame(
        [
            ["reg_1", "gene_1"],
            ["reg_1", "gene_2"],
            ["reg_1", "gene_3"],
            ["reg_1", "gene_4"],
            ["reg_1", "gene_5"],
            ["reg_2", "gene_1"],
            ["reg_2", "gene_6"],
            ["reg_3", "gene_7"],
            ["reg_4", "gene_6"],
            ["reg_5", "gene_8"],
            ["reg_6", "gene_2"],
            ["reg_7", "gene_6"],
            ["reg_2", "gene_7"],
            ["reg_4", "gene_7"],
        ],
        columns=["regulator", "gene_id"],
    )

    res1 = compute_trn_enrichment(test_gene_set1, test_all_genes, test_trn)
    res2 = compute_trn_enrichment(test_gene_set2, test_all_genes, test_trn)
    res3 = compute_trn_enrichment(test_gene_set3, test_all_genes, test_trn)
    res4 = compute_trn_enrichment(test_gene_set4, test_all_genes, test_trn, max_regs=2)
    res5 = compute_trn_enrichment(
        test_gene_set5, test_all_genes, test_trn, max_regs=3, force=True
    )

    assert (
        res1.index == "reg_1"
        and res1.pvalue[0] == 0
        and res1.precision[0] == 1.0
        and res1.recall[0] == 1.0
        and res1.f1score[0] == 1.0
        and res1.TP[0] == 5.0
        and res1.regulon_size[0] == 5.0
        and res1.gene_set_size[0] == 5.0
        and res1.n_regs[0] == 1.0
        and res1.qvalue[0] == 0.0
    )

    assert res2.empty

    assert (
        res3.index == "reg_1"
        and 0 <= res3.pvalue[0] <= 1
        and res3.precision[0] == 1
        and 0 <= res3.recall[0] <= 1
        and 0 <= res3.f1score[0] <= 1
        and res3.TP[0] == 4
        and res3.regulon_size[0] == 5
        and res3.gene_set_size[0] == 4
        and res3.n_regs[0] == 1
        and 0 <= res3.qvalue[0] <= 1
    )

    assert (
        res4.index == "reg_1+reg_2"
        and res4.pvalue[0] == 0
        and res4.precision[0] == 1
        and res4.recall[0] == 1
        and res4.f1score[0] == 1
        and res4.TP[0] == 1
        and res4.regulon_size[0] == 1
        and res4.gene_set_size[0] == 1
        and res4.n_regs[0] == 2
        and res4.qvalue[0] == 0
    )

    assert res5.shape == (4, 9)


def test_compute_annotation_enrichment():
    test_all_genes_list = ["gene_" + str(i) for i in range(1, 16)]
    test_all_genes = set(test_all_genes_list)
    test_gene_set1 = set(test_all_genes_list[0:5])  # complete overlap with annot_1
    test_gene_set2 = set(test_all_genes_list[10:13])  # no overlap/matches found
    test_gene_set3 = set(test_all_genes_list[1:5])  # some overlap with annot_1
    test_gene_set4 = {"gene_6", "gene_7"}  # few genes with more than 1 matches

    test_annot = pd.DataFrame(
        [
            ["annot_1", "gene_1"],
            ["annot_1", "gene_2"],
            ["annot_1", "gene_3"],
            ["annot_1", "gene_4"],
            ["annot_1", "gene_5"],
            ["annot_2", "gene_1"],
            ["annot_2", "gene_6"],
            ["annot_3", "gene_7"],
            ["annot_4", "gene_6"],
            ["annot_5", "gene_8"],
            ["annot_6", "gene_2"],
            ["annot_7", "gene_6"],
            ["annot_2", "gene_7"],
            ["annot_4", "gene_7"],
        ],
        columns=["gene_annotation", "gene_id"],
    )

    res1 = compute_annotation_enrichment(
        test_gene_set1, test_all_genes, test_annot, column="gene_annotation"
    )
    res2 = compute_annotation_enrichment(
        test_gene_set2, test_all_genes, test_annot, column="gene_annotation"
    )
    res3 = compute_annotation_enrichment(
        test_gene_set3, test_all_genes, test_annot, column="gene_annotation"
    )
    res4 = compute_annotation_enrichment(
        test_gene_set4, test_all_genes, test_annot, column="gene_annotation"
    )

    assert (
        res1.index == "annot_1"
        and res1.pvalue[0] == 0
        and res1.precision[0] == 1
        and res1.recall[0] == 1
        and res1.f1score[0] == 1
        and res1.TP[0] == 5
        and res1.target_set_size[0] == 5
        and res1.gene_set_size[0] == 5
        and res1.qvalue[0] == 0
    )

    assert res2.empty

    assert res3.empty

    assert (
        res4.index == "annot_4"
        and res4.pvalue[0] == 0
        and res4.precision[0] == 1
        and res4.recall[0] == 1
        and res4.f1score[0] == 1
        and res4.TP[0] == 2
        and res4.target_set_size[0] == 2
        and res4.gene_set_size[0] == 2
        and res4.qvalue[0] == 0
    )
