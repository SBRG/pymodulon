import os
import urllib.request
from os.path import abspath, dirname, isfile, join

import pandas as pd
import pytest

from pymodulon.gene_util import (
    _get_attr,
    cog2str,
    gff2pandas,
    reformat_biocyc_tu,
    uniprot_id_mapping,
)


PYMOD_DIR = abspath(join(dirname(abspath(__file__)), ".."))
"""str: The directory location of where :mod:`pymodulon` is installed."""

TEST_DIR = join(PYMOD_DIR, "tests")
"""str: The directory location of the test directory."""

TEST_DATA_DIR = join(TEST_DIR, "data")
"""str: The directory location of the data directory."""


def test_cog2str():
    cog_list = ["A", "B", "Z"]
    res1 = [cog2str(i) for i in cog_list]
    assert (
        len(res1) == 3
        and res1[0] == "RNA processing and modification"
        and res1[1] == "Chromatin structure and dynamics"
        and res1[2] == "Cytoskeleton"
    )


def test_get_attr():
    valid_test_input = (
        "NC_007604.1	RefSeq	gene	4596	6077	.	+	.	"
        "ID=gene-1;Name=ABC_1;gbkey="
        "Gene;gene_biotype=protein_coding;locus_tag=LT_1"
        ";old_locus_tag=OLT_1"
    )

    invalid_test_input = "NC_007604.1	RefSeq	"

    assert (
        _get_attr(valid_test_input, "ID", ignore=False) == "gene-1"
        and _get_attr(valid_test_input, "Name", ignore=False) == "ABC_1"
        and _get_attr(valid_test_input, "gbkey", ignore=False) == "Gene"
        and _get_attr(valid_test_input, "gene_biotype", ignore=False)
        == "protein_coding"
        and _get_attr(valid_test_input, "locus_tag", ignore=False) == "LT_1"
        and _get_attr(valid_test_input, "old_locus_tag", ignore=False) == "OLT_1"
    )
    with pytest.raises(
        ValueError, match=r"ID not in attributes: NC_007604.1\tRefSeq\t"
    ):
        _get_attr(invalid_test_input, "ID", ignore=False)


def test_reformat_biocyc_tu():
    test_input = "thrA // thrB // thrC // thrL"
    test_input2 = "thrA . thrB . thrC . thrL"
    res1 = reformat_biocyc_tu(test_input)
    assert res1 == "thrA;thrB;thrC;thrL"
    with pytest.raises(AttributeError, match=None):
        reformat_biocyc_tu(test_input2)


def test_uniprot_id_mapping():
    test_df = pd.DataFrame(
        [
            ["LT_1", "WP_011243806.1"],
            ["LT_2", "WP_011243805.1"],
            ["LT_3", "WP_011243804.1"],
        ],
        columns=["locus_tag", "ncbi_protein"],
    )

    res1 = uniprot_id_mapping(
        test_df.ncbi_protein,
        input_id="P_REFSEQ_AC",
        output_id="ACC",
        input_name="ncbi_protein",
        output_name="uniprot",
    )
    assert (
        len(res1.uniprot) == 3
        and res1.uniprot[0] == "UPI000049B79B"
        and res1.uniprot[1] == "A0A0H3K3Q0"
        and res1.uniprot[3] == "Q55041"
    )


def test_gff2pandas():
    test_file_dir = join(TEST_DATA_DIR, "test_genome_dup.gff3")
    res1 = gff2pandas(test_file_dir, feature="CDS", index=None)
    res2 = gff2pandas(
        test_file_dir, feature="CDS", index="locus_tag"
    )  # skips duplicated indicies
    assert (
        res1.shape == (6, 14)
        and res2.shape == (3, 14)
        and res2.iloc[0].gene_name == "thrL"
        and res2.iloc[0].locus_tag == "b0001"
        and res2.iloc[0].gene_product == "thr operon leader peptide"
        and res2.iloc[0].ncbi_protein == "NP_414542.1"
        and res2.iloc[0].old_locus_tag is None
    )
