import os

import pytest
from Bio import SeqIO

import pymodulon.compare as cmp
from pymodulon.core import IcaData
from pymodulon.example_data import load_example_bbh, load_staph_data


@pytest.fixture
def staph_obj():
    return load_staph_data()


@pytest.fixture
def example_bbh():
    return load_example_bbh()


def test_convert_gene_index(staph_obj, example_bbh, ecoli_obj):
    ten_genes = IcaData(
        ecoli_obj.M.iloc[:10, :10],
        ecoli_obj.A.iloc[:10, :10],
        gene_table=ecoli_obj.gene_table.iloc[:10],
        sample_table=ecoli_obj.sample_table.iloc[:10],
        imodulon_table=ecoli_obj.imodulon_table.iloc[:10],
    )

    # Test conforming M matrix from same organism
    M1, M2 = cmp.convert_gene_index(ecoli_obj.M, ten_genes.M)
    assert (M1.index == M2.index).all()
    assert len(M1.index) == 10

    # Test conforming M matrix from different organisms
    orgM1, orgM2 = cmp.convert_gene_index(
        ecoli_obj.M, staph_obj.M, ortho_file=example_bbh
    )
    assert (orgM1.index == orgM2.index).all()
    assert len(orgM1) > 10

    # Test conforming gene info data between organisms
    org_table1, org_table2 = cmp.convert_gene_index(
        ecoli_obj.gene_table,
        staph_obj.gene_table,
        ortho_file=example_bbh,
    )

    assert (org_table1.index == org_table2.index).all()
    assert (org_table1.index == orgM1.index).all()


@pytest.mark.filterwarnings("ignore:BiopythonWarning")
def test_make_prots_faout(tmp_path):
    gbk = os.path.join("tests", "data", "genome.gb")
    fa_out = tmp_path / "test.fa"
    fa_out.touch()
    cmp.make_prots(gbk, fa_out)

    with open(fa_out) as fa:
        line = fa.readline().strip()
        assert line == ">b0001"

    rsid = []
    for refseq in SeqIO.parse(fa_out, "fasta"):
        rsid.append(refseq.id)
    assert len(rsid) == len(set(rsid))


# create single and multi-file params for test_make_prots_db

single_file = {
    "fasta_file": os.path.join("tests", "data", "proteins.faa"),
    "outname": "test_db.fa",
}
multi_file = {
    "fasta_file": [
        os.path.join("tests", "data", "proteins.faa"),
        os.path.join("tests", "data", "truncated_proteins.faa"),
    ],
    "outname": "test_db.fa",
    "combined": "combined.fa",
}

make_prots_test = [(single_file, "test_db.fa.pin"), (multi_file, "test_db.fa.pin")]


@pytest.mark.parametrize("arg, expected", make_prots_test)
def test_make_prots_db(tmp_path, arg, expected):
    # update tmp_dirs for outputs
    arg["outname"] = str(tmp_path / arg["outname"])
    expected = tmp_path / expected
    try:
        arg["combined"] = str(tmp_path / arg["combined"])
    except KeyError:
        pass

    cmp.make_prot_db(**arg)
    assert os.path.isfile(expected)


# def test_compare_ica():
#     S2 = pd.read_csv("data/mtb_S.csv", index_col=0)
#     S1 = pd.read_csv("data/ecoli_S.csv", index_col=0)
#     ortho_dir = (
#         "/home/sbrg/Desktop/modulome/data/organism_compare/"
#         "bbh_csv/eColi_full_protein_vs"
#         "_mTuberculosis_full_protein_parsed.csv"
#     )
#     dot, links = compare_ica(S1, S2, ortho_file=ortho_dir)
