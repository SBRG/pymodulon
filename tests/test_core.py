import numpy as np

from pymodulon.core import IcaData


def test_m_binarized(ecoli_obj):
    binary_m = ecoli_obj.M_binarized
    assert binary_m["GlpR"].sum() == 9


def test_imodulon_names(ecoli_obj, recwarn):
    # Ensure that iModulon names are consistent
    imodulon_list = ecoli_obj.imodulon_names
    assert (
        imodulon_list == ecoli_obj.M.columns.tolist()
        and imodulon_list == ecoli_obj.A.index.tolist()
        and imodulon_list == ecoli_obj.imodulon_table.index.tolist()
    )

    # Test imodulon_names setter
    ecoli_obj.imodulon_names = ["test"] * 92
    assert ecoli_obj.imodulon_names[0] == "test-1"
    assert ecoli_obj.imodulon_names[71] == "test-72"

    w = recwarn.pop(UserWarning)
    assert issubclass(w.category, UserWarning)
    assert str(w.message).startswith("Duplicate iModulon names detected.")


def test_sample_names(ecoli_obj):
    # Ensure that sample names are consistent
    sample_list = ecoli_obj.sample_names
    assert (
        sample_list == ecoli_obj.X.columns.tolist()
        and sample_list == ecoli_obj.A.columns.tolist()
        and sample_list == ecoli_obj.sample_table.index.tolist()
    )


def test_gene_names(ecoli_obj):
    # Ensure that gene names are consistent
    gene_list = ecoli_obj.gene_names
    assert (
        gene_list == ecoli_obj.X.index.tolist()
        and gene_list == ecoli_obj.M.index.tolist()
        and gene_list == ecoli_obj.gene_table.index.tolist()
    )

    # Check if gene names are used
    assert ecoli_obj.gene_names[0] == "b0002"


# def test_trn(mini_obj_opt):
#     assert mini_obj_opt._cutoff_optimized
#     mini_obj_opt.trn = trn
#     # TODO: Test that extra genes are removed
#     assert not mini_obj_opt._cutoff_optimized


def test_rename_imodulons(ecoli_obj, recwarn):

    # Check if renaming works for iModulons
    ecoli_obj.rename_imodulons({"AllR/AraC/FucR": "all3"})
    assert ecoli_obj.imodulon_names[0] == "all3"
    assert ecoli_obj.M.columns[0] == "all3"
    assert ecoli_obj.A.index[0] == "all3"
    assert "all3" in ecoli_obj.thresholds.keys()

    # Try renaming with duplicate names
    ecoli_obj.rename_imodulons({"all3": "test", "ArcA-1": "test", "ArcA-2": "test"})
    assert ecoli_obj.imodulon_names[0] == "test-1"
    assert ecoli_obj.imodulon_names[1] == "test-2"
    assert ecoli_obj.imodulon_names[2] == "test-3"

    w = recwarn.pop()
    assert issubclass(w.category, UserWarning)
    assert str(w.message).startswith("Duplicate iModulon names detected.")


# def test_view_imodulon():
#     assert False


def test_find_single_gene_imodulons(ecoli_obj):
    # check that we can call out single-gene iModulons
    assert ecoli_obj.find_single_gene_imodulons(save=True) == [
        "Pyruvate",
        "fur-KO",
        "sgrT",
        "thrA-KO",
        "ydcI-KO",
    ]
    assert ecoli_obj.imodulon_table.single_gene.sum() == 5


def test_dagostino_cutoff(mini_obj, recwarn):

    assert mini_obj.dagostino_cutoff == 2000
    assert not mini_obj._cutoff_optimized

    copy_data = IcaData(
        mini_obj.M,
        mini_obj.A,
        gene_table=mini_obj.gene_table,
        imodulon_table=mini_obj.imodulon_table,
        trn=mini_obj.trn,
        optimize_cutoff=True,
        dagostino_cutoff=2000,
    )

    assert copy_data.dagostino_cutoff == 800
    # noinspection PyProtectedMember
    assert copy_data._cutoff_optimized

    w = recwarn.pop()
    assert str(w.message).startswith("Optimizing iModulon thresholds")


def test_thresholds(mini_obj_opt):

    mini_obj_opt.thresholds = list(range(10))
    assert list(mini_obj_opt.thresholds.values()) == list(range(10))
    assert not mini_obj_opt._cutoff_optimized

    mini_obj_opt.thresholds = list(range(10))

    copy_data = IcaData(
        mini_obj_opt.M,
        mini_obj_opt.A,
        optimize_cutoff=False,
        thresholds=list(range(10, 20)),
    )

    assert list(copy_data.thresholds.values()) == list(range(10, 20))
    assert not copy_data._cutoff_optimized


# def test_change_threshold():
#     assert False


def test_recompute_thresholds(mini_obj_opt):

    assert mini_obj_opt._cutoff_optimized

    mini_obj_opt.recompute_thresholds(1000)
    assert mini_obj_opt.dagostino_cutoff == 1000
    assert not mini_obj_opt._cutoff_optimized


def test_compute_kmeans_thresholds(mini_obj):

    # Make sure thresholds are different when two threshold methods are used
    ica_data1 = IcaData(
        mini_obj.M,
        mini_obj.A,
        gene_table=mini_obj.gene_table,
        imodulon_table=mini_obj.imodulon_table,
        trn=mini_obj.trn,
        threshold_method="kmeans",
    )
    ica_data2 = IcaData(
        mini_obj.M,
        mini_obj.A,
        gene_table=mini_obj.gene_table,
        imodulon_table=mini_obj.imodulon_table,
        trn=mini_obj.trn,
        threshold_method="dagostino",
    )
    # Check that kmeans is used when no TRN is given
    ica_no_trn = IcaData(
        mini_obj.M,
        mini_obj.A,
        gene_table=mini_obj.gene_table,
        imodulon_table=mini_obj.imodulon_table,
    )

    assert not np.allclose(
        list(ica_data1.thresholds.values()), list(ica_data2.thresholds.values())
    )
    assert np.allclose(
        list(ica_no_trn.thresholds.values()), list(ica_data1.thresholds.values())
    )


def test_reoptimize_thresholds(mini_obj, capsys):

    assert mini_obj.dagostino_cutoff == 2000
    assert not mini_obj._cutoff_optimized

    mini_obj.reoptimize_thresholds(progress=False, plot=False)
    assert mini_obj.dagostino_cutoff == 800
    assert mini_obj._cutoff_optimized

    mini_obj.reoptimize_thresholds(progress=False, plot=False)
    captured = capsys.readouterr()
    assert "Cutoff already optimized" in captured.out


def test_compute_regulon_enrichment(ecoli_obj):
    # Single enrichment
    enrich = ecoli_obj.compute_regulon_enrichment("GlpR", "glpR")
    assert enrich.pvalue == 0
    assert enrich.precision == enrich.recall == enrich.f1score == 1
    assert enrich.TP == enrich.regulon_size == enrich.imodulon_size == 9
    assert enrich.n_regs == 1
    assert enrich.name == "glpR"

    # Multiple enrichment
    enrich = ecoli_obj.compute_regulon_enrichment("GlpR", "glpR+crp")
    assert enrich.pvalue == 0
    assert enrich.precision == enrich.recall == enrich.f1score == 1
    assert enrich.TP == enrich.regulon_size == enrich.imodulon_size == 9
    assert enrich.n_regs == 2
    assert enrich.name == "glpR+crp"

    # Empty enrichment
    enrich = ecoli_obj.compute_regulon_enrichment("GlpR", "nothing")
    assert enrich.pvalue == 1
    assert (
        enrich.precision
        == enrich.recall
        == enrich.f1score
        == enrich.TP
        == enrich.regulon_size
        == 0
    )

    # Test save option
    ecoli_obj.compute_regulon_enrichment("GlpR", "nothing", save=True)
    enrich = ecoli_obj.imodulon_table.loc["GlpR"]

    assert enrich.pvalue == 1
    assert (
        enrich.precision
        == enrich.recall
        == enrich.f1score
        == enrich.TP
        == enrich.regulon_size
        == 0
    )
    assert enrich.regulator == "nothing"


def test_compute_trn_enrichment(ecoli_obj):
    enrich = ecoli_obj.compute_trn_enrichment(imodulons="GlpR").iloc[0]

    ecoli_obj.compute_trn_enrichment(imodulons="GlpR", save=True)
    enrich2 = ecoli_obj.imodulon_table.loc["GlpR"]
    assert enrich.pvalue == enrich2.pvalue
    assert enrich.precision == enrich2.precision
    assert enrich.recall == enrich2.recall
    assert enrich.f1score == enrich2.f1score
    assert enrich.TP == enrich2.TP
    assert enrich.regulon_size == enrich2.regulon_size


# def test_compute_annotation_enrichment():
#     assert False


# def test_copy():
#     assert False
#
#
# def test_imodulons_with():
#     assert False
#
#
def test_name2num(ecoli_obj):
    assert ecoli_obj.name2num("thrA") == "b0002"
    assert ecoli_obj.name2num(["thrA", "thrB"]) == ["b0002", "b0003"]


def test_num2name(ecoli_obj):
    assert ecoli_obj.num2name("b0002") == "thrA"
    assert ecoli_obj.num2name(["b0002", "b0003"]) == ["thrA", "thrB"]
