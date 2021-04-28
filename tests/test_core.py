import logging

import numpy as np

from pymodulon.core import IcaData
from pymodulon.io import load_json_model, save_to_json


LOGGER = logging.getLogger(__name__)


def test_init_thresholds(ecoli_obj, mini_obj_opt, caplog, tmp_path):
    # Test all threshold initialization options
    M = mini_obj_opt.M
    A = mini_obj_opt.A

    # Reduce TRN size
    trn = ecoli_obj.trn.iloc[:10].copy()

    # Define custom thresholds
    thresh_list = list(np.arange(1, 2, 0.1))
    thresh_dict = dict(zip(M.columns, thresh_list))

    thresholds_opts = [None, thresh_dict]
    threshold_method_opts = ["kmeans", "dagostino"]
    trn_opts = [None, trn]
    optimize_cutoff_opts = [True, False]
    dagostino_cutoff_opts = [None, 600]

    for opt1 in thresholds_opts:
        for opt2 in threshold_method_opts:
            for opt3 in trn_opts:
                for opt4 in optimize_cutoff_opts:
                    for opt5 in dagostino_cutoff_opts:

                        caplog.clear()

                        obj1 = IcaData(
                            M,
                            A,
                            thresholds=opt1,
                            threshold_method=opt2,
                            trn=opt3,
                            optimize_cutoff=opt4,
                            dagostino_cutoff=opt5,
                        )

                        fname = str(tmp_path / "test_data.json")
                        save_to_json(obj1, fname)
                        obj2 = load_json_model(fname)

                        for obj in [obj1, obj2]:
                            if opt1 is not None:
                                assert obj.thresholds == thresh_dict
                                assert obj.dagostino_cutoff is None
                                assert not obj.cutoff_optimized

                                if opt4:  # optimize_cutoff == True
                                    assert caplog.messages == [
                                        "Using manually input thresholds. "
                                        "D'agostino optimization will not be "
                                        "performed."
                                    ]
                                else:
                                    assert caplog.text == ""

                            # opt1 == None
                            elif opt2 == "kmeans" or opt3 is None:
                                assert obj.dagostino_cutoff is None
                                assert not obj.cutoff_optimized
                                if opt4:  # optimize_cutoff == True
                                    assert caplog.messages == [
                                        "Using Kmeans threshold "
                                        "method. D'agostino optimization will not "
                                        "be performed."
                                    ]
                                else:
                                    assert caplog.text == ""

                            # opt2 == 'dagostino'
                            # opt3 == trn
                            elif opt4:
                                assert obj.cutoff_optimized
                                assert obj.dagostino_cutoff is not None
                                assert caplog.messages == [
                                    "Optimizing iModulon thresholds, may take "
                                    "2-3 minutes..."
                                ]
                            # opt4 == False
                            elif opt5 is None:
                                assert not obj.cutoff_optimized
                                assert obj.dagostino_cutoff == 550
                                assert caplog.messages == [
                                    "Using the default "
                                    "dagostino_cutoff of 550. This may not be "
                                    "optimal for your dataset. Use "
                                    "ica_data.reoptimize_thresholds() to find the "
                                    "optimal threshold."
                                ]
                            elif opt5 == 600:
                                assert not obj.cutoff_optimized
                                assert obj.dagostino_cutoff == 600
                            else:
                                raise ValueError("Missing test case!")


def test_m_binarized(ecoli_obj):
    binary_m = ecoli_obj.M_binarized
    assert binary_m["GlpR"].sum() == 9


def test_imodulon_names(ecoli_obj, caplog):
    # Ensure that iModulon names are consistent
    imodulon_list = ecoli_obj.imodulon_names
    assert (
        imodulon_list == ecoli_obj.M.columns.tolist()
        and imodulon_list == ecoli_obj.A.index.tolist()
        and imodulon_list == ecoli_obj.imodulon_table.index.tolist()
    )

    # Test imodulon_names setter
    with caplog.at_level(logging.WARNING):
        ecoli_obj.imodulon_names = ["test"] * 92

    assert ecoli_obj.imodulon_names[0] == "test-1"
    assert ecoli_obj.imodulon_names[71] == "test-72"
    assert "Duplicate iModulon names detected" in caplog.text


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


def test_trn(mini_obj_opt, ecoli_obj):
    assert mini_obj_opt._cutoff_optimized
    mini_obj_opt.trn = ecoli_obj.trn
    # TODO: Test that extra genes are removed
    assert not mini_obj_opt._cutoff_optimized


def test_rename_imodulons(ecoli_obj, caplog):
    # Check if renaming works for iModulons
    ecoli_obj.rename_imodulons({"AllR/AraC/FucR": "all3"})
    assert ecoli_obj.imodulon_names[0] == "all3"
    assert ecoli_obj.M.columns[0] == "all3"
    assert ecoli_obj.A.index[0] == "all3"
    assert "all3" in ecoli_obj.thresholds.keys()

    # Try renaming with duplicate names
    with caplog.at_level(logging.WARNING):
        ecoli_obj.rename_imodulons({"all3": "test", "ArcA-1": "test", "ArcA-2": "test"})

    assert ecoli_obj.imodulon_names[0] == "test-1"
    assert ecoli_obj.imodulon_names[1] == "test-2"
    assert ecoli_obj.imodulon_names[2] == "test-3"

    assert "Duplicate iModulon names detected" in caplog.text


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


def test_thresholds(mini_obj_opt, ecoli_obj):
    # Test changing thresholds
    mini_obj_opt.thresholds = dict(zip(mini_obj_opt.imodulon_names, list(range(10))))
    assert list(mini_obj_opt.thresholds.values()) == list(range(10))
    assert not mini_obj_opt._cutoff_optimized


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
