""" Test functions for plotting scripts. Does not check accuracy, only ensures that
functions do not throw errors. """

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from pymodulon.plotting import (
    cluster_activities,
    compare_activities,
    compare_expression,
    compare_gene_weights,
    compare_imodulons_vs_regulons,
    metadata_boxplot,
    plot_activities,
    plot_dima,
    plot_explained_variance,
    plot_expression,
    plot_gene_weights,
    plot_metadata,
    plot_regulon_histogram,
)


plt.rcParams["figure.max_open_warning"] = 100


@pytest.fixture(autouse=True)
def no_plots(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)


def test_plot_expression(ecoli_obj, tmp_path, monkeypatch):
    plot_expression(ecoli_obj, "b0002")
    plot_expression(
        ecoli_obj,
        "thrA",
        projects=["fur", "acid"],
        highlight="fur",
        legend_kwargs={"bbox_to_anchor": (1, 1)},
    )

    copy_obj = ecoli_obj.copy()
    copy_obj.sample_table = copy_obj.sample_table.drop(columns=["project", "condition"])
    plot_expression(copy_obj, "b0002")


def test_plot_activities(ecoli_obj, tmp_path):
    plot_activities(ecoli_obj, "GlpR")

    fig, ax = plt.subplots()
    plot_activities(ecoli_obj, "GlpR", projects="fur", highlight="fur", ax=ax)


def test_plot_metadata(ecoli_obj):
    plot_metadata(ecoli_obj, "Growth Rate (1/hr)")
    plot_metadata(
        ecoli_obj,
        "Growth Rate (1/hr)",
        projects=["fur", "acid"],
        highlight=["fur", "acid"],
    )


def test_plot_regulon_histogram(ecoli_obj, tmp_path):
    plot_regulon_histogram(ecoli_obj, "GlpR")

    fig, ax = plt.subplots()
    plot_regulon_histogram(
        ecoli_obj,
        "GlpR",
        regulator="Zur",
        bins=20,
        kind="side",
        ax=ax,
        hist_label=["test1", "test2"],
        color=["gray", "blue"],
        alpha=0.5,
        ax_font_kwargs={"fontsize": 8},
        legend_kwargs={"ncol": 2},
    )
    plot_regulon_histogram(ecoli_obj, "proVWX")

    copy_obj = ecoli_obj.copy()
    copy_obj.imodulon_table = None
    plot_regulon_histogram(copy_obj, "GlpR")
    print(copy_obj.imodulon_names)
    plot_regulon_histogram(copy_obj, "proVWX")
    copy_obj.trn = None
    plot_regulon_histogram(copy_obj, "GlpR")


def test_plot_gene_weights(ecoli_obj):
    plot_gene_weights(ecoli_obj, "GlpR")
    plot_gene_weights(ecoli_obj, "GlpR", adjust_labels=False, by="start")
    plot_gene_weights(ecoli_obj, "GlpR", adjust_labels=False, by="length")
    plot_gene_weights(
        ecoli_obj, "GlpR", adjust_labels=False, xaxis=ecoli_obj.gene_table["start"]
    )


def test_compare_gene_weights(ecoli_obj, tmp_path):
    fig, ax = plt.subplots()
    compare_gene_weights(ecoli_obj, "CysB", "Cbl+CysB", ax=ax, show_labels=True)


def test_compare_expression(ecoli_obj):
    fig, ax = plt.subplots()
    compare_expression(
        ecoli_obj,
        "b0002",
        "b0003",
        show_labels=False,
        ax=ax,
        ax_font_kwargs={"fontsize": 12},
        label_font_kwargs={"fontsize": 5},
        scatter_kwargs={"s": 2},
    )
    compare_expression(ecoli_obj, "thrB", "cysB", show_labels=False)


def test_compare_activities(ecoli_obj):
    compare_activities(
        ecoli_obj, "CysB", "Cbl+CysB", show_labels=True, adjust_labels=False
    )
    compare_activities(ecoli_obj, "CysB", "Cbl+CysB", fit_metric="spearman")
    compare_activities(ecoli_obj, "CysB", "Cbl+CysB", fit_metric="r2", colors="blue")
    compare_activities(ecoli_obj, "CysB", "Cbl+CysB", fit_metric=None, colors=None)
    compare_activities(
        ecoli_obj,
        "CysB",
        "Cbl+CysB",
        fit_metric=None,
        colors=None,
        line45=True,
        line45_margin=10,
    )


def test_plot_dima(ecoli_obj, tmp_path):
    plot_dima(ecoli_obj, "fur:wt_fe", "fur:wt_dpd", show_labels=False)

    ax, df = plot_dima(
        ecoli_obj,
        ["control__wt_glc__1", "control__wt_glc__2"],
        ["fur__wt_dpd__1", "fur__wt_dpd__2"],
        show_labels=False,
        legend=True,
        table=True,
    )

    assert isinstance(df, pd.DataFrame)


def test_plot_explained_variance(ecoli_obj, tmp_path):
    plot_explained_variance(ecoli_obj)

    fig, ax = plt.subplots()
    plot_explained_variance(ecoli_obj, pc=True, ax=ax)


def test_cluster_activities(ecoli_obj):
    cluster_activities(ecoli_obj)
    cluster_activities(
        ecoli_obj,
        correlation_method="mutual_info",
        show_thresholding=True,
        show_best_clusters=True,
        return_clustermap=True,
        dimca_sample1="fur:wt_fe",
        dimca_sample2="fur:wt_dpd",
    )


def test_metadata_boxplot(ecoli_obj, tmp_path):
    metadata_boxplot(ecoli_obj, "EvgA")
    metadata_boxplot(
        ecoli_obj, "EvgA", samples=ecoli_obj.sample_names[:20], show_points="swarm"
    )

    fig, ax = plt.subplots()
    metadata_boxplot(
        ecoli_obj,
        "EvgA",
        ignore_cols=["project", "condition"],
        strip_conc=False,
        ax=ax,
        show_points=False,
    )


def test_compare_imodulons_vs_regulons(ecoli_obj, tmp_path):
    compare_imodulons_vs_regulons(ecoli_obj)

    fig, ax = plt.subplots()
    compare_imodulons_vs_regulons(
        ecoli_obj,
        imodulons=ecoli_obj.imodulon_names[:10],
        cat_column="Category",
        size_column="n_genes",
        scale=2,
        reg_only=False,
        xlabel="Recall",
        ylabel="Precision",
        vline=None,
        hline=None,
        ax=ax,
        scatter_kwargs={"s": 10, "edgecolor": "white"},
        ax_font_kwargs={"fontsize": 10},
        legend_kwargs={"bbox_to_anchor": (1, 0)},
    )
