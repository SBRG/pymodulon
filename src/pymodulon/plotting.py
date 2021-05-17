"""
Plotting functions for iModulons
"""
import logging
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.patches import Rectangle
from scipy import sparse, stats
from scipy.optimize import OptimizeWarning, curve_fit
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from pymodulon.compare import convert_gene_index
from pymodulon.enrichment import parse_regulon_str
from pymodulon.util import _parse_sample, dima, explained_variance, mutual_info_distance


#############
# Bar Plots #
#############


def barplot(
    values,
    sample_table,
    ylabel="",
    projects=None,
    highlight=None,
    ax=None,
    legend_kwargs=None,
    savefig=False,
    savefig_kwargs=None,
):
    """
    Creates an overlaid scatter and barplot for a set of values (either gene
    expression levels or iModulon activities)

    Parameters
    ----------
    values: ~pandas.Series
        List of `values` to plot
    sample_table: ~pandas.DataFrame
        Sample table from :class:`~pymodulon.core.IcaData` object
    ylabel: str, optional
        Y-axis label
    projects: list or str, optional
        Name(s) of `projects` to show (default: show all)
    highlight: list or str, optional
        Project(s) to `highlight` (default: None)
    ax: ~matplotlib.axes.Axes, optional
        Axes object to plot on, otherwise use current Axes
    legend_kwargs: dict, optional
        Additional keyword arguments passed to :func:`matplotlib.pyplot.legend`

    Returns
    -------
    ax: ~matplotlib.axes.Axes
        :class:`~matplotlib.axes.Axes` containing the barplot
    """

    # Remove extra projects
    if isinstance(projects, str):
        projects = [projects]

    if projects is not None and len(projects) == 1:
        highlight = projects

    if projects is not None and "project" in sample_table:
        sample_table = sample_table[sample_table.project.isin(projects)]
        values = values[sample_table.index]

    if ax is None:
        figsize = (len(values) / 15 + 0.5, 2)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Get ymin and max
    ymin = values.min()
    ymax = values.max()
    yrange = ymax - ymin
    ymax = max(1, max(ymax * 1.1, ymax + yrange * 0.1))
    ymin = min(-1, min(ymin * 1.1, ymin - yrange * 0.1))
    yrange = ymax - ymin

    # Add project-specific information
    if "project" in sample_table.columns and "condition" in sample_table.columns:

        # Sort data by project/condition to ensure replicates are together
        metadata = sample_table.loc[:, ["project", "condition"]]
        metadata = metadata.sort_values(["project", "condition"])
        metadata["name"] = metadata.project + " - " + metadata.condition

        # Coerce highlight to iterable
        if highlight is None:
            highlight = []
        elif isinstance(highlight, str):
            highlight = [highlight]

        # Get X and Y values for scatter points
        metadata["y"] = values
        metadata["x"] = np.cumsum(~metadata[["name"]].duplicated())

        # Get heights for barplot
        bar_vals = metadata.groupby("x").mean()

        # Add colors and names
        bar_vals["name"] = metadata.drop_duplicates("name").name.values
        bar_vals["project"] = metadata.drop_duplicates("name").project.values

        # Plot bars for highlighted samples
        color_vals = bar_vals[bar_vals.project.isin(highlight)]
        color_cycle = [
            "tab:red",
            "tab:orange",
            "tab:green",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]
        i = 0
        for name, group in color_vals.groupby("name"):
            ax.bar(
                group.index,
                group.y,
                color=color_cycle[i],
                width=1,
                linewidth=0,
                align="edge",
                zorder=1,
                label=name,
            )
            i = (i + 1) % len(color_cycle)

        # Plot bars for non-highlighted samples
        other_vals = bar_vals[~bar_vals.project.isin(highlight)]
        ax.bar(
            other_vals.index,
            other_vals.y,
            color="tab:blue",
            width=1,
            linewidth=0,
            align="edge",
            zorder=1,
            label=None,
        )
        ax.scatter(metadata.x + 0.5, metadata.y, color="k", zorder=2, s=10)

        # Get project names and sizes
        projects = metadata.project.drop_duplicates()
        md_cond = metadata.drop_duplicates(["name"])
        project_sizes = [len(md_cond[md_cond.project == proj]) for proj in projects]
        nbars = len(md_cond)

        # Draw lines to discriminate between projects
        proj_lines = np.cumsum([1] + project_sizes)
        ax.vlines(proj_lines, ymin, ymax, colors="lightgray", linewidth=1)

        # Add project names
        texts = []
        start = 2
        for proj, size in zip(projects, project_sizes):
            x = start + size / 2
            texts.append(
                ax.text(
                    x, ymin - yrange * 0.02, proj, ha="right", va="top", rotation=45
                )
            )
            start += size

        # Add legend
        if not color_vals.empty:
            kwargs = {
                "bbox_to_anchor": (1, 1),
                "ncol": len(color_vals.name.unique()) // 6 + 1,
            }

            if legend_kwargs is not None:
                kwargs.update(legend_kwargs)

            ax.legend(**kwargs)

    else:
        logging.warning('Missing "project" and "condition" columns in sample ' "table.")
        ax.bar(range(len(values)), values, width=1, align="edge")
        nbars = len(values)

    # Set axis limits
    xmin = -0.5
    xmax = nbars + 2.5
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Axis labels
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks([])

    # X-axis
    ax.hlines(0, xmin, xmax, color="k")

    # Save figure
    if savefig:
        _save_figures(fig, savefig_kwargs)

    return ax


def plot_expression(
    ica_data,
    gene,
    projects=None,
    highlight=None,
    ax=None,
    legend_kwargs=None,
    savefig=False,
    savefig_kwargs=None,
):
    """
    Creates a barplot showing an gene's expression across the compendium

    Parameters
    ----------
    ica_data: ~pymodulon.core.IcaData
        :class:`~pymodulon.core.IcaData` object
    gene: str
        `Gene` locus tag or name
    projects: str or list, optional
        Name(s) of `projects` to show (default: show all)
    highlight: str or list, optional
        Name(s) of projects to `highlight` (default: None)
    ax: ~matplotlib.axes.Axes, optional
        Axes object to plot on, otherwise use current Axes
    legend_kwargs: dict, optional
        Additional keyword arguments passed to :func:`matplotlib.pyplot.legend`

    Returns
    -------
    ax: ~matplotlib.axes.Axes
        :class:`~matplotlib.axes.Axes` containing the barplot
    """

    # Check that gene exists
    if gene in ica_data.X.index:
        values = ica_data.X.loc[gene]
        label = "{} Expression".format(gene)
    else:
        locus = ica_data.name2num(gene)
        values = ica_data.X.loc[locus]
        label = "${}$ Expression".format(gene)

    return barplot(
        values,
        ica_data.sample_table,
        label,
        projects,
        highlight,
        ax,
        legend_kwargs,
        savefig=savefig,
        savefig_kwargs=savefig_kwargs,
    )


def plot_activities(
    ica_data,
    imodulon,
    projects=None,
    highlight=None,
    ax=None,
    legend_kwargs=None,
    savefig=False,
    savefig_kwargs=None,
):
    """
    Creates a barplot showing an iModulon's activity across the compendium

    Parameters
    ----------
    ica_data: ~pymodulon.core.IcaData
        :class:`~pymodulon.core.IcaData` object
    imodulon: int or str
        `iModulon` name
    projects: list or str, optional
        Name(s) of `projects` to show (default: show all)
    highlight: str or list, optional
        Name(s) of projects to `highlight` (default: None)
    ax: ~matplotlib.axes.Axes, optional
        Axes object to plot on, otherwise use current Axes
    legend_kwargs: dict, optional
        Additional keyword arguments passed to :func:`matplotlib.pyplot.legend`

    Returns
    -------
    ax: ~matplotlib.axes.Axes
        :class:`~matplotlib.axes.Axes` containing the barplot
    """

    # Check that iModulon exists
    if imodulon in ica_data.A.index:
        values = ica_data.A.loc[imodulon]
    else:
        raise ValueError(f"iModulon does not exist: {imodulon}")

    label = "{} iModulon\nActivity".format(imodulon)

    return barplot(
        values,
        ica_data.sample_table,
        label,
        projects,
        highlight,
        ax,
        legend_kwargs,
        savefig=savefig,
        savefig_kwargs=savefig_kwargs,
    )


def plot_metadata(
    ica_data,
    column,
    projects=None,
    highlight=None,
    ax=None,
    legend_kwargs=None,
    savefig=False,
    savefig_kwargs=None,
):
    """
    Creates a barplot for values in the sample table

    Parameters
    ----------
    ica_data: ~pymodulon.core.IcaData
        :class:`~pymodulon.core.IcaData` object
    column : str
        `Column` name to plot
    projects: list or str, optional
        Name(s) of `projects` to show (default: show all)
    highlight: str or list, optional
        Name(s) of projects to `highlight` (default: None)
    ax: ~matplotlib.axes.Axes, optional
        Axes object to plot on, otherwise use current Axes
    legend_kwargs: dict, optional
        Additional keyword arguments passed to :func:`matplotlib.pyplot.legend`

    Returns
    -------
    ax: ~matplotlib.axes.Axes
        :class:`~matplotlib.axes.Axes` containing the barplot
    """

    # Check that column exists
    if column in ica_data.sample_table.columns:
        # Make sure the column is filled with numbers
        if not pd.api.types.is_numeric_dtype(ica_data.sample_table[column]):
            raise ValueError("Metadata column {} is not numeric".format(column))

        # Remove null values
        table = ica_data.sample_table[ica_data.sample_table[column].notnull()]
        values = ica_data.sample_table[column]

    else:
        raise ValueError("Column not in sample table: {}".format(column))

    return barplot(
        values,
        table,
        column,
        projects,
        highlight,
        ax,
        legend_kwargs,
        savefig=savefig,
        savefig_kwargs=savefig_kwargs,
    )


def plot_regulon_histogram(
    ica_data,
    imodulon,
    regulator=None,
    bins=None,
    kind="overlap",
    ax=None,
    hist_label=("Not regulated", "Regulon Genes"),
    color=("#aaaaaa", "salmon"),
    alpha=0.7,
    ax_font_kwargs=None,
    legend_kwargs=None,
    savefig=False,
    savefig_kwargs=None,
):
    """
    Plots a histogram of regulon vs non-regulon genes by iModulon weighting.

    Parameters
    ----------
    ica_data: ~pymodulon.core.IcaData
        :class:`~pymodulon.core.IcaData` object
    imodulon : int or str
        `iModulon` name
    regulator: str, optional
        Name of `regulator` to compare enrichment against. If no regulator is given,
        choose top enrichment from :attr:`~pymodulon.core.IcaData.imodulon_table`
    bins: int, str or list, optional
        The `bins` to use when generating the histogram. Passed on to
        :func:`matplotlib.pyplot.hist`
    kind: 'overlap' or 'side', optional
        Whether to plot an overlapping or side-by-side comparison histogram (
        default: 'overlap')
    ax: ~matplotlib.axes.Axes, optional
        Axes object to plot on, otherwise use current Axes
    hist_label: tuple, optional
        The label to use when plotting the regulon and non-regulon genes.
        Takes into a tuple of 2 values (first for non-regulon genes,
        second for regulon genes). Passed on to :func:`matplotlib.pyplot.hist`
    color: list or str, optional
        The colors to use for regulon and non-regulon genes. Takes a
        Sequence of 2 values (first for non-regulon genes, second for
        regulon genes). Passed on to :func:`matplotlib.pyplot.hist`
    alpha: float, optional
        Sets the opacity of the histogram (0 = transparent, 1 = opaque).
        Passed on to :func:`matplotlib.pyplot.hist`
    ax_font_kwargs: dict, optional
        Additional keyword arguments for axes labels
    legend_kwargs: dict, optional
        Additional keyword arguments passed to :func:`matplotlib.pyplot.legend`

    Returns
    -------
    ax: ~matplotlib.axes.Axes
        :class:`~matplotlib.axes.Axes` containing the histogram
    """

    # Check that iModulon exists
    if imodulon not in ica_data.M.columns:
        raise ValueError(f"iModulon does not exist: {imodulon}")

    # If ax is None, create ax on which to generate histogram
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # If bins is None, generate optimal number of bins
    if bins is None:
        bin_arr = _mod_freedman_diaconis(ica_data, imodulon)
    else:
        bin_arr = bins

    # If no TRN in IcaData, regulon genes cannot be determined and plotted
    if ica_data.trn.empty:
        reg = None

    # If regulator is given, use it to find regulon genes
    elif regulator is not None:
        reg = regulator

    # If regulator is not given, use imodulon_table to find regulator
    elif not ica_data.imodulon_table.empty:
        reg = ica_data.imodulon_table.loc[imodulon, "regulator"]
        if pd.isna(reg):
            reg = None

    # If no imodulon_table in IcaData, compute trn enrichment to find the
    # regulator. Note that trn enrichment is computed using `max_regs` = 1
    else:
        df_enriched = ica_data.compute_trn_enrichment(imodulons=imodulon)
        df_top_enrich = df_enriched.sort_values(
            ["imodulon", "qvalue", "n_regs"]
        ).drop_duplicates("imodulon")
        reg = df_top_enrich.set_index("imodulon").loc[imodulon, "regulator"]
        if not isinstance(reg, str):
            if pd.isna(reg):
                reg = None

    # Use regulator value to find regulon genes
    if reg is not None:
        reg_genes = parse_regulon_str(reg, ica_data.trn)
    else:
        reg_genes = set()

    # Handle custom kwargs
    if ax_font_kwargs is None:
        ax_font_kwargs = {}

    if legend_kwargs is None:
        legend_kwargs = dict({"loc": "upper right"})

    # Histogram
    non_reg_genes = set(ica_data.gene_names) - reg_genes
    reg_arr = ica_data.M[imodulon].loc[reg_genes]
    non_reg_arr = ica_data.M[imodulon].loc[non_reg_genes]

    if kind == "overlap":
        ax.hist(
            non_reg_arr, bins=bin_arr, alpha=alpha, color=color[0], label=hist_label[0]
        )
        ax.hist(reg_arr, bins=bin_arr, alpha=alpha, color=color[1], label=hist_label[1])

    elif kind == "side":
        arr = np.array([non_reg_arr, reg_arr], dtype="object")
        # noinspection PyTypeChecker
        ax.hist(arr, bins=bin_arr, alpha=alpha, color=color, label=hist_label)

    else:
        raise ValueError(
            f"{kind} is not a valid option. `kind` must be "
            'either "overlap" or "side"'
        )

    # Set y-axis to log-scale
    ax.set_yscale("log")

    # Add thresholds to scatterplot (dashed lines)
    ymin, ymax = ax.get_ylim()
    thresh = abs(ica_data.thresholds[imodulon])
    if thresh != 0:
        ax.vlines(
            [-thresh, thresh],
            ymin=ymin,
            ymax=ymax,
            colors="k",
            linestyles="dashed",
            linewidth=1,
        )

    ax.set_ylim(ymin, ymax)

    # Set x and y labels
    ax.set_xlabel(f"{imodulon} Gene Weight", **ax_font_kwargs)
    ax.set_ylabel("Number of Genes", **ax_font_kwargs)

    # Add legend
    ax.legend(**legend_kwargs)

    # Save figure
    if savefig:
        _save_figures(fig, savefig_kwargs)

    return ax


################
# Scatterplots #
################


def scatterplot(
    x,
    y,
    groups=None,
    colors=None,
    show_labels="auto",
    adjust_labels=True,
    line45=False,
    line45_margin=0,
    fit_line=False,
    fit_metric="pearson",
    xlabel="",
    ylabel="",
    ax=None,
    legend=True,
    ax_font_kwargs=None,
    scatter_kwargs=None,
    label_font_kwargs=None,
    legend_kwargs=None,
    savefig=False,
    savefig_kwargs=None,
):
    """
    Generates a scatter-plot of the data given, with options for coloring by
    group, adding labels, adding lines, and generating correlation or
    determination coefficients.

    Parameters
    ----------
    x: ~pandas.Series
        `X`-axis data
    y: ~pandas.Series
        `Y`-axis data
    groups: dict, optional
        A dictionary mapping samples to group names
    colors: str, list or dict, optional
        Color of points, list of colors for different groups, or dictionary
        mapping groups to colors
    show_labels: bool or 'auto'
        Show labels for data points
    adjust_labels: bool
        Auto-adjust labels for data points
    line45: bool
        Show 45-degree line of equal values
    line45_margin: float
        Show 45-degreen lines offset by a margin
    fit_line: bool
        Draw a line of best fit on the scatterplot
    fit_metric: 'pearson', 'spearman', 'r2', or None
        Metric to report in legend for line of best fit
    xlabel: str
        X-axis label
    ylabel: str
        Y-axis label
    ax: ~matplotlib.axes.Axes, optional
        Axes object to plot on, otherwise use current Axes
    legend: bool
        Show legend
    ax_font_kwargs: dict, optional
        Additional keyword arguments for axis labels
    scatter_kwargs: dict, optional
        Additional keyword arguments passed to :func:`matplotlib.pyplot.scatter`
    label_font_kwargs: dict, optional
        Additional keyword arguments for labels passed to
        :func:`matplotlib.pyplot.text`
    legend_kwargs: dict, optional
        Additional keyword arguments passed to :func:`matplotlib.pyplot.legend`

    Returns
    -------
    ax: ~matplotlib.axes.Axes
        :class:`~matplotlib.axes.Axes` containing the scatterplot
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if show_labels == "auto":
        show_labels = len(x) <= 20

    if not (
        isinstance(x, pd.Series)
        and isinstance(y, pd.Series)
        and (
            x.index.astype(str).sort_values() == y.index.astype(str).sort_values()
        ).all()
    ):
        raise TypeError("X and Y must be pandas series with the same index")

    # Set up data object
    data = pd.DataFrame({"x": x, "y": y})

    # Add group information
    data["group"] = ""
    if groups is not None:
        for k, val in groups.items():
            data.loc[k, "group"] = val

    # Handle custom kwargs
    if ax_font_kwargs is None:
        ax_font_kwargs = {}

    if label_font_kwargs is None:
        label_font_kwargs = {}

    if legend_kwargs is None:
        legend_kwargs = {}

    if scatter_kwargs is None:
        scatter_kwargs = {}

    # Get x and y limits
    margin = 0.1
    xrange = x.max() - x.min()
    yrange = y.max() - y.min()
    xmin = x.min() - xrange * margin
    xmax = x.max() + xrange * margin
    ymin = y.min() - yrange * margin
    ymax = y.max() + yrange * margin
    allmin = max(xmin, ymin)
    allmax = min(xmax, ymax)

    # Add 45 degree line
    if line45:
        # Plot diagonal lines
        ax.plot(
            [allmin, allmax],
            [allmin, allmax],
            color="k",
            linestyle="dashed",
            linewidth=0.5,
            zorder=0,
        )

        if line45_margin > 0:
            diff = pd.DataFrame(abs(data.x - data.y), index=data.index)
            hidden = diff.loc[diff[0] < line45_margin]
            data.loc[hidden.index, "group"] = "hidden"
            ax.plot(
                [max(xmin, ymin + line45_margin), min(xmax, ymax + line45_margin)],
                [max(ymin, xmin - line45_margin), min(ymax, xmax - line45_margin)],
                color="gray",
                linestyle="dashed",
                linewidth=0.5,
                zorder=0,
            )
            ax.plot(
                [max(xmin, ymin - line45_margin), min(xmax, ymax - line45_margin)],
                [max(ymin, xmin + line45_margin), min(ymax, xmax + line45_margin)],
                color="gray",
                linestyle="dashed",
                linewidth=0.5,
                zorder=0,
            )

    # Add colors to the data
    # If colors is already a dict, just update the hidden color
    try:
        if "hidden" not in colors.keys():
            colors.update({"hidden": "gray"})
        if "" not in colors.keys():
            colors.update({"": "tab:blue"})
    except AttributeError:

        groups = [item for item in data["group"].unique() if item != "hidden"]
        if colors is None:
            colorlist = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        elif isinstance(colors, str):
            colorlist = [colors] * len(groups)
        else:
            colorlist = list(colors)

        # Deal with short colorlists
        if len(colorlist) < len(groups):
            colorlist = colorlist * (len(groups) // len(colorlist) + 1)

        colors = dict(zip(groups, colorlist))
        colors.update({"hidden": "gray"})

    for name, group in data.groupby("group"):
        kwargs = scatter_kwargs.copy()
        kwargs["color"] = colors[name]
        # Override defaults for hidden points
        if name == "hidden":
            kwargs.update({"alpha": 0.7, "linewidth": 0, "label": None})
        elif name == "":
            kwargs.update({"label": None})
        else:
            kwargs.update({"label": name})

        ax.scatter(group.x, group.y, **kwargs, zorder=1)

    # Add regression
    if fit_line:
        _fit_line(x, y, ax, fit_metric)

    # Add lines at 0
    if xmin < 0 < xmax:
        ax.hlines(0, xmin, xmax, linewidth=0.5, color="gray", zorder=2)
    if ymin < 0 < ymax:
        ax.vlines(0, ymin, ymax, linewidth=0.5, color="gray", zorder=2)

    # Add labels
    if show_labels:
        texts = []
        for idx in x.index:
            texts.append(ax.text(x[idx], y[idx], idx, **label_font_kwargs))
        if adjust_labels:
            adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(arrowstyle="-", color="k", lw=0.5),
                only_move={"objects": "y"},
                expand_objects=(1.2, 1.4),
                expand_points=(1.3, 1.3),
            )

    if np.allclose(xmin, xmax):
        ax.set_xlim(xmin - 1, xmax + 1)
    else:
        ax.set_xlim(xmin, xmax)

    if np.allclose(ymin, ymax):
        ax.set_ylim(ymin - 1, ymax + 1)
    else:
        ax.set_ylim(ymin, ymax)

    ax.set_xlabel(xlabel, **ax_font_kwargs)
    ax.set_ylabel(ylabel, **ax_font_kwargs)

    if legend or fit_line:
        ax.legend(**legend_kwargs)

    # Save figure
    if savefig:
        _save_figures(fig, savefig_kwargs)

    return ax


def plot_gene_weights(ica_data, imodulon, by="start", xaxis=None, xname="", **kwargs):
    """
    Plot gene weights on a scatter plot.

    Parameters
    ----------
    ica_data: ~pymodulon.core.IcaData
        :class:`~pymodulon.core.IcaData` object
    imodulon : int or str
        `iModulon` name
    by: 'log-tpm-norm', 'length', or 'start'
        Property to plot on x-axis. Superceded by `xaxis`
    xaxis: list, dict or ~pandas.Series, optional
        Values on custom x-axis
    xname: str, optional
        Name of x-axis if using custom x-axis
    **kwargs: dict, optional
        Additional keyword arguments passed to :func:`pymodulon.plotting.scatterplot`
    Returns
    -------
    ax: ~matplotlib.axes.Axes
        :class:`~matplotlib.axes.Axes` containing the scatterplot
    """
    # Check that iModulon exists
    if imodulon in ica_data.M.columns:
        y = ica_data.M[imodulon]
        ylabel = f"{imodulon} Gene Weight"
    else:
        raise ValueError(f"iModulon does not exist: {imodulon}")

    # Get genes in the iModulon
    bin_M = ica_data.M_binarized
    component_genes = set(bin_M[imodulon].loc[bin_M[imodulon] == 1].index)
    other_genes = set(bin_M[imodulon].loc[bin_M[imodulon] == 0].index)

    # If experimental `xaxis` parameter is used, use custom values for x-axis
    if xaxis is not None:
        x = _set_xaxis(xaxis=xaxis, y=y)
        xlabel = xname

    else:
        #  Ensure 'by' has a valid input and assign x, xlabel accordingly
        if by == "log-tpm":
            x = ica_data.log_tpm.mean(axis=1)
            xlabel = "Mean Expression"
        elif by == "log-tpm-norm":
            x = ica_data.X.mean(axis=1)
            xlabel = "Mean Centered Expression"
        elif by == "length":
            x = np.log10(ica_data.gene_table.length)
            xlabel = "Gene Length (log10-scale)"
        elif by == "start":
            x = ica_data.gene_table.start
            xlabel = "Gene Start"
        else:
            raise ValueError(
                '"by" must be "log-tpm", "log-tpm-norm", "length", ' 'or "start"'
            )

    # Override specific kwargs (their implementation is different
    # in this function)
    show_labels_pgw = kwargs.pop("show_labels", "auto")
    adjust_labels_pgw = kwargs.pop("adjust_labels", True)
    legend_kwargs_pgw = kwargs.pop("legend_kwargs", {})
    label_font_kwargs_pgw = kwargs.pop("label_font_kwargs", {})

    kwargs["show_labels"] = kwargs["adjust_labels"] = False

    # Remove xlabel and ylabel kwargs if provided
    kwargs.pop("xlabel", None)
    kwargs.pop("ylabel", None)

    # Default legend should be on the side of the plot
    if (
        "bbox_to_anchor" not in legend_kwargs_pgw.keys()
        and "loc" not in legend_kwargs_pgw.keys()
    ):
        legend_kwargs_pgw.update({"bbox_to_anchor": (1, 1), "loc": 2})
        kwargs["legend_kwargs"] = legend_kwargs_pgw

    # Update colors for COG groups
    if "COG" in ica_data.gene_table.columns and "groups" not in kwargs:
        mod_cogs = ica_data.gene_table.loc[component_genes].COG
        hidden_cogs = pd.Series("hidden", index=other_genes)
        all_cogs = pd.concat([mod_cogs, hidden_cogs])
        # colors = {cog:ica_data.cog_colors[cog] for cog in sorted(mod_cogs.unique())}
        kwargs.update({"groups": all_cogs, "colors": ica_data.cog_colors})

    # Scatter Plot
    ax = scatterplot(x, y, xlabel=xlabel, ylabel=ylabel, **kwargs)

    # Add thresholds to scatter-plot (dashed lines)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    thresh = ica_data.thresholds[imodulon]
    if thresh != 0:
        ax.hlines(
            [thresh, -thresh],
            xmin=xmin,
            xmax=xmax,
            colors="k",
            linestyles="dashed",
            linewidth=1,
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    texts = []
    expand_kwargs = {"expand_objects": (1.2, 1.4), "expand_points": (1.3, 1.3)}

    # Add labels: Put gene name if components contain under 20 genes
    if show_labels_pgw is True or (
        show_labels_pgw is not False and len(component_genes) <= 20
    ):
        for gene in component_genes:

            # Add labels
            text_kwargs = label_font_kwargs_pgw.copy()

            if "fontstyle" not in text_kwargs:
                text_kwargs.update({"fontstyle": "normal"})

            # Italicize gene if there is a defined name (not locus tag)
            try:
                gene_name = ica_data.gene_table.loc[gene, "gene_name"]

                if gene_name != gene:
                    text_kwargs.update({"fontstyle": "italic"})

            except KeyError:
                gene_name = gene

            # Set default fontsize
            if "fontsize" not in text_kwargs:
                text_kwargs.update({"fontsize": 12})

            texts.append(
                ax.text(
                    x[gene],
                    ica_data.M.loc[gene, imodulon],
                    gene_name,
                    **text_kwargs,
                )
            )

        expand_kwargs["expand_text"] = (1.4, 1.4)

    # Add labels: Repel texts from other text and points
    rect = ax.add_patch(
        Rectangle(
            xy=(xmin, -abs(thresh)),
            width=xmax - xmin,
            height=2 * abs(thresh),
            fill=False,
            linewidth=0,
        )
    )

    if adjust_labels_pgw:
        adjust_text(
            texts=texts,
            add_objects=[rect],
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="k", lw=0.5),
            only_move={"objects": "y"},
            **expand_kwargs,
        )

    return ax


def compare_gene_weights(
    ica_data,
    imodulon1,
    imodulon2,
    ica_data2=None,
    ortho_file=None,
    use_org1_names=True,
    **kwargs,
):
    """
    Create a scatterplot comparing the gene weights between two iModulons

    Parameters
    ----------
    ica_data: ~pymodulon.core.IcaData
        :class:`~pymodulon.core.IcaData` object
    imodulon1 : int or str
        Name of `iModulon` on X-axis
    imodulon2 : int or str
        Name of `iModulon` on X-axis
    ica_data2: ~pymodulon.core.IcaData, optional
        :class:`~pymodulon.core.IcaData` object for second iModulon (if comparing
        iModulons across objects)
    ortho_file: str, optional
        Path to orthology file between organisms
    use_org1_names: bool
        If true, use gene names from first organism. If false, use gene names
        from second organism (default: True)
    **kwargs: dict, optional
        Additional keyword arguments passed to :func:`pymodulon.plotting.scatterplot`

    Returns
    -------
    ax: ~matplotlib.axes.Axes
        :class:`~matplotlib.axes.Axes` containing the scatterplot
    """
    if ica_data2 is None:
        ica_data2 = ica_data.copy()

    M1, M2 = convert_gene_index(ica_data.M, ica_data2.M, ortho_file)
    bin_M1, bin_M2 = convert_gene_index(
        ica_data.M_binarized, ica_data2.M_binarized, ortho_file
    )

    # Convert gene table
    gene_table1, gene_table2 = convert_gene_index(
        ica_data.gene_table, ica_data2.gene_table, ortho_file, keep_locus=True
    )

    if use_org1_names:
        gene_table = gene_table1
        gene_table["locus_tag"] = gene_table.index
    else:
        gene_table = gene_table2

    x = M1[imodulon1]
    y = M2[imodulon2]

    xlabel = f"{imodulon1} Gene Weight"
    ylabel = f"{imodulon2} Gene Weight"

    # Override specific kwargs (their implementation is different
    # in this function)
    show_labels_cgw = kwargs.pop("show_labels", "auto")
    adjust_labels_cgw = kwargs.pop("adjust_labels", True)
    legend_cgw = kwargs.pop("legend", False)
    legend_kwargs_cgw = kwargs.pop("legend_kwargs", {})
    label_font_kwargs_cgw = kwargs.pop("label_font_kwargs", {})

    kwargs["show_labels"] = kwargs["adjust_labels"] = kwargs["legend"] = False
    kwargs["legend_kwargs"] = None

    # Remove xlabel and ylabel kwargs if provided
    kwargs.pop("xlabel", None)
    kwargs.pop("ylabel", None)

    # Scatter Plot
    ax = scatterplot(x, y, xlabel=xlabel, ylabel=ylabel, **kwargs)

    # Add thresholds to scatterplot (dashed lines)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    thresh1 = ica_data.thresholds[imodulon1]
    thresh2 = ica_data2.thresholds[imodulon2]

    if thresh1 != 0:
        ax.vlines(
            [thresh1, -thresh1],
            ymin=ymin,
            ymax=ymax,
            colors="k",
            linestyles="dashed",
            linewidth=1,
        )

    if thresh2 != 0:
        ax.hlines(
            [thresh2, -thresh2],
            xmin=xmin,
            xmax=xmax,
            colors="k",
            linestyles="dashed",
            linewidth=1,
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Add labels on data-points
    component_genes_x = bin_M1[bin_M1[imodulon1] == 1].index
    component_genes_y = bin_M2[bin_M2[imodulon2] == 1].index
    component_genes = component_genes_x.intersection(component_genes_y)
    texts = []
    expand_kwargs = {"expand_objects": (1.2, 1.4), "expand_points": (1.3, 1.3)}

    # Add labels: Put gene name if components contain under 20 genes
    auto = None
    if show_labels_cgw == "auto":
        auto = (
            bin_M1[imodulon1].astype(bool) & bin_M2[imodulon2].astype(bool)
        ).sum() <= 20

    if show_labels_cgw or auto:
        for gene in component_genes:
            ax.scatter(M1.loc[gene, imodulon1], M2.loc[gene, imodulon2], color="r")

            # Add labels
            text_kwargs = label_font_kwargs_cgw.copy()

            if "fontstyle" not in text_kwargs:
                text_kwargs.update({"fontstyle": "normal"})

            # Italicize gene if there is a defined name (not locus tag)
            try:
                gene_name = gene_table.loc[gene, "gene_name"]
                if gene_name != gene_table.loc[gene, "locus_tag"]:
                    text_kwargs.update({"fontstyle": "italic"})

            except KeyError:
                gene_name = gene

            # Set default fontsize
            if "fontsize" not in text_kwargs:
                text_kwargs.update({"fontsize": 12})

            texts.append(
                ax.text(
                    M1.loc[gene, imodulon1],
                    M2.loc[gene, imodulon2],
                    gene_name,
                    **text_kwargs,
                )
            )

        expand_kwargs["expand_text"] = (1.4, 1.4)

    # Add labels: Repel texts from other text and points
    rectx = ax.add_patch(
        Rectangle(
            xy=(xmin, -abs(thresh2)),
            width=xmax - xmin,
            height=2 * abs(thresh2),
            fill=False,
            linewidth=0,
        )
    )

    recty = ax.add_patch(
        Rectangle(
            xy=(-abs(thresh1), ymin),
            width=2 * abs(thresh1),
            height=ymax - ymin,
            fill=False,
            linewidth=0,
        )
    )

    if adjust_labels_cgw:
        adjust_text(
            texts=texts,
            add_objects=[rectx, recty],
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="k", lw=0.5),
            only_move={"objects": "y"},
            **expand_kwargs,
        )

    # Add legend
    if legend_cgw:
        ax.legend(**legend_kwargs_cgw)

    return ax


def compare_expression(ica_data, gene1, gene2, **kwargs):
    """
    Create a scatterplot comparing the compendium-wide expression profiles of two genes

    Parameters
    ----------
    ica_data: ~pymodulon.core.IcaData
        :class:`~pymodulon.core.IcaData` object
    gene1: str
        Gene to plot on the x-axis
    gene2: str
        Gene to plot on the y-axis
    **kwargs: dict, optional
        Additional keyword arguments passed to :func:`pymodulon.plotting.scatterplot`

    Returns
    -------
    ax: ~matplotlib.axes.Axes
        :class:`~matplotlib.axes.Axes` containing the scatterplot
    """

    # Check that gene1 exists
    if gene1 in ica_data.X.index:
        x = ica_data.X.loc[gene1]
        xlabel = f"{gene1} Expression"
    else:
        locus = ica_data.name2num(gene1)
        x = ica_data.X.loc[locus]
        xlabel = f"${gene1}$ Expression"

    # Check that gene2 exists
    if gene2 in ica_data.X.index:
        y = ica_data.X.loc[gene2]
        ylabel = f"{gene2} Expression"
    else:
        locus = ica_data.name2num(gene2)
        y = ica_data.X.loc[locus]
        ylabel = f"${gene2}$ Expression"

    # Remove xlabel, ylabel, and fit_line kwargs if provided
    kwargs.pop("xlabel", None)
    kwargs.pop("ylabel", None)
    kwargs.pop("fit_line", None)

    # Scatter Plot
    ax = scatterplot(x, y, xlabel=xlabel, ylabel=ylabel, fit_line=True, **kwargs)

    return ax


def compare_activities(ica_data, imodulon1, imodulon2, **kwargs):
    """
    Create a scatterplot comparing the compendium-wide activities of two iModulons

    Parameters
    ----------
    ica_data: ~pymodulon.core.IcaData
        :class:`~pymodulon.core.IcaData` object
    imodulon1: int or str
        Name of the iModulon to plot on the x-axis
    imodulon2: int or str
        Name of the iModulon to plot on the y-axis
    **kwargs: dict, optional
        Additional keyword arguments passed to :func:`pymodulon.plotting.scatterplot`

    Returns
    -------
    ax: ~matplotlib.axes.Axes
        :class:`~matplotlib.axes.Axes` containing the scatterplot
    """

    x = ica_data.A.loc[imodulon1]
    y = ica_data.A.loc[imodulon2]

    xlabel = f"{imodulon1} iModulon Activity"
    ylabel = f"{imodulon2} iModulon Activity"

    # Remove xlabel, ylabel, and fit_line kwargs if provided
    kwargs.pop("xlabel", None)
    kwargs.pop("ylabel", None)
    kwargs.pop("fit_line", None)

    # Scatter Plot
    ax = scatterplot(x, y, xlabel=xlabel, ylabel=ylabel, fit_line=True, **kwargs)

    return ax


def plot_dima(
    ica_data,
    sample1,
    sample2,
    threshold=5,
    fdr=0.1,
    label=True,
    adjust=True,
    table=False,
    **kwargs,
):
    """
    Plots a DiMA plot between two projects or two sets of samples

    Parameters
    ----------
    ica_data: ~pymodulon.core.IcaData
        :class:`~pymodulon.core.IcaData` object
    sample1: list or str
        List of sample IDs or name of "project:condition" for x-axis
    sample2: list or str
        List of sample IDs or name of "project:condition" for y-axis
    threshold: float
        Minimum activity difference to determine DiMAs (default: 5)
    fdr: float
        False detection rate (default: 0.1)
    label: bool
        Label differentially activated iModulons (default: True)
    adjust: bool
        Automatically adjust labels (default: True)
    table: bool
        Return differential iModulon activity table (default: False)
    **kwargs: dict, optional
        Additional keyword arguments passed to :func:`pymodulon.plotting.scatterplot`

    Returns
    -------
    ax: ~matplotlib.axes.Axes
        :class:`~matplotlib.axes.Axes` containing the scatterplot

    df_diff: ~pandas.DataFrame, optional
        Table reporting differentially activated iModulons

    """

    # use secret option to enable passing of clustered activity matrix
    try:
        A_to_use = kwargs.pop("alternate_A")
    except KeyError:
        A_to_use = ica_data.A

    # Override specific kwargs (their implementation is different
    # in this function)
    legend_cgw = kwargs.pop("legend", False)
    legend_kwargs_cgw = kwargs.pop("legend_kwargs", {})

    kwargs["legend"] = False
    kwargs["legend_kwargs"] = None

    # Get x and y coordinates
    sample1_list = _parse_sample(ica_data, sample1)
    sample2_list = _parse_sample(ica_data, sample2)
    if isinstance(sample1, str):
        xlabel = sample1
    else:
        xlabel = "\n".join(sample1)
    if isinstance(sample2, str):
        ylabel = sample2
    else:
        ylabel = "\n".join(sample2)

    a1 = A_to_use[sample1_list].mean(axis=1)
    a2 = A_to_use[sample2_list].mean(axis=1)

    df_diff = dima(
        ica_data,
        sample1_list,
        sample2_list,
        threshold=threshold,
        fdr=fdr,
        alternate_A=A_to_use,
    )

    groups = {}
    for i in A_to_use.index:
        if i not in df_diff.index:
            groups.update({i: "hidden"})
        else:
            groups.update({i: ""})

    ax = scatterplot(
        a1,
        a2,
        groups=groups,
        line45=True,
        xlabel=xlabel,
        ylabel=ylabel,
        **kwargs,
    )

    if label:
        df_diff = pd.concat([df_diff, a1, a2], join="inner", axis=1)
        texts = []
        for k in df_diff.index:
            texts.append(ax.text(df_diff.loc[k, 0], df_diff.loc[k, 1], k, fontsize=10))
        if adjust:
            expand_args = {
                "expand_objects": (1.2, 1.4),
                "expand_points": (1.3, 1.3),
                "expand_text": (1.4, 1.4),
            }
            adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(arrowstyle="-", color="k", lw=0.5),
                only_move={"objects": "y"},
                **expand_args,
            )

    # Add legend if requested
    if legend_cgw:
        ax.legend(**legend_kwargs_cgw)

    if table:
        return ax, df_diff

    else:
        return ax


###############
# Other plots #
###############


def plot_explained_variance(ica_data, pc=True, ax=None):
    """
    Plots the cumulative explained variance for independent components and,
    optionally, principal components

    Parameters
    ----------
    ica_data: ~pymodulon.core.IcaData
        :class:`~pymodulon.core.IcaData` object
    pc: bool
        If True, plot cumulative explained variance of independent components
    ax: ~matplotlib.axes.Axes, optional
        Axes object to plot on, otherwise use current Axes

    Returns
    -------
    ax: ~matplotlib.axes.Axes
        :class:`~matplotlib.axes.Axes` containing the line plot
    """

    # Get IC explained variance
    ic_var = []
    for imodulon in ica_data.imodulon_names:
        ic_var.append(explained_variance(ica_data, imodulons=imodulon))
    ic_var = np.insert(np.cumsum(sorted(ic_var, reverse=True)), 0, 0)

    if not ax:
        fig, ax = plt.subplots()

    ax.plot(range(len(ic_var)), ic_var, label="Independent Components")

    if pc:
        # Get PC explained variance
        pca = PCA().fit(ica_data.X.T)
        pc_var = np.insert(np.cumsum(pca.explained_variance_ratio_), 0, 0)

        # Only keep number of PCs as ICs
        pc_var = pc_var[: len(ic_var)]

        # Plot PCs
        ax.plot(range(len(ic_var)), pc_var, label="Principal Components")

    ax.legend()

    ax.set_xlabel("Components")
    ax.set_ylabel("Cumulative Explained Varaince")
    ax.set_ylim([0, 1])
    ax.set_xlim([0, len(ic_var)])

    return ax


def cluster_activities(
    ica_data,
    correlation_method="spearman",
    distance_threshold=None,
    show_thresholding=False,
    show_clustermap=True,
    show_best_clusters=False,
    n_best_clusters="auto",
    cluster_names=None,
    return_clustermap=False,
    # dimca options
    dimca_sample1=None,
    dimca_sample2=None,
    dimca_threshold=5,
    dimca_fdr=0.1,
    dimca_label=True,
    dimca_adjust=True,
    dimca_table=False,
    **dimca_kwargs,
):
    """
    Cluster all iModulon activity profiles using hierarchical clustering and display
    the results using :func:`seaborn.clustermap`

    Parameters
    ----------
    ica_data: ~pymodulon.core.IcaData
        :class:`~pymodulon.core.IcaData` object
    correlation_method: 'pearson', 'spearman', 'kendall', 'mutual_info' or callable
        Method for computing correlations between iModulon activities. See
        :meth:`pandas.DataFrame.corr` Default is 'spearman'.
    distance_threshold: float, optional
        A distance from 0 to 1 to define flat clusters from the hierarchical
        clustering. Larger values yield fewer clusters. If None, automatic selection
        of optimal threshold will occur by maximizing the silhouette score across
        iModulons. (default: None)
    show_thresholding: bool
        Show the plot of distance thresholds vs. silhouette scores (default: False)
    show_clustermap: bool
        Show the clustermap (default: True)
    show_best_clusters: bool
        Show individual clusters below complete clustermap
    n_best_clusters: int or str
        Number of best clusters to show. If 'auto', only clusters with above-average
        silhouette scores are shown
    cluster_names: dict, optional
        A dictionary mapping cluster indices to names, usually used after clustering
        has been performed previously.
    return_clustermap: bool
        Return the axis containing the clustermap
    dimca_sample1: list or str
        List of sample IDs or name of "project:condition" of reference samples for
        Differential iModulon Cluster Analysis (DiMCA)
    dimca_sample2: list or str
        List of sample IDs or name of "project:condition" of target samples for
        DiMCA
    dimca_threshold: float
        Minimum activity difference to determine DiMCAs
    dimca_fdr: float
        False detection rate for DiMCA
    dimca_label: bool
        Label differentially activated imodulon clusters (default: True)
    dimca_adjust: bool
        Auto-adjust DiMCA cluster labels (default: True)
    dimca_table: bool
        Return DiMCA table (default: False)
    **dimca_kwargs: dict, optional
        Additional keyword arguments passed to :func:`pymodulon.plotting.dima`

    Returns
    -------
    clusters: ~sklearn.cluster.AgglomerativeClustering
        :class:`sklearn.cluster.AgglomerativeClustering` of activity matrix
    cg: ~seaborn.matrix.ClusterGrid, optional
        :class:`~seaborn.matrix.ClusterGrid` containing the clusterplot
    dimca_ax: ~matplotlib.axes.Axes, optional
        :class:`~matplotlib.axes.Axes` containing the DiMCA scatterplot
    dimca_table: ~pandas.DataFrame, optional
        Table of differentially activated iModulon clusters
    """

    # compute distance matrix; distance metric defined as 1 - correlation to
    # ensure that correlated iModulons are close in distance, can be clustered

    if correlation_method == "mutual_info":
        correlation_df = 1 - ica_data.A.T.corr(method=mutual_info_distance)
        distance_matrix = 1 - correlation_df.abs() - np.eye(len(correlation_df))
        correlation_df = (correlation_df - correlation_df.min().min()) / (
            correlation_df.max().max()
        ) + np.eye(len(correlation_df))
    else:
        correlation_df = ica_data.A.T.corr(method=correlation_method)
        distance_matrix = 1 - correlation_df.abs()

    best_clusters = []
    cluster_score_dict = {}

    # define a base instance of the clustering object we will use throughout;
    # the distance threshold here is a dummy; we will replace this with either
    # the user-specified value or an automatically-selected value
    agg_cluster_base = AgglomerativeClustering(
        n_clusters=None,
        affinity="precomputed",
        compute_full_tree=True,
        linkage="complete",
        distance_threshold=0.5,
    )

    # perform automatic thresholding for default case
    if distance_threshold is None:

        auto_threshold_df = pd.DataFrame(columns=["threshold", "score", "n_clusters"])
        auto_threshold_df["threshold"] = np.arange(0, 1, 0.025)

        for row in auto_threshold_df.itertuples(index=True):

            # create a copy of the base clusterer with the threshold to try; fit
            agg_cluster_auto_threshold = clone(agg_cluster_base).set_params(
                distance_threshold=row.threshold
            )
            agg_cluster_auto_threshold.fit(distance_matrix)
            n_clusters = agg_cluster_auto_threshold.n_clusters_
            auto_threshold_df.loc[row.Index, "n_clusters"] = n_clusters

            # score the clustering; handle the edge case where all clusters have
            # size 1; this is invalid input for silhouette_score
            if n_clusters == distance_matrix.shape[0] or n_clusters == 1:
                auto_threshold_df.loc[row.Index, "score"] = 0
            else:
                auto_threshold_df.loc[row.Index, "score"] = silhouette_score(
                    distance_matrix,
                    agg_cluster_auto_threshold.labels_,
                    metric="precomputed",
                )

        best_threshold = auto_threshold_df.sort_values(
            by="score", ascending=False
        ).iloc[0]["threshold"]

        if show_thresholding:
            _, ax = plt.subplots()
            ax.yaxis.grid(False)
            sns.scatterplot(
                x="threshold", y="score", data=auto_threshold_df, ax=ax, color="blue"
            )
            ax.axvline(best_threshold, linestyle="--", color="k")
            ax.tick_params(axis="both", labelsize=13)
            ax.tick_params(axis="y", color="blue", labelcolor="blue")
            ax.set_xlabel("Threshold", fontsize=14)
            ax.set_ylabel("Silhouette", color="blue", fontsize=14)
            ax2 = ax.twinx()
            ax2.yaxis.grid(False)
            sns.scatterplot(
                x="threshold",
                y="n_clusters",
                data=auto_threshold_df,
                ax=ax2,
                color="red",
            )
            ax2.tick_params(axis="both", labelsize=13)
            ax2.tick_params(axis="y", color="red", labelcolor="red")
            ax2.set_ylabel("# Clusters", fontsize=14, color="red")
            plt.show()

    else:
        best_threshold = distance_threshold

    # re-perform the clustering with the best threshold from above
    agg_cluster_final = clone(agg_cluster_base).set_params(
        distance_threshold=best_threshold
    )
    agg_cluster_final.fit(distance_matrix)
    labels = agg_cluster_final.labels_

    returns = [agg_cluster_final]

    # compute silhouette scores for each cluster and determine what our best
    # clusters are; we only need to do this if we are displaying best clusters
    # OR if we're doing a DIMCA later
    if show_best_clusters or dimca_sample1 is not None:

        sample_scores = silhouette_samples(
            distance_matrix, labels, metric="precomputed"
        )
        cluster_score_dict = {}
        for label in set(labels):
            cluster_score_dict[label] = np.mean(
                np.array(sample_scores)[np.where(labels == label)[0]]
            )

        # calculate the best clusters based on the requested method
        if n_best_clusters == "auto":
            mean_cluster_score = np.mean(list(cluster_score_dict.values()))
            best_clusters = [
                cluster
                for cluster, score in cluster_score_dict.items()
                if score > mean_cluster_score
            ]
        else:
            best_clusters = list(
                list(
                    zip(
                        *sorted(
                            cluster_score_dict.items(),
                            key=lambda label_score: label_score[1],
                            reverse=True,
                        )
                    )
                )[0]
            )[: min(n_best_clusters, len(cluster_score_dict))]

    if show_clustermap:

        # prepare a linkage matrix so that we can specify the dendrogram to sns
        # https://stackoverflow.com/questions/29127013/
        children = agg_cluster_final.children_
        distance = np.arange(children.shape[0])
        no_of_observations = np.arange(2, children.shape[0] + 2)
        linkage_matrix = np.column_stack(
            [children, distance, no_of_observations]
        ).astype(float)

        clustermap = sns.clustermap(
            correlation_df,
            row_linkage=linkage_matrix,
            col_linkage=linkage_matrix,
            xticklabels=False,
            yticklabels=False,
            figsize=(10, 10),
            center=0,
        )

        if correlation_method == "mutual_info":
            clustermap.ax_cbar.set_ylabel("Normalized MI", fontsize=14)
        else:
            clustermap.ax_cbar.set_ylabel(f"{correlation_method} R", fontsize=14)

        # the following code draws squares on the clustermap to highlight
        # the cluster locations; will also number the best clusters if they
        # are to be called out after the fact
        cluster_size_dict = Counter(labels)
        size_counter = 1
        top_left = 0
        best_cluster_labels = []
        best_cluster_matrices = []
        best_cluster_scores = []
        for i, imod_idx in enumerate(clustermap.dendrogram_row.reordered_ind):

            # draw a box around the cluster if we're at the end of the cluster
            cluster_for_imod = agg_cluster_final.labels_[imod_idx]
            cluster_size = cluster_size_dict[cluster_for_imod]
            if cluster_size == size_counter:
                # usually Rectangle wants bottom left, but the heatmap that sns
                # makes has the positive direction reversed (axes positions
                # increase towards the bottom right)
                clustermap.ax_heatmap.add_patch(
                    Rectangle(
                        (top_left, top_left),
                        cluster_size,
                        cluster_size,
                        fill=False,
                        color="white",
                        lw=1,
                    )
                )

                # also add a number (or name) IF we're calling out later
                if show_best_clusters:
                    if cluster_for_imod in best_clusters:
                        if cluster_names is not None:
                            cluster_name = cluster_names.get(
                                cluster_for_imod, cluster_for_imod
                            )
                        else:
                            cluster_name = cluster_for_imod
                        clustermap.ax_heatmap.text(
                            top_left + cluster_size + 0.05,
                            top_left - 0.05,
                            str(cluster_name),
                            color="white",
                            fontsize=16,
                        )
                        # stash the clustermap data for the best cluster
                        best_cluster_submatrix = clustermap.data2d.iloc[
                            top_left : (top_left + cluster_size),
                            top_left : (top_left + cluster_size),
                        ]
                        best_cluster_labels.append(cluster_for_imod)
                        best_cluster_matrices.append(best_cluster_submatrix)
                        best_cluster_scores.append(cluster_score_dict[cluster_for_imod])

                # reset the counters being used to establish where to draw boxes
                size_counter = 1
                top_left = i + 1

            else:
                size_counter += 1

        if return_clustermap:
            returns.append(clustermap)

        plt.show()

        # now actually display the best clusters if asked to
        if show_best_clusters:

            # calculate the figure size and arrangement needed to cleanly
            # display the number of best clusters we have on hand
            subplot_rows, subplot_cols = 1, 1
            while subplot_rows * subplot_cols < len(best_cluster_matrices):
                if subplot_rows / subplot_cols >= 1:
                    subplot_cols += 1
                else:
                    subplot_rows += 1
            if len(best_cluster_matrices) <= 4:
                figsize = (10, 6)
            elif len(best_cluster_matrices) <= 30:
                figsize = (14, 9)
            else:
                figsize = (20, 14)

            _, axs = plt.subplots(subplot_rows, subplot_cols, figsize=figsize)
            axs = axs.flatten()

            # sort the best clusters by score to display the best one first
            sorted_lab_mtrx_score_tups = sorted(
                zip(best_cluster_labels, best_cluster_matrices, best_cluster_scores),
                key=lambda label_matrix_score_tup: label_matrix_score_tup[2],
                reverse=True,
            )
            for lab_mtrx_score_tup, ax in zip(sorted_lab_mtrx_score_tups, axs):

                cluster_lab, cluster_mtrx, cluster_score = lab_mtrx_score_tup

                sns.heatmap(
                    cluster_mtrx,
                    xticklabels=False,
                    center=0,
                    square=True,
                    cbar=False,
                    ax=ax,
                )
                if cluster_names is not None:
                    cluster_name = cluster_names.get(
                        cluster_lab, f"Cluster {cluster_lab}"
                    )
                    cluster_title = f"{cluster_name} ({cluster_score:.2f})"
                else:
                    cluster_title = f"Cluster {cluster_lab} " f"({cluster_score:.2f})"
                ax.set_title(cluster_title, fontsize=16)
                ax.set_yticks(np.arange(0.5, cluster_mtrx.shape[0] + 0.5, 1))
                ax.set_yticklabels(cluster_mtrx.index)
                ax.tick_params(axis="both", labelsize=11)
                ax.tick_params(axis="y", rotation=0)

            for i in range(1, len(axs) - len(best_cluster_matrices) + 1):
                axs[-i].set_visible(False)

            plt.tight_layout()

    # do the requested DIMCA comparison if asked; re-use machinery from base
    # DIMA as much as possible
    if dimca_sample1 is not None:

        # compute the new activity matrix with the best cluster activities
        # consolidated; define positive direction as direction with more + corrs
        cluster_A_df = pd.DataFrame(columns=ica_data.A.columns)
        all_clustered_ims = []
        for best_cluster_label in best_clusters:

            cluster_im_inds = np.where(labels == best_cluster_label)[0]
            cluster_ims = np.array(ica_data.A.index)[cluster_im_inds]
            all_clustered_ims += list(cluster_ims)
            cluster_im_A_df = ica_data.A.loc[cluster_ims]
            cluster_corr_df = correlation_df.loc[cluster_ims, cluster_ims]

            # get the average non-self correlation for each iM in the cluster
            # use this to invert the row in the cluster A df if negative
            cluster_size = cluster_corr_df.shape[0]
            for i in range(cluster_size):
                non_self_inds = list(set(list(range(cluster_size))) - {i})
                non_self_corrs = cluster_corr_df.iloc[i, non_self_inds]
                im_corr_mean = np.mean(non_self_corrs)
                if im_corr_mean < 0:
                    cluster_im_A_df.iloc[i, :] = cluster_im_A_df.iloc[i, :] * -1

            # compute the final averaged cluster activity, add to new dataframe
            # use the provided name for this cluster if we have one
            if cluster_names is not None and best_cluster_label in cluster_names:
                cluster_name = f"{cluster_names[best_cluster_label]} [Clst]"
            else:
                cluster_name = f"Cluster {best_cluster_label}"
            cluster_A_df.loc[cluster_name] = cluster_im_A_df.mean(axis=0)

        # now we can add in the singleton iModulons to our new A matrix
        singleton_ims = set(list(ica_data.A.index)) - set(all_clustered_ims)
        cluster_A_df = cluster_A_df.append(ica_data.A.loc[list(singleton_ims)])

        # add to kwargs
        dimca_kwargs["alternate_A"] = cluster_A_df

        # now we can pretty much proceed as normal with DIMCA; just have a
        # different activity matrix, but the procedure is the same from here
        dimca_return = plot_dima(
            ica_data,
            sample1=dimca_sample1,
            sample2=dimca_sample2,
            threshold=dimca_threshold,
            fdr=dimca_fdr,
            label=dimca_label,
            adjust=dimca_adjust,
            table=dimca_table,
            **dimca_kwargs,
        )
        if dimca_table:
            returns += dimca_return
        else:
            returns.append(dimca_return)

    return returns


####################
# Metadata Boxplot #
####################


def metadata_boxplot(
    ica_data,
    imodulon,
    n_boxes=3,
    strip_conc=True,
    ignore_cols=None,
    use_cols=None,
    return_results=False,
):
    """
    Uses a decision tree regressor to automatically cluster iModulon activities
    using metadata. Displays results as a box plot.

    Parameters
    ----------
    ica_data: ~pymodulon.core.IcaData
        :class:`~pymodulon.core.IcaData` object
    imodulon : int or str
        `iModulon` name
    n_boxes: int
        Number of boxes to create
    strip_conc: bool
        Remove concentrations from metadata (e.g. "glucose(2g/L)" would be
        interpreted as just "glucose")
    ignore_cols: list, optional
        List of columns to ignore. If None, only "project" and "condition" are
        ignored
    use_cols: list, optional
        List of columns to use. This supercedes ignore_cols.
    return_results: bool
        Return a dataframe describing the classifications

    Returns
    -------
    ax: ~matplotlib.axes.Axes
        :class:`~matplotlib.axes.Axes` containing the boxplot
    df_classes: ~pd.DataFrame
        Metadata classifications of the samples
    """
    activities = ica_data.A.loc[imodulon]

    encoding, features = _encode_metadata(
        ica_data, ignore_cols=ignore_cols, use_cols=use_cols, strip_conc=strip_conc
    )

    clf = _train_classifier(activities, features, max_leaf_nodes=n_boxes)
    labels = _get_labels_from_tree(clf, encoding)
    df_classes = _get_sample_leaves(clf, features, labels, activities)

    df_classes = df_classes.sort_values(imodulon, ascending=False)

    fig, ax = plt.subplots()
    sns.boxplot(data=df_classes, x=imodulon, y="category")
    ax.set_ylabel("")
    if return_results:
        return ax, df_classes
    else:
        return ax


def _encode_metadata(ica_data, use_cols=None, ignore_cols=None, strip_conc=True):
    # Convert values to strings
    metadata = ica_data.sample_table.copy()
    metadata = metadata.astype(str).fillna("")

    # Select columns

    if ignore_cols is None:
        ignore_cols = []

    if use_cols is not None:
        metadata = metadata[use_cols]
    else:
        metadata.drop(set(ignore_cols) & set(metadata.columns), axis=1, inplace=True)

    # Remove concentrations
    if strip_conc:
        metadata = metadata.replace(r"(.*?)\(.*?\)", r"\1", regex=True)

    # One-hot encode features
    enc = OneHotEncoder()
    enc.fit(metadata)
    features = enc.fit_transform(metadata)

    # Get feature information
    list2struct = []
    for cat, vals in zip(metadata.columns, enc.categories_):
        list2struct += [[cat, val] for val in vals]
    encoding = pd.DataFrame(list2struct, columns=["category", "value"])

    return encoding, features


def _train_classifier(component, features, max_leaf_nodes=3):
    # Run Decision Tree Regressor
    clf = DecisionTreeRegressor(
        min_samples_leaf=2, criterion="mae", max_leaf_nodes=max_leaf_nodes
    )
    clf.fit(features, component)
    return clf


def _get_labels_from_tree(clf, encoding):
    # Load tree information
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature_ids = clf.tree_.feature

    # Initialize traversal
    stack = [(0, "")]  # node_id, label
    final_labels = [""] * len(feature_ids)

    # Traverse Tree
    while len(stack) > 0:
        node_id, label = stack.pop()

        # Add feature split to dataframe
        feat = feature_ids[node_id]

        # Check for leaf
        if feat != -2:

            cat, val = encoding.loc[feat].values
            label_left = f"{label}\n{cat}: NOT {val}"
            label_right = f"{label}\n{cat}: {val}"

            # Add children to stack
            stack.append((children_left[node_id], label_left))
            stack.append((children_right[node_id], label_right))

        else:
            final_labels[node_id] = label.strip()

    return final_labels


def _get_sample_leaves(clf, features, labels, component):
    # Initialize final dataframe
    DF_result = pd.DataFrame(component)
    DF_result["category"] = ""

    # Get decision path
    node_mat = clf.decision_path(features)

    for i, lab in enumerate(labels):
        if lab != "":
            DF_result.iloc[sparse.find(node_mat[:, i])[0], 1] = lab

    return DF_result


####################
# Helper Functions #
####################


def _fit_line(x, y, ax, metric):
    # Get line parameters and metric of correlation/regression
    if metric is None:
        return

    if metric == "r2":
        params, r2 = _get_fit(x, y)
        label = "$R^2_{{adj}}$ = {:.2f}".format(r2)

    elif metric == "pearson":
        params = curve_fit(_solid_line, x, y)[0]
        r, pval = stats.pearsonr(x, y)
        if pval < 1e-10:
            label = f"Pearson R = {r:.2f}\np-value < 1e-10"
        else:
            label = f"Pearson R = {r:.2f}\np-value = {pval:.2e}"

    elif metric == "spearman":
        params = curve_fit(_solid_line, x, y)[0]
        r, pval = stats.spearmanr(x, y)
        if pval < 1e-10:
            label = f"Spearman R = {r:.2f}\np-value < 1e-10"
        else:
            label = f"Spearman R = {r:.2f}\np-value = {pval:.2e}"

    else:
        raise ValueError('Metric must be "pearson", "spearman", or "r2"')

    # Plot line
    if len(params) == 2:
        xvals = np.array([min(x), max(x)])
        ax.plot(
            xvals,
            _solid_line(xvals, *params),
            label=label,
            color="k",
            linestyle="dashed",
            linewidth=1,
            zorder=5,
        )
    else:
        mid = params[2]
        xvals = np.array([x.min(), mid, x.max()])
        ax.plot(
            xvals,
            _broken_line(xvals, *params),
            label=label,
            color="k",
            linestyle="dashed",
            linewidth=1,
            zorder=5,
        )


def _get_fit(x, y):
    all_params = [curve_fit(_solid_line, x, y)[0]]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=OptimizeWarning)
        for c in [min(x), np.mean(x), max(x)]:
            try:
                all_params.append(
                    curve_fit(_broken_line, x, y, p0=[1, 1, c], maxfev=5000)[0]
                )
            except OptimizeWarning:
                pass

    best_r2 = -np.inf
    best_params = all_params[0]

    for params in all_params:
        if len(params) == 2:
            r2 = _adj_r2(_solid_line, x, y, params)
        else:
            r2 = _adj_r2(_broken_line, x, y, params)

        if r2 > best_r2:
            best_r2 = r2
            best_params = params

    if best_r2 < 0:
        return [0, np.mean(y)], 0

    return best_params, best_r2


def _broken_line(x, A, B, C):
    y = np.zeros(len(x), dtype=np.float)
    y += (A * x + B) * (x >= C)
    y += (A * C + B) * (x < C)
    return y


def _solid_line(x, A, B):  # this is your 'straight line' y=f(x)
    y = A * x + B
    return y


def _adj_r2(f, x, y, params):
    n = len(x)
    k = len(params) - 1
    r2 = r2_score(y, f(x, *params))
    return 1 - np.true_divide((1 - r2) * (n - 1), (n - k - 1))


def _mod_freedman_diaconis(ica_data, imodulon):
    """
    Generates bins using optimal bin width estimate using the Freedman-Diaconis rule

    Parameters
    ----------
    ica_data: ~pymodulon.core.IcaData
        :class:`~pymodulon.core.IcaData` object
    imodulon : int or str
        `iModulon` name

    Returns
    -------
    bins: ~numpy.ndarray
        Numpy array of bins

    See Also
    --------
    Wikipedia:
        {https://en.wikipedia.org/wiki/Freedman-Diaconis_rule}
    """
    x = ica_data.M[imodulon]
    thresh = abs(ica_data.thresholds[imodulon])

    # Modified Freedman-Diaconis
    opt_width = (x.max() - x.min()) / (len(x) ** (1 / 3))

    # Width calculated using optimal width and iModulon threshold
    if thresh > opt_width:
        width = thresh / int(thresh / opt_width)
    else:
        width = thresh / 2

    # Use width and thresh to calculate xmin, xmax
    if x.min() < -thresh:
        multiple = np.ceil(abs(x.min() / width))
        xmin = -(multiple + 1) * width
    else:
        xmin = -(thresh + width)

    if x.max() > thresh:
        multiple = np.ceil(x.max() / width)
        xmax = (multiple + 1) * width
    else:
        xmax = thresh + width

    return np.arange(xmin, xmax + width, width)


def _save_figures(fig, savefig_kwargs):
    """Check for savefig_kwargs, then save current figure instance"""

    # Check for savefig_kwargs
    if savefig_kwargs:
        savefig_kwargs["fname"] = savefig_kwargs.get("fname", "./plot.svg")
    else:
        savefig_kwargs = {"fname": "./plot.svg"}

    # Plot current figure instance using savefig_kwargs
    fig.savefig(**savefig_kwargs)


##########################
# Experimental Functions #
##########################


def _set_xaxis(xaxis, y):
    """
    Implements experimental `xaxis` param from `plot_gene_weights`. This
    allows for users to generate a scatterplot comparing gene weight on
    the y-axis with any collection of numbers on the x-axis (as long as
    the lengths match).

    Parameters
    ----------
    xaxis: list, set, tuple, dict, np.array, pd.Series
        Any collection or mapping of numbers (plots on x-axis)
    y: pd.Series
        pandas Series of Gene Weights to be plotted on the y-axis of
        `plot_gene_weights`

    Returns
    -------
    x: pd.Series
        Returns a pd.Series to be used as the x-axis data-points for
        generating the plot_gene_weights scatter-plot.
    """
    # Determine type of `xaxis` and set `x` accordingly
    x = xaxis if isinstance(xaxis, pd.Series) else pd.Series(xaxis)

    if not x.sort_index().index.equals(y.sort_index().index):
        raise ValueError(
            "Given x-values do not align with gene and their "
            "respective gene-weights on the y-axis"
        )

    return x
