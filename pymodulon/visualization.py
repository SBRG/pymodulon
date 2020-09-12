"""

"""
import warnings
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from adjustText import adjust_text
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.optimize import curve_fit, OptimizeWarning
from sklearn.metrics import r2_score
from itertools import combinations

from pymodulon.core import IcaData
from pymodulon.enrichment import parse_regulon_str, FDR
from pymodulon.util import Ax, ImodName, SeqSetStr, name2num


#############
# Bar Plots #
#############

def barplot(values: pd.Series, sample_table: pd.DataFrame,
            ylabel: str = '',
            projects: Optional[Union[List, str]] = None,
            highlight: Optional[Union[List, str]] = None,
            ax: Optional[Ax] = None,
            legend_kwargs: Optional[Mapping] = None) -> Ax:
    """
    Creates an overlaid scatter and barplot for a set of values (either gene
    expression levels or iModulon activities)

    Args:
        values: List of values to plot
        sample_table: Sample table from IcaData object
        ylabel: y-axis label
        projects: Project(s) to show
        highlight: Project(s) to highlight
        ax: Matplotlib axis object
        legend_kwargs: Dictionary of arguments for the legend

    Returns: A matplotlib axis object

    """

    # Remove extra projects
    if isinstance(projects, str):
        projects = [projects]

    if projects is not None and 'project' in sample_table:
        sample_table = sample_table[sample_table.project.isin(projects)]
        values = values[sample_table.index]

    if ax is None:
        figsize = (len(values) / 15 + 0.5, 2)
        fig, ax = plt.subplots(figsize=figsize)

    # Get ymin and max
    ymin = values.min()
    ymax = values.max()
    yrange = ymax - ymin
    ymax = max(1, max(ymax * 1.1, ymax + yrange * 0.1))
    ymin = min(-1, min(ymin * 1.1, ymin - yrange * 0.1))
    yrange = ymax - ymin

    # Add project-specific information
    if 'project' in sample_table.columns and \
            'condition' in sample_table.columns:

        # Sort data by project/condition to ensure replicates are together
        metadata = sample_table.loc[:, ['project', 'condition']]
        metadata = metadata.sort_values(['project', 'condition'])
        metadata['name'] = metadata.project + ' - ' + metadata.condition

        # Coerce highlight to iterable
        if highlight is None:
            highlight = []
        elif isinstance(highlight, str):
            highlight = [highlight]

        # Get X and Y values for scatter points
        metadata['y'] = values
        metadata['x'] = np.cumsum(~metadata[['name']].duplicated())

        # Get heights for barplot
        bar_vals = metadata.groupby('x').mean()

        # Add colors and names
        bar_vals['name'] = metadata.drop_duplicates('name').name.values
        bar_vals['project'] = metadata.drop_duplicates('name').project.values

        # Plot bars for highlighted samples
        color_vals = bar_vals[bar_vals.project.isin(highlight)]
        color_cycle = ['tab:red', 'tab:orange', 'tab:green',
                       'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                       'tab:olive', 'tab:cyan']
        i = 0
        for name, group in color_vals.groupby('name'):
            ax.bar(group.index, group.y, color=color_cycle[i], width=1,
                   linewidth=0, align='edge', zorder=1, label=name)
            i = (i + 1) % len(color_cycle)

        # Plot bars for non-highlighted samples
        other_vals = bar_vals[~bar_vals.project.isin(highlight)]
        ax.bar(other_vals.index, other_vals.y, color='tab:blue', width=1,
               linewidth=0, align='edge', zorder=1, label=None)
        ax.scatter(metadata.x + 0.5, metadata.y, color='k', zorder=2, s=10)

        # Get project names and sizes
        projects = metadata.project.drop_duplicates()
        md_cond = metadata.drop_duplicates(['name'])
        project_sizes = [len(md_cond[md_cond.project == proj]) for proj in
                         projects]
        nbars = len(md_cond)

        # Draw lines to discriminate between projects
        proj_lines = np.cumsum([1] + project_sizes)
        ax.vlines(proj_lines, ymin, ymax,
                  colors='lightgray',
                  linewidth=1)

        # Add project names
        texts = []
        start = 2
        for proj, size in zip(projects, project_sizes):
            x = start + size / 2
            texts.append(ax.text(x, ymin - yrange * 0.02, proj, ha='right',
                                 va='top', rotation=45))
            start += size

        # Add legend
        if not color_vals.empty:
            kwargs = {'bbox_to_anchor': (1, 1), 'ncol': len(
                color_vals.name.unique()) // 6 + 1}

            if legend_kwargs is not None:
                kwargs.update(legend_kwargs)

            ax.legend(**kwargs)

    else:
        warnings.warn('Missing "project" and "condition" columns in sample '
                      'table.')
        ax.bar(range(len(values)), values, width=1, align='edge')
        nbars = len(values)

    # Set axis limits
    xmin = -0.5
    xmax = nbars + 2.5
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    # Axis labels
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks([])

    # X-axis
    ax.hlines(0, xmin, xmax, color='k')

    return ax


def plot_expression(ica_data: IcaData, gene: str,
                    projects: Union[List, str] = None,
                    highlight: Union[List, str] = None,
                    ax: Optional[Ax] = None,
                    legend_kwargs: Optional[Mapping] = None) -> Ax:
    """
    Creates a barplot showing an gene's expression across the compendium
    Args:
        ica_data: IcaData Object
        gene: Gene locus tag or name
        projects: Name(s) of projects to show (default: show all)
        highlight: Name(s) of projects to highlight (default: None)
        ax: Matplotlib axis object
        legend_kwargs: Dictionary of arguments to be passed to legend

    Returns: A matplotlib axis object

    """
    # Check that gene exists
    if gene in ica_data.X.index:
        values = ica_data.X.loc[gene]
        label = '{} Expression'.format(gene)
    else:
        locus = name2num(ica_data, gene)
        values = ica_data.X.loc[locus]
        label = '${}$ Expression'.format(gene)

    return barplot(values, ica_data.sample_table, label, projects,
                   highlight, ax, legend_kwargs)


def plot_activities(ica_data: IcaData, imodulon: ImodName,
                    projects: Union[List, str] = None,
                    highlight: Union[List, str] = None,
                    ax: Optional[Ax] = None,
                    legend_kwargs: Optional[Mapping] = None) -> Ax:
    """
    Creates a barplot showing an iModulon's activity across the compendium
    Args:
        ica_data: IcaData Object
        imodulon: iModulon name
        projects: Name(s) of projects to show (default: show all)
        highlight: Name(s) of projects to highlight (default: None)
        ax: Matplotlib axis object
        legend_kwargs: Dictionary of arguments to be passed to legend

    Returns: A matplotlib axis object

    """
    # Check that iModulon exists
    if imodulon in ica_data.A.index:
        values = ica_data.A.loc[imodulon]
    else:
        raise ValueError(f'iModulon does not exist: {imodulon}')

    label = '{} iModulon\nActivity'.format(imodulon)

    return barplot(values, ica_data.sample_table, label, projects,
                   highlight, ax, legend_kwargs)


def plot_metadata(ica_data: IcaData, column,
                  projects: Union[List, str] = None,
                  highlight: Union[List, str] = None,
                  ax: Optional[Ax] = None,
                  legend_kwargs: Optional[Mapping] = None) -> Ax:
    """
    Creates a barplot for values in the sample table

    Args:
        ica_data: IcaData Object
        column: Column name to plot
        projects: Name(s) of projects to show (default: show all)
        highlight: Name(s) of projects to highlight (default: None)
        ax: Matplotlib axis object
        legend_kwargs: Dictionary of arguments to be passed to legend

    Returns: A matplotlib axis object
    """
    # Check that column exists
    if column in ica_data.sample_table.columns:
        # Make sure the column is filled with numbers
        if not pd.api.types.is_numeric_dtype(ica_data.sample_table[column]):
            raise ValueError('Metadata column {} is not numeric'.format(column))

        # Remove null values
        table = ica_data.sample_table[ica_data.sample_table[column].notnull()]
        values = ica_data.sample_table[column]

    else:
        raise ValueError('Column not in sample table: {}'.format(column))

    return barplot(values, table, column, projects,
                   highlight, ax, legend_kwargs)


def plot_regulon_histogram(ica_data: IcaData, imodulon: ImodName,
                           regulator: str = None,
                           bins: Optional[Union[int, Sequence, str]] = None,
                           kind: Union[Literal['overlap'],
                                       Literal['side']] = 'overlap',
                           ax: Optional[Ax] = None,
                           hist_label: Tuple[str, str] = ('Not regulated',
                                                          'Regulon Genes'),
                           color: Union[Sequence[Tuple],
                                        Sequence[str]] = ('#aaaaaa', 'salmon'),
                           alpha: float = 0.7,
                           ax_font_kwargs: Optional[Mapping] = None,
                           legend_kwargs: Optional[Mapping] = None) -> Ax:
    """
    Plots a histogram of regulon vs non-regulon genes by iModulon weighting.

    Parameters
    ----------
    ica_data: pymodulon.core.IcaData
        IcaData container object
    imodulon: int, str
        The name of the iModulon to plot in regards to. Used to determine
        gene weights
    regulator: str
        Name of regulator to compare enrichment against. Determines which
        genes are in the regulon and which are not.
    bins: int, Sequence, str
        The bins to use when generating the histogram. Passed on to
        `ax.hist()`
    kind: 'overlap', 'side'
        Whether to plot an overlapping or side-by-side comparison histogram
    ax: matplotlib.axes instance
        The axes instance on which to generate the scatter-plot. If None is
        provided, generates a new figure and axes instance to use
    hist_label: Tuple[str, str]
        The label to use when plotting the regulon and non-regulon genes.
        Takes into a tuple of 2 values (first for non-regulon genes,
        second for regulon genes). Passed on to `ax.hist()`
    color: Sequence of tuples and/or str
        The colors to use for regulon and non-regulon genes. Takes a
        Sequence of 2 values (first for non-regulon genes, second for
        regulon genes). Passed on to `ax.hist()`
    alpha: float
        Sets the opacity of the histogram (0 = transparent, 1 = opaque).
        Passed on to `ax.hist()`
    ax_font_kwargs: dict
        kwargs that are passed onto `ax.set_xlabel()` and `ax.set_ylabel()`
    legend_kwargs: dict
        kwargs that are passed onto `ax.legend()`

    Returns
    -------
    ax: matplotlib.axes instance
        Returns the axes instance on which the histogram is generated
    """
    # Check that iModulon exists
    if imodulon not in ica_data.M.columns:
        raise ValueError(f'iModulon does not exist: {imodulon}')

    # If ax is None, create ax on which to generate histogram
    if ax is None:
        fig, ax = plt.subplots()

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
        reg = ica_data.imodulon_table.loc[imodulon, 'regulator']
        if pd.isnan(reg):
            reg = None

    # If no imodulon_table in IcaData, compute trn enrichment to find the
    # regulator. Note that trn enrichment is computed using `max_regs` = 1
    else:
        df_enriched = ica_data.compute_trn_enrichment(imodulons=imodulon)
        df_top_enrich = df_enriched.sort_values(
            ['imodulon', 'qvalue', 'n_regs']).drop_duplicates('imodulon')
        reg = df_top_enrich.set_index('imodulon').loc[imodulon, 'regulator']
        if not isinstance(reg, str):
            if pd.isnan(reg):
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
        legend_kwargs = dict({'loc': 'upper right'})

    # Histogram
    non_reg_genes = set(ica_data.gene_names) - reg_genes
    reg_arr = ica_data.M[imodulon].loc[reg_genes]
    non_reg_arr = ica_data.M[imodulon].loc[non_reg_genes]

    if kind == 'overlap':
        ax.hist(non_reg_arr, bins=bin_arr, alpha=alpha, color=color[0],
                label=hist_label[0])
        ax.hist(reg_arr, bins=bin_arr, alpha=alpha, color=color[1],
                label=hist_label[1])

    elif kind == 'side':
        arr = np.array([non_reg_arr, reg_arr], dtype='object')
        ax.hist(arr, bins=bin_arr, alpha=alpha, color=color, label=hist_label)

    else:
        raise ValueError(f'{kind} is not a valid option. `kind` must be '
                         'either "overlap" or "side"')

    # Set y-axis to log-scale
    ax.set_yscale('log')

    # Add thresholds to scatterplot (dashed lines)
    ymin, ymax = ax.get_ylim()
    thresh = abs(ica_data.thresholds[imodulon])
    if thresh != 0:
        ax.vlines([-thresh, thresh], ymin=ymin, ymax=ymax,
                  colors='k', linestyles='dashed', linewidth=1)

    ax.set_ylim(ymin, ymax)

    # Set x and y labels
    ax.set_xlabel(f'{imodulon} Gene Weight', **ax_font_kwargs)
    ax.set_ylabel('Number of Genes', **ax_font_kwargs)

    # Add legend
    ax.legend(**legend_kwargs)

    return ax


################
# Scatterplots #
################

def scatterplot(x: pd.Series, y: pd.Series,
                groups: Optional[Mapping] = None,
                show_labels: Union[bool, Literal['auto']] = 'auto',
                adjust_labels: bool = True,
                line45: bool = False,
                line45_margin: float = 0,
                fit_line: bool = False,
                fit_metric: Union[Literal['pearson'], Literal['spearman'],
                                  Literal['r2adj']] = 'pearson',
                xlabel: str = '', ylabel: str = '',
                ax: Optional[Ax] = None,
                legend: bool = True,
                ax_font_kwargs: Optional[Mapping] = None,
                scatter_kwargs: Optional[Mapping] = None,
                label_font_kwargs: Optional[Mapping] = None,
                legend_kwargs: Optional[Mapping] = None) -> Ax:
    """
    Generates a scatter-plot of the data given, with options for coloring by
    group, adding labels, adding lines, and generating correlation or
    determination coefficients.

    Parameters
    ----------
    x: pd.Series
        The data to be plotted on the x-axis
    y: pd.Series
        The data to be plotted on the x-axis
    groups: dict
        A mapping of data-points that form groups in the data
    show_labels: bool, str
        An option that toggles whether data-points are given labels
    adjust_labels: bool
        An option that ensures labels on data are sufficiently spread out
        and readable
    line45: bool
        An option to add a 45 degree line to the scatter-plot, useful
        for comparison with R^2 values
    line45_margin: float
        An option that adds margins around the 45 degree line. The larger
        this number, the larger the margin (distance from line45)
    fit_line: bool
        An option to add a line of best fit on the scatter-plot
    fit_metric: str
        The metric to use for finding the line of best fit. Options include
        pearson-r, spearman-r, or r^2
    xlabel: str
        The label to use for the x-axis of the plot
    ylabel: str
        The label to use for the y-axis of the plot
    ax: matplotlib.axes instance
        The axes instance on which to generate the scatter-plot. If None is
        provided, generates a new figure and axes instance to use
    legend: bool
        An option on whether to show the legend
    ax_font_kwargs: dict
        kwargs that are passed onto `ax.set_xlabel()` and `ax.set_ylabel()`
    scatter_kwargs: dict
        kwargs that are passed onto `ax.scatter()`
    label_font_kwargs: dict
        kwargs that are passed onto `ax.text()`
    legend_kwargs: dict
        kwargs that are passed onto `ax.legend()`

    Returns
    -------
    ax: matplotlib.axes instance
        Returns the axes instance on which the scatter-plot is generated
    """

    if ax is None:
        fig, ax = plt.subplots()

    if show_labels == 'auto':
        show_labels = (len(x) <= 20)

    if not (isinstance(x, pd.Series) and isinstance(y, pd.Series) and
            (x.sort_index().index == y.sort_index().index).all()):
        raise TypeError('X and Y must be pandas series with the same index')

    # Set up data object
    data = pd.DataFrame({'x': x, 'y': y})

    # Add group information
    data['group'] = ''
    if groups is not None:
        for k, val in groups.items():
            data.loc[k, 'group'] = val

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
        ax.plot([allmin, allmax], [allmin, allmax], color='k',
                linestyle='dashed', linewidth=0.5, zorder=0)

        if line45_margin > 0:
            diff = pd.DataFrame(abs(data.x - data.y), index=data.index)
            diff = diff.loc[diff[0] > line45_margin]
            data.loc[diff.index, 'group'] = 'hidden'
            ax.plot([max(xmin, ymin + line45_margin),
                     min(xmax, ymax + line45_margin)],
                    [max(ymin, xmin - line45_margin),
                     min(ymax, xmax - line45_margin)],
                    color='gray', linestyle='dashed',
                    linewidth=0.5, zorder=0)
            ax.plot([max(xmin, ymin - line45_margin),
                     min(xmax, ymax - line45_margin)],
                    [max(ymin, xmin + line45_margin),
                     min(ymax, xmax + line45_margin)],
                    color='gray', linestyle='dashed',
                    linewidth=0.5, zorder=0)

    for name, group in data.groupby('group'):

        # Override defaults for hidden points
        kwargs = scatter_kwargs.copy()
        if name == 'hidden':
            kwargs.update({'c': 'gray', 'alpha': 0.7, 'linewidth': 0,
                           'label': None})
        elif name == '':
            kwargs.update({'label': None})
        else:
            kwargs.update({'label': name})

        ax.scatter(group.x, group.y, **kwargs, zorder=1)

    # Add regression
    if fit_line:
        _fit_line(x, y, ax, fit_metric)

    # Add lines at 0
    if xmin < 0 < xmax:
        ax.hlines(0, xmin, xmax, linewidth=0.5, color='gray', zorder=2)
    if ymin < 0 < ymax:
        ax.vlines(0, ymin, ymax, linewidth=0.5, color='gray', zorder=2)

    # Add labels
    if show_labels:
        texts = []
        for idx in x.index:
            texts.append(ax.text(x[idx], y[idx], idx, **label_font_kwargs))
        if adjust_labels:
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle="-", color='k', lw=0.5),
                        only_move={'objects': 'y'},
                        expand_objects=(1.2, 1.4),
                        expand_points=(1.3, 1.3))

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    ax.set_xlabel(xlabel, **ax_font_kwargs)
    ax.set_ylabel(ylabel, **ax_font_kwargs)

    if (legend and legend_kwargs) or fit_line:
        ax.legend(**legend_kwargs)

    return ax


def plot_gene_weights(ica_data: IcaData, imodulon: ImodName,
                      xaxis=None, xname='',
                      by: Union[Literal['log-tpm-norm'], Literal['length'],
                                Literal['start'], None] = None,
                      ref_cols: Optional[SeqSetStr] = None,
                      **kwargs) -> Ax:
    """
    Generates a scatter-plot, with gene weights on the y-axis, and either
    the mean expression, gene length, or gene start site on the x-axis.
    Also shows the D'Agostino cutoff. Labels the statistically
    enriched genes, if the appropriate parameters are given.

    Parameters
    ----------
    ica_data: pymodulon.core.IcaData
        IcaData container object
    imodulon: int, str
        The name of the iModulon to plot
    xaxis:
        Experimental parameter. See `_set_axis()` for further details.
    xname:
        Experimental parameter. See `_set_axis()` for further details.
    by: 'log-tpm-norm', 'length', 'start'
        Gene property to plot on the x-axis. log-tpm-norm plots mean
        expression, length plots gene length, and start plots gene start
        position
    ref_cols: Sequence, set, or str
        A str or list of str values to use for normalizing the log-tpm data.
        Only used if 'log-tpm-norm' is given for the `by` parameter.
    **kwargs: dict
        keyword arguments passed onto `scatterplot()`

    Returns
    -------
    ax: matplotlib.axes instance
        Returns the axes instance on which the scatter-plot is generated
    """
    # Check that iModulon exists
    if imodulon in ica_data.M.columns:
        y = ica_data.M[imodulon]
        ylabel = f'{imodulon} Gene Weight'
    else:
        raise ValueError(f'iModulon does not exist: {imodulon}')

    # If experimental `xaxis` parameter is used, use custom values for x-axis
    if xaxis is not None:
        x = _set_xaxis(xaxis=xaxis, y=y)
        xlabel = xname

    else:
        #  Ensure 'by' has a valid input and assign x, xlabel accordingly
        if by in ('log-tpm', 'log-tpm-norm'):
            x = _normalize_expr(ica_data, ref_cols)
            xlabel = 'Mean Expression'
        elif by == 'length':
            x = np.log10(ica_data.gene_table.length)
            xlabel = 'Gene Length (log10-scale)'
        elif by == 'start':
            x = ica_data.gene_table.start
            xlabel = 'Gene Start'
        else:
            raise ValueError('"by" must be "log-tpm-norm", "length", '
                             'or "start"')

    # Override specific kwargs (their implementation is different
    # in this function)
    show_labels_pgw = kwargs.pop('show_labels', 'auto')
    adjust_labels_pgw = kwargs.pop('adjust_labels', True)
    legend_pgw = kwargs.pop('legend', True)
    legend_kwargs_pgw = kwargs.pop('legend_kwargs', None)

    kwargs['show_labels'] = kwargs['adjust_labels'] = kwargs['legend'] = False
    kwargs['legend_kwargs'] = None

    # Remove xlabel and ylabel kwargs if provided
    kwargs.pop('xlabel', None)
    kwargs.pop('ylabel', None)

    # Scatter Plot
    ax = scatterplot(x, y, xlabel=xlabel, ylabel=ylabel, **kwargs)

    # Add thresholds to scatter-plot (dashed lines)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    thresh = ica_data.thresholds[imodulon]
    if thresh != 0:
        ax.hlines([thresh, -thresh], xmin=xmin, xmax=xmax,
                  colors='k', linestyles='dashed', linewidth=1)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    bin_M = ica_data.M_binarized
    component_genes = set(bin_M[imodulon].loc[bin_M[imodulon] == 1].index)
    texts = []
    expand_kwargs = {'expand_objects': (1.2, 1.4),
                     'expand_points': (1.3, 1.3)}

    # Add labels: Put gene name if components contain under 20 genes
    if show_labels_pgw is True or (show_labels_pgw is not False
                                   and len(component_genes) <= 20):
        for gene in component_genes:
            texts.append(ax.text(x[gene], ica_data.M.loc[gene, imodulon],
                                 ica_data.gene_table.loc[gene, 'gene_name'],
                                 fontsize=12))

        expand_kwargs['expand_text'] = (1.4, 1.4)

    # Add labels: Repel texts from other text and points
    rect = ax.add_patch(Rectangle(xy=(xmin, -abs(thresh)),
                                  width=xmax - xmin,
                                  height=2 * abs(thresh),
                                  fill=False, linewidth=0))

    if adjust_labels_pgw:
        adjust_text(texts=texts, add_objects=[rect], ax=ax,
                    arrowprops=dict(arrowstyle='-', color='k', lw=0.5),
                    only_move={'objects': 'y'}, **expand_kwargs)

    # Add legend
    if legend_pgw and legend_kwargs_pgw:
        ax.legend(**legend_kwargs_pgw)

    return ax


def compare_gene_weights(ica_data: IcaData, imodulon1: ImodName,
                         imodulon2: ImodName, **kwargs) -> Ax:
    """
    Compare gene weights between 2 iModulons. The result is shown as a
    scatter-plot. Also shows the D'Agostino cutoff for both iModulons,
    and labels significantly enriched genes for both iModulons, if
    appropriate parameters are selected.

    Parameters
    ----------
    ica_data: pymodulon.core.IcaData
        IcaData container object
    imodulon1: int, str
        The name of the iModulon to plot on the x-axis
    imodulon2: int, str
        The name of the iModulon to plot on the y-axis
    **kwargs: dict
        keyword arguments passed onto `scatterplot()`

    Returns
    -------
    ax: matplotlib.axes instance
        Returns the axes instance on which the scatter-plot is generated
    """
    x = ica_data.M[imodulon1]
    y = ica_data.M[imodulon2]

    xlabel = f'{imodulon1} Gene Weight'
    ylabel = f'{imodulon2} Gene Weight'

    # Override specific kwargs (their implementation is different
    # in this function)
    show_labels_cgw = kwargs.pop('show_labels', 'auto')
    adjust_labels_cgw = kwargs.pop('adjust_labels', True)
    legend_cgw = kwargs.pop('legend', True)
    legend_kwargs_cgw = kwargs.pop('legend_kwargs', None)

    kwargs['show_labels'] = kwargs['adjust_labels'] = kwargs['legend'] = False
    kwargs['legend_kwargs'] = None

    # Remove xlabel and ylabel kwargs if provided
    kwargs.pop('xlabel', None)
    kwargs.pop('ylabel', None)

    # Scatter Plot
    ax = scatterplot(x, y, xlabel=xlabel, ylabel=ylabel, **kwargs)

    # Add thresholds to scatterplot (dashed lines)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    thresh1 = ica_data.thresholds[imodulon1]
    thresh2 = ica_data.thresholds[imodulon2]

    if thresh1 != 0:
        ax.vlines([thresh1, -thresh1], ymin=ymin, ymax=ymax,
                  colors='k', linestyles='dashed', linewidth=1)

    if thresh2 != 0:
        ax.hlines([thresh2, -thresh2], xmin=xmin, xmax=xmax,
                  colors='k', linestyles='dashed', linewidth=1)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Add labels on data-points
    bin_M = ica_data.M_binarized
    component_genes_x = set(bin_M[imodulon1].loc[bin_M[imodulon1] == 1].index)
    component_genes_y = set(bin_M[imodulon2].loc[bin_M[imodulon2] == 1].index)
    component_genes = component_genes_x.intersection(component_genes_y)
    texts = []
    expand_kwargs = {'expand_objects': (1.2, 1.4),
                     'expand_points': (1.3, 1.3)}

    # Add labels: Put gene name if components contain under 20 genes
    auto = None
    if show_labels_cgw == 'auto':
        auto = (bin_M[imodulon1].astype(bool)
                & bin_M[imodulon2].astype(bool)).sum() <= 20

    if show_labels_cgw or auto:
        for gene in component_genes:
            ax.scatter(ica_data.M.loc[gene, imodulon1],
                       ica_data.M.loc[gene, imodulon2],
                       color='r')
            texts.append(ax.text(ica_data.M.loc[gene, imodulon1],
                                 ica_data.M.loc[gene, imodulon2],
                                 ica_data.gene_table.loc[gene, 'gene_name'],
                                 fontsize=12))

        expand_kwargs['expand_text'] = (1.4, 1.4)

    # Add labels: Repel texts from other text and points
    rectx = ax.add_patch(Rectangle(xy=(xmin, -abs(thresh2)),
                                   width=xmax - xmin,
                                   height=2 * abs(thresh2),
                                   fill=False, linewidth=0))

    recty = ax.add_patch(Rectangle(xy=(-abs(thresh1), ymin),
                                   width=2 * abs(thresh1),
                                   height=ymax - ymin,
                                   fill=False, linewidth=0))

    if adjust_labels_cgw:
        adjust_text(texts=texts, add_objects=[rectx, recty], ax=ax,
                    arrowprops=dict(arrowstyle='-', color='k', lw=0.5),
                    only_move={'objects': 'y'}, **expand_kwargs)

    # Add legend
    if legend_cgw and legend_kwargs_cgw:
        ax.legend(**legend_kwargs_cgw)

    return ax


def compare_expression(ica_data: IcaData, gene1: str, gene2: str,
                       **kwargs) -> Ax:
    """
    Compares Gene Expression values between two genes. The result is shown
    as a scatter-plot.

    Parameters
    ----------
    ica_data: pymodulon.core.IcaData
        IcaData container object
    gene1: str
        Gene to plot on the x-axis
    gene2: str
        Gene to plot on the y-axis
    **kwargs: dict
        keyword arguments passed onto `scatterplot()`

    Returns
    -------
    ax: matplotlib.axes instance
        Returns the axes instance on which the scatter-plot is generated
    """

    # Check that gene1 exists
    if gene1 in ica_data.X.index:
        x = ica_data.X.loc[gene1]
        xlabel = f'{gene1} Expression'
    else:
        locus = name2num(ica_data, gene1)
        x = ica_data.X.loc[locus]
        xlabel = f'${gene1}$ Expression'

    # Check that gene2 exists
    if gene2 in ica_data.X.index:
        y = ica_data.X.loc[gene2]
        ylabel = f'{gene2} Expression'
    else:
        locus = name2num(ica_data, gene2)
        y = ica_data.X.loc[locus]
        ylabel = f'${gene2}$ Expression'

    # Remove xlabel, ylabel, and fit_line kwargs if provided
    kwargs.pop('xlabel', None)
    kwargs.pop('ylabel', None)
    kwargs.pop('fit_line', None)

    # Scatter Plot
    ax = scatterplot(x, y, xlabel=xlabel, ylabel=ylabel,
                     fit_line=True, **kwargs)

    return ax


def compare_activities(ica_data, imodulon1, imodulon2, **kwargs) -> Ax:
    """
    Compare activities between two iModulons.  The result is shown as a
    scatter-plot.

    Parameters
    ----------
    ica_data: pymodulon.core.IcaData
        IcaData container object
    imodulon1: int, str
        The name of the iModulon to plot on the x-axis
    imodulon2: int, str
        The name of the iModulon to plot on the y-axis
    **kwargs: dict
        keyword arguments passed onto `scatterplot()`

    Returns
    -------
    ax: matplotlib.axes instance
        Returns the axes instance on which the scatter-plot is generated
    """

    x = ica_data.A.loc[imodulon1]
    y = ica_data.A.loc[imodulon2]

    xlabel = f'{imodulon1} iModulon Activity'
    ylabel = f'{imodulon2} iModulon Activity'

    # Remove xlabel, ylabel, and fit_line kwargs if provided
    kwargs.pop('xlabel', None)
    kwargs.pop('ylabel', None)
    kwargs.pop('fit_line', None)

    # Scatter Plot
    ax = scatterplot(x, y, xlabel=xlabel, ylabel=ylabel,
                     fit_line=True, **kwargs)

    return ax


def plot_dima(ica_data_1: IcaData, project1: Optional[str],
              project2: Optional[str],
              sample1_list: Optional[list] = None,
              sample2_list: Optional[list] = None,
              lfc: float = 5, fdr_rate: float = .1, label: bool = True,
              adjust: bool = True, gene_table = False, **kwargs) -> Ax:
    """
    Plots a Dima plot between two projects or two sets of samples
    Args:
        ica_data_1: IcaData object that contains your data
        project1: Project and condition name in form "PROJECT__CONDITION" for
        condition one; Ex. "ica__wt_glc"
        project2: Project and condition name in form "PROJECT__CONDITION" for
        condition two; Ex. "ica__wt_glc"
        sample1_list: A list of samples you would like to analyze. The full
        name of the sample must be inputed; Ex. ["ica__wt_glc__1",
        "ica__wt_glc__2"]
        sample2_list: A list of samples you would like to analyze. The full
        name of the sample must be inputed; Ex. ["ica__wt_glc__1",
        "ica__wt_glc__2"]
        lfc: Log fold change that will serve as the cutoff for differentially
        expressed genes
        fdr_rate: False Detection Rate
        label: True/false option to label differentially expressed genes
        adjust: Additional true/false option to adjust labels
        gene_table: True/false option to display Pandas DataFrame of
        differentially expressed genes
        **kwargs: Additional arguments for scatterplot

    Returns: ax, Optional[diff_DF]

    """
    if sample1_list is None or sample2_list is None:
        for names, groups in \
                ica_data_1.sample_table.groupby(["project_id", "condition_id"]):
            if names[0] in project1 and names[1] in project1:
                sample1_list = list(groups.index)

            if names[0] in project2 and names[1] in project2:
                sample2_list = list(groups.index)

        a1 = ica_data_1.A[sample1_list].mean(axis=1)
        a2 = ica_data_1.A[sample2_list].mean(axis=1)

    df_diff = _diff_act(ica_data_1, sample1_list, sample2_list, lfc=lfc,
                        fdr_rate=fdr_rate)

    x_lims = max([abs(max(a1)), abs(min(a1))])
    y_lims = max([abs(max(a2)), abs(min(a2))])

    lims = max([x_lims, y_lims])

    ax = scatterplot(a1, a2, line45=True, line45_margin=lfc,
                     xlabel=re.search('(.*)__', sample1_list[0]).group(1),
                     ylabel=re.search('(.*)__', sample2_list[0]).group(1),
                     **kwargs)

    ax.set_xlim([-lims, lims])
    ax.set_ylim([-lims, lims])

    ax.hlines(0, -lims, lims, linewidth=0.5, color='gray', zorder=2)
    ax.vlines(0, -lims, lims, linewidth=0.5, color='gray', zorder=2)
    ax.plot([-lims, lims], [-lims, lims], color='k',
            linestyle='dashed', linewidth=0.5, zorder=0)
    ax.plot([max(-lims, -lims + lfc), min(lims, lims + lfc)],
            [max(-lims, -lims - lfc), min(lims, lims - lfc)],
            color='gray', linestyle='dashed', linewidth=0.5, zorder=0)
    ax.plot([max(-lims, -lims - lfc), min(lims, lims - lfc)],
            [max(-lims, -lims + lfc), min(lims, lims + lfc)],
            color='gray', linestyle='dashed', linewidth=0.5, zorder=0)

    if label:
        df_diff = pd.concat([df_diff, a1, a2], join='inner', axis=1)
        texts = []
        for k in df_diff.index:
            texts.append(ax.text(df_diff.loc[k, 0], df_diff.loc[k, 1], k,
                                 fontsize=10))
        if adjust:
            expand_args = {'expand_objects': (1.2, 1.4),
                           'expand_points': (1.3, 1.3),
                           'expand_text': (1.4, 1.4)}
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle="-", color='k', lw=0.5),
                        only_move={'objects': 'y'}, **expand_args)
    if gene_table:
        diff_genes = sorted(list(df_diff.index))
        non_diff_genes = []
        for i in range(0, len(ica_data_1.A.index)):
            if list(ica_data_1.A.index)[i] not in diff_genes:
                non_diff_genes.append(list(ica_data_1.A.index)[i])
        if len(diff_genes) - len(non_diff_genes) > 0:
            for i in range(0, (len(diff_genes) - len(non_diff_genes))):
                non_diff_genes.append(np.NaN)
        elif len(diff_genes) - len(non_diff_genes) < 0:
            for i in range(0, (len(non_diff_genes) - len(diff_genes))):
                diff_genes.append(np.NaN)
        compare_DF = pd.DataFrame([diff_genes, non_diff_genes],
                                  index=["diff_genes", "corr_genes"])
        return ax, compare_DF

    else:
        return ax


####################
# Helper Functions #
####################

def _fit_line(x, y, ax, metric):
    # Get line parameters and metric of correlation/regression
    if metric == 'r2':
        params, r2 = _get_fit(x, y)
        label = '$R^2_{{adj}}$ = {:.2f}'.format(r2)

    elif metric == 'pearson':
        params = curve_fit(_solid_line, x, y)[0]
        r, pval = stats.pearsonr(x, y)
        if pval < 1e-10:
            label = f'Pearson R = {r:.2f}\np-value < 1e-10'
        else:
            label = f'Pearson R = {r:.2f}\np-value = {pval:.2e}'

    elif metric == 'spearman':
        params = curve_fit(_solid_line, x, y)[0]
        r, pval = stats.spearmanr(x, y)
        if pval < 1e-10:
            label = f'Spearman R = {r:.2f}\np-value < 1e-10'
        else:
            label = f'Spearman R = {r:.2f}\np-value = {pval:.2e}'

    else:
        raise ValueError('Metric must be "pearson", "spearman", or "r2"')

    # Plot line
    if len(params) == 2:
        xvals = np.array([min(x), max(x)])
        ax.plot(xvals, _solid_line(xvals, *params), label=label,
                color='k', linestyle='dashed', linewidth=1, zorder=5)
    else:
        mid = params[2]
        xvals = np.array([x.min(), mid, x.max()])
        ax.plot(xvals, _broken_line(xvals, *params), label=label,
                color='k', linestyle='dashed', linewidth=1, zorder=5)


def _get_fit(x, y):
    all_params = [curve_fit(_solid_line, x, y)[0]]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=OptimizeWarning)
        for c in [min(x), np.mean(x), max(x)]:
            try:
                all_params.append(
                    curve_fit(_broken_line, x, y, p0=[1, 1, c])[0])
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
    y = (A * x + B)
    return y


def _adj_r2(f, x, y, params):
    n = len(x)
    k = len(params) - 1
    r2 = r2_score(y, f(x, *params))
    return 1 - np.true_divide((1 - r2) * (n - 1), (n - k - 1))


def _mod_freedman_diaconis(ica_data, imodulon):
    """
    Generates bins using optimal bin width estimate.

    This is done using a modified Freedman-Diaconis rule. The modification
    is necessary as iModulon gene-weights inherently contains many
    statistical `outliers`, which are the enriched genes of interest in
    the iModulon. For this reason, the interquartile range is not a
    sufficient bin width estimator by itself (it is too limited in its
    range). Thus, the full range of the dataset `x` is used instead of
    2*IQR. This strategy is generally fine as it leads to a better
    number of bins (< 20) and ensures that bin width continues to be
    proportional to n^-1/3 (where n is the number of samples in
    dataset `x`).

    See Also
    --------
    Wikipedia:
        {https://en.wikipedia.org/wiki/Freedman-Diaconis_rule}
    StackExchange:
        {https://tinyurl.com/pymodulonFreedmanDiaconis}
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
        xmax = (thresh + width)

    return np.arange(xmin, xmax + width, width)


def _normalize_expr(ica_data, ref_cols):
    x = ica_data.X

    if ref_cols:
        drop_cols = x[ref_cols].mean(axis=1)
        norm = x.sub(drop_cols, axis=0).drop(ref_cols, axis=1).mean(axis=1)
    else:
        norm = x.mean(axis=1)

    return norm


def _diff_act(ica_data: IcaData, sample1: List, sample2: List, lfc: float,
              fdr_rate: float):
    """

    Args:
        ica_data:
        sample1:
        sample2:
        lfc:
        fdr_rate:

    Returns:

    """
    _diff = pd.DataFrame()
    for name, group in ica_data.sample_table.groupby(
            ['project_id', 'condition_id']):
        for i1, i2 in combinations(group.index, 2):
            _diff['__'.join(name)] = abs(ica_data.A[i1] - ica_data.A[i2])
    dist = {}
    for k in ica_data.A.index:
        dist[k] = stats.lognorm(*stats.lognorm.fit(_diff.loc[k].values)).cdf

    res = pd.DataFrame(index=ica_data.A.index)
    for k in res.index:
        a1 = ica_data.A.loc[k, sample1].mean()
        a2 = ica_data.A.loc[k, sample2].mean()
        res.loc[k, 'LFC'] = a2 - a1
        res.loc[k, 'pvalue'] = 1 - dist[k](abs(a1 - a2))
    final = FDR(res, fdr_rate)
    return final[(abs(final.LFC) > lfc)].sort_values('LFC', ascending=False)


def FDR(p_values, fdr_rate, total=None):
    """

    Args:
        p_values:
        fdr_rate:
        total:

    Returns:

    """

    if total is not None:
        pvals = p_values.pvalue.values.tolist() + [1] * (total - len(p_values))
        idx = p_values.pvalue.index.tolist() + [None] * (total - len(p_values))
    else:
        pvals = p_values.pvalue.values
        idx = p_values.pvalue.index

    keep, qvals = fdrcorrection(pvals, alpha=fdr_rate)

    result = p_values.copy()
    result['qvalue'] = qvals[:len(p_values)]
    result = result[keep[:len(p_values)]]

    return result.sort_values('qvalue')


##########################
# Experimental Functions #
##########################

def _set_xaxis(xaxis: Union[Mapping, pd.Series], y: pd.Series):
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
        raise ValueError('Given x-values do not align with gene and their '
                         'respective gene-weights on the y-axis')

    return x
