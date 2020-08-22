"""

"""
import numpy as np
import matplotlib.pyplot as plt

from pymodulon.core import IcaData
from typing import List, Union, Dict
from pymodulon.util import ImodName
from warnings import warn
from adjustText import adjust_text
import pandas as pd


########################
# Component Gene Plots #
########################

def bar_scatter_plot(values: pd.Series, sample_table: pd.DataFrame,
                     label: str, projects: Union[List, str],
                     highlight: Union[List, str], ax, legend_args):
    """
    Creates an overlaid scatter and barplot for a set of values (either gene
    expression levels or iModulon activities)

    Args:
        values: List of values to plot
        sample_table: Sample table from IcaData object
        label: Name of gene or iModulon
        projects: Project(s) to show
        highlight: Project(s) to highlight
        ax: Matplotlib axis object
        legend_args:

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

            if legend_args is not None:
                kwargs.update(legend_args)

            ax.legend(**kwargs)

    else:
        warn('Missing "project" and "condition" columns in sample table.')
        ax.bar(range(len(values)), values, width=1, align='edge')
        nbars = len(values)

    # Set axis limits
    xmin = -0.5
    xmax = nbars + 2.5
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    # Axis labels
    ax.set_ylabel(label, fontsize=12)
    ax.set_xticks([])

    # X-axis
    ax.hlines(0, xmin, xmax, color='k')

    return ax


def plot_expression(ica_data: IcaData, gene: str,
                    projects: Union[List, str] = None,
                    highlight: Union[List, str] = None,
                    ax=None, legend_args: Dict = None):
    """
    Creates a barplot showing an gene's expression across the compendium
    Args:
        ica_data: IcaData Object
        gene: Gene locus tag or name
        projects: Name(s) of projects to show (default: show all)
        highlight: Name(s) of projects to highlight (default: None)
        ax: Matplotlib axis object
        legend_args: Dictionary of arguments to be passed to legend

    Returns: A matplotlib axis object

    """
    # Check that gene exists
    if gene in ica_data.X.index:
        values = ica_data.X.loc[gene]
        label = '{} Expression'.format(gene)
    else:
        gene_table = ica_data.gene_table
        if 'gene_name' in gene_table.columns:
            loci = gene_table[gene_table.gene_name == gene].index

            # Ensure only one locus maps to this gene
            if len(loci) == 0:
                raise ValueError('Gene does not exist: {}'.format(gene))
            elif len(loci) > 1:
                warn('Found multiple genes named {}'.format(gene))

            # Get expression data
            values = ica_data.X.loc[loci[0]]

            # Italicize label
            label = '${}$ Expression'.format(gene)
        else:
            raise ValueError('Gene does not exist: {}'.format(gene))

    return bar_scatter_plot(values, ica_data.sample_table, label, projects,
                            highlight, ax, legend_args)


def plot_activities(ica_data: IcaData, imodulon: ImodName,
                    projects: Union[List, str] = None,
                    highlight: Union[List, str] = None,
                    ax=None, legend_args: Dict = None):
    """
    Creates a barplot showing an iModulon's activity across the compendium
    Args:
        ica_data: IcaData Object
        imodulon: iModulon name
        projects: Name(s) of projects to show (default: show all)
        highlight: Name(s) of projects to highlight (default: None)
        ax: Matplotlib axis object
        legend_args: Dictionary of arguments to be passed to legend

    Returns: A matplotlib axis object

    """
    # Check that iModulon exists
    if imodulon in ica_data.A.index:
        values = ica_data.A.loc[imodulon]
    else:
        raise ValueError('iModulon does not exist: {}'.format(imodulon))

    label = '{} iModulon\nActivity'.format(imodulon)

    return bar_scatter_plot(values, ica_data.sample_table, label, projects,
                            highlight, ax, legend_args)

#
#
# def plot_gene_weights(kind='scatter'):
#     """Plots iModulon gene weights
#
#     If the ica_data object contains an expression matrix (X), this will
#     produce a scatter plot between gene weights and a reference gene
#     expression set. If not, this will produce a histogram. This can be
#     manually selected using the `kind` argument.
#
#     Args:
#         kind: Kind of plot ('scatter' or 'histogram')
#
#
#
#     Returns:
#
#     """
#     pass
#
#
# def plot_dima():
#     pass
#
#
# def plot_deg():
#     pass
#
#
# def plot_scatter():
#     pass
#
#
# def plot_samples_bar(ica_data: IcaData,
#                      imodulon: ImodName = None,
#                      project: Optional[str] = None,
#                      ax=None,
#                      figsize: Tuple[float, float] = (15, 2),
#                      **legend_kwargs):
#     """Generates plot of iModulon Activity levels, grouped by project name
#
#     Args:
#         ica_data: iModulon Data container object
#         imodulon: Name of iModulon. Either an iModulon or gene must be
#         provided.
#         project:
#         ax:
#         figsize:
#         **legend_kwargs:
#
#     Returns:
#
#     """
#
#     """
#
#
#     :param ica_data:
#     :param imodulon:
#     :param gene: Name of gene. Either an iModulon or gene must be provided.
#     :param project: Name of project (from metadata)
#     :param ax: matplotlib Axes instance to output plot onto
#     :param figsize: Size of output plot
#     :param legend_kwargs: kwargs that get passed onto `ax.legend()`
#     :return: Matplotlib Axes instance
#     """
#
#     # Check that iModulon exists
#     if imodulon not in ica_data.imodulon_names:
#         raise ValueError('Component does not exist: {}'.format(imodulon))
#
#     # Create ax obj if None is provided
#     if ax is None:
#         fig, ax = plt.subplots(figsize=figsize)
#
#     # Get ymin and ymax
#     ymin = ica_data.A.loc[imodulon].min() - 3
#     ymax = ica_data.A.loc[imodulon].max() + 3
#
#     # Check that metadata (sample_table) exists/is not empty
#     sample_table = ica_data.sample_table
#     if sample_table.empty:
#         raise ValueError('Metadata does not exist, sample_table is empty')
#
#     # Plot all projects not in the highlighted set
#     other = sample_table[sample_table.project_name != project].copy()
#     other.index.name = 'sample_id'
#     other.reset_index(inplace=True)
#     ax.bar(range(len(other)), ica_data.A.loc[imodulon, other['sample_id']],
#            width=1, linewidth=0, align='edge', label='Previous Experiments')
#
#     # Draw lines to discriminate between projects
#     p_lines = other.project_name.drop_duplicates().index.tolist() \
#               + [len(other), len(sample_table)]
#
#     # Add project labels
#     move = True
#     locs = (np.array(p_lines)[:-1] + np.array(p_lines)[1:]) / 2
#
#     for loc, name in zip(locs, other.project_name.drop_duplicates().tolist()
#                                + [project]):
#         ax.text(loc, ymax + 2 + move * 4, name, fontsize=12,
#                 horizontalalignment='center')
#         move = not move
#
#     # Plot project of interest
#     idx = len(other)
#     for name, group in sample_table[sample_table.project_name
#                                     == project].groupby('condition_name'):
#         values = ica_data.A.loc[imodulon, group.index].values
#         ax.bar(range(idx, idx + len(group)), values, width=1,
#                linewidth=0, align='edge', label=name)
#         idx += len(group)
#
#     # Make legend
#     kwargs = {'loc': 2, 'ncol': 7, 'bbox_to_anchor': (0, 0)}
#     kwargs.update(legend_kwargs)
#     ax.legend(**kwargs)
#
#     # Prettify
#     ax.set_xticklabels([])
#     ax.grid(False, axis='x')
#     ax.set_ylim([ymin, ymax])
#     ax.set_xlim([0, ica_data.A.shape[1]])
#
#     return ax
