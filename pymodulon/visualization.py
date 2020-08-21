"""

"""
import numpy as np
import matplotlib.pyplot as plt

from pymodulon.core import IcaData
# from typing import Optional, Tuple
# from pymodulon.util import ImodName
from warnings import warn
from adjustText import adjust_text


########################
# Component Gene Plots #
########################

def barplot(ica_data, name, values, figsize, ax, legend_args):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Add project-specific information
    if 'project' in ica_data.sample_table.columns and \
            'condition' in ica_data.sample_table.columns:

        show_projects = True

        # Sort data by project/condition to ensure replicates are together
        metadata = ica_data.sample_table.sort_values(['project', 'condition'])
        values = values.reindex(metadata.index)

        # Get project names and sizes
        projects = metadata.project.drop_duplicates()
        project_sizes = [len(metadata[metadata.project == proj]) for proj in
                         projects]

    else:
        warn('Missing "project" and "condition" columns in sample table.')
        show_projects = False
        projects = None
        project_sizes = None

    # Plot values
    ax.bar(range(len(values)), values, width=1, linewidth=0, align='edge')

    # Get ymin and max
    ymax = max(1, values.max()) * 1.1
    ymin = min(-1, values.min()) * 1.1

    if show_projects:
        # Draw lines to discriminate between projects
        ax.vlines(np.cumsum(project_sizes)[:-1], ymin, ymax, colors='lightgray',
                  linewidth=1)

        # Add project names
        texts = []
        start = 0  # Offset above axis by just a bit
        for proj, size in zip(projects, project_sizes):
            x = start + size / 2
            texts.append(ax.text(x, 0, proj, ha='center'))
            start += size

        adjust_text(texts, avoid_points=False, avoid_self=False,
                    autoalign='y', expand_text=(1, 1),
                    only_move={'text': 'y', 'points': 'y', 'objects': 'y'})
        for text in texts:
            x, y = text.get_position()
            text.set_position((x, y + (ymax-ymin) * 1.1))

    # Set axis limits
    xmin = -0.01*len(values)
    xmax = len(values)*1.01
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    # Axis labels
    ax.set_ylabel('{} Expression'.format(name), fontsize=12)
    ax.set_xticks([])

    # X-axis
    ax.hlines(0, xmin, xmax, color='k')

    return ax


def plot_expression(ica_data: IcaData, gene: str,
                    figsize=(15, 2), ax=None, legend_args=None):
    # Check that gene exists
    if gene in ica_data.X.index:
        values = ica_data.X.loc[gene]
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
        else:
            raise ValueError('Gene does not exist: {}'.format(gene))

    return barplot(ica_data, gene, values, figsize, ax, legend_args)

#
# def plot_activities():
#     pass
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
