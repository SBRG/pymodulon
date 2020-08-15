import seaborn as sns
import matplotlib.pyplot as plt

from pymodulon.core import IcaData
from pymodulon.util import *

sns.set_style('whitegrid')

########################
# Component Gene Plots #
########################


def plot_samples_bar(ica_data: IcaData, imodulon: ImodName,
                     project: Optional[str] = None,
                     ax: Optional[Ax] = None,
                     figsize: Tuple[float, float] = (15, 2),
                     **legend_kwargs) -> Ax:
    """
    Generates plot of iModulon Activity levels, grouped by project name

    :param ica_data: iModulon Data container object
    :param imodulon: Name of iModulon
    :param project: Name of project (from metadata)
    :param ax: matplotlib Axes instance to output plot onto
    :param figsize: Size of output plot
    :param legend_kwargs: kwargs that get passed onto `ax.legend()`
    :return: Matplotlib Axes instance
    """

    # Check that iModulon exists
    if imodulon not in ica_data.imodulon_names:
        raise ValueError('Component does not exist: {}'.format(imodulon))

    # Create ax obj if None is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Get ymin and ymax
    ymin = ica_data.A.loc[imodulon].min()-3
    ymax = ica_data.A.loc[imodulon].max()+3

    # Check that metadata (sample_table) exists/is not empty
    sample_table = ica_data.sample_table
    if sample_table.empty():
        raise ValueError('Metadata does not exist, sample_table is empty')

    # Plot all projects not in the highlighted set
    other = sample_table[sample_table.project_name != project].copy()
    other.index.name = 'sample_id'
    other.reset_index(inplace=True)
    ax.bar(range(len(other)), ica_data.A.loc[imodulon, other['sample_id']],
           width=1, linewidth=0, align='edge', label='Previous Experiments')

    # Draw lines to discriminate between projects
    p_lines = other.project_name.drop_duplicates().index.tolist() \
        + [len(other), len(sample_table)]

    # Add project labels
    move = True
    locs = (np.array(p_lines)[:-1] + np.array(p_lines)[1:])/2

    for loc, name in zip(locs, other.project_name.drop_duplicates().tolist()
                         + [project]):
        ax.text(loc, ymax+2+move*4, name, fontsize=12,
                horizontalalignment='center')
        move = not move

    # Plot project of interest
    idx = len(other)
    for name, group in sample_table[sample_table.project_name
                                    == project].groupby('condition_name'):

        values = ica_data.A.loc[imodulon, group.index].values
        ax.bar(range(idx, idx+len(group)), values, width=1,
               linewidth=0, align='edge', label=name)
        idx += len(group)

    # Make legend
    kwargs = {'loc': 2, 'ncol': 7, 'bbox_to_anchor': (0, 0)}
    kwargs.update(legend_kwargs)
    ax.legend(**kwargs)

    # Prettify
    ax.set_xticklabels([])
    ax.grid(False, axis='x')
    ax.set_ylim([ymin, ymax])
    ax.set_xlim([0, ica_data.A.shape[1]])

    return ax
