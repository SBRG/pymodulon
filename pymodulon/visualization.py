import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.axes._subplots import Subplot
from pymodulon.core import IcaData
from pymodulon.util import *

sns.set_style('whitegrid')

########################
# Component Gene Plots #
########################


def plot_samples_bar(ica_data: IcaData, imodulon: ImodName,
                     project: Optional[str] = None,
                     ax: Optional[Subplot] = None,
                     figsize: Tuple[float, float] = (15, 2)):
    """
    :param ica_data: iModulon Data container object
    :param imodulon: Name of iModulon
    :param project: Name of project (from metadata)
    :param ax: matplotlib Axes instance to output plot onto
    :param figsize: Size of output plot
    """

    # Check that iModulon exists
    if imodulon not in ica_data.imodulon_names:
        raise ValueError('Component does not exist: {}'.format(imodulon))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Get ymin and ymax
    ymin = ica_data.A.loc[imodulon].min()-3
    ymax = ica_data.A.loc[imodulon].max()+3

    # Plot all projects not in the highlighted set
    other = ica_data.sample_table[ica_data.sample_table.project_name
                                  != project].copy()
    other.index.name = 'sample_id'
    other.reset_index(inplace=True)
    ax.bar(range(len(other)), ica_data.A.loc[imodulon, other['sample_id']],
           width=1, linewidth=0, align='edge', label='Previous Experiments')

    # Draw lines to discriminate between projects
    p_lines = other.project_name.drop_duplicates().index.tolist() \
        + [len(other), len(ica_data.sample_table)]

    print(ica_data)  # placeholder, remove later
    print(imodulon)  # placeholder, remove later
