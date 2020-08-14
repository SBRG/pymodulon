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
                     ax: Optional[Subplot] = None,
                     figsize: Tuple[float, float] = (15, 2)):
    """
    :param ica_data: iModulon Data container object
    :param imodulon: Name of iModulon
    :param ax: matplotlib Axes instance to output plot onto
    :param figsize: Size of output plot
    """

    # Check that iModulon exists
    if imodulon not in ica_data.imodulon_names:
        raise ValueError('Component does not exist: {}'.format(imodulon))

    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)

    # Get ymin and ymax
    ymin = ica_data.A.loc[imodulon].min()-3
    ymax = ica_data.A.loc[imodulon].max()+3

    print(ica_data)  # placeholder, remove later
    print(imodulon)  # placeholder, remove later
