import seaborn as sns
import matplotlib.pyplot as plt

from pymodulon.core import IcaData
from pymodulon.util import *

sns.set_style('whitegrid')

########################
# Component Gene Plots #
########################


def plot_samples_bar(ica_data: IcaData, imodulon: ImodName):
    """
    :param ica_data:
    :param imodulon: Name of iModulon
    """
    if imodulon not in ica_data.imodulon_names:
        raise ValueError('Component does not exist: {}'.format(imodulon))

    print(ica_data)  # placeholder
    print(imodulon)  # placeholder
