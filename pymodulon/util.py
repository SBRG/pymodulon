import numpy as np
from scipy import stats


def rename_imodulon(ica_data, old_name, new_name):
    """ Rename an iModulon """
    # Check that new names is not already in use
    if new_name in ica_data.get_names():
        raise ValueError('iModulon name ({:s}) already in use. Please choose a different name.'.format(new_name))
    if old_name not in ica_data.get_names():
        raise ValueError('No iModulon named {:s}'.format(old_name))
    name_list = [name if name != old_name else new_name for name in ica_data.get_names()]
    ica_data.set_imodulon_names(name_list)


def rename_sample(ica_data, old_name, new_name):
    """ Rename a sample """


def compute_threshold(k, s_matrix, dagostino_cutoff):
    """Computes D'agostino-test-based threshold for a component of an S matrix
        s_matrix: Component matrix with gene weights
        k: Component name
        dagostino_cutoff: Minimum D'agostino test statistic value to determine threshold
    """
    i = 0
    # Sort genes based on absolute value
    ordered_genes = abs(s_matrix[k]).sort_values()
    ksquare, p = stats.normaltest(s_matrix.loc[:, k])
    while ksquare > dagostino_cutoff:
        i -= 1
        # Check if K statistic is below cutoff
        ksquare, p = stats.normaltest(s_matrix.loc[ordered_genes.index[:i], k])
    comp_genes = ordered_genes.iloc[i:]
    if len(comp_genes) == len(s_matrix.index):
        return max(comp_genes) + .05
    else:
        return np.mean([ordered_genes.iloc[i], ordered_genes.iloc[i - 1]])