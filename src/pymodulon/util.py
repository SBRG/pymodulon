"""
General utility functions for the pymodulon package
"""
import json
import logging
import re
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import digamma
from sklearn.neighbors import BallTree, KDTree

from pymodulon.enrichment import FDR


################
# Type Aliases #
################


def _check_table(table, name, index=None, index_col=0):
    # Set as empty dataframe if not input given
    if table is None:
        return pd.DataFrame(index=index)

    # Load table if necessary
    elif isinstance(table, str):
        try:
            table = pd.read_json(table)
        except ValueError:
            sep = "\t" if table.endswith(".tsv") else ","
            table = pd.read_csv(table, index_col=index_col, sep=sep)

    # Coerce indices and columns to ints if necessary
    newcols = []
    for col in table.columns:
        try:
            newcols.append(int(col))
        except ValueError:
            newcols.append(col)
    table.columns = newcols

    newrows = []
    for row in table.index:
        try:
            newrows.append(int(row))
        except ValueError:
            newrows.append(row)
    table.index = newrows

    # Replace empty strings with None
    table = table.replace("", np.nan)

    if isinstance(table, pd.DataFrame):
        # dont run _check_table_helper if no index is passed
        return table if index is None else _check_table_helper(table, index, name)
    else:
        raise TypeError(
            "{}_table must be a pandas DataFrame "
            "filename or a valid JSON string".format(name)
        )


def _check_table_helper(table, index, name):
    if table.shape == (0, 0):
        return pd.DataFrame(index=index)

    # Check if all indices are in table
    missing_index = list(set(index) - set(table.index))
    if len(missing_index) > 0:
        logging.warning(
            "Some {} are missing from the {} table: {}".format(
                name, name, missing_index
            )
        )

    # Remove extra indices from table
    table = table.reindex(index)
    return table


def _check_dict(table, index_col=0):
    try:
        table = json.loads(table.replace("'", '"'))
    except ValueError:
        sep = "\t" if table.endswith(".tsv") else ","
        table = pd.read_csv(table, index_col=index_col, header=None, sep=sep)
        table = table.to_dict()[1]
    return table


def compute_threshold(ic, dagostino_cutoff):
    """
    Computes D'agostino-test-based threshold for a component of an M matrix

    Parameters
    ----------
    ic: ~pandas.Series
        Pandas Series containing an independent component
    dagostino_cutoff: int
        Minimum D'agostino test statistic value to determine threshold

    Returns
    -------
    iModulon threshold: list
        List of thresholds for each iModulon
    """

    i = 0

    # Sort genes based on absolute value
    ordered_genes = abs(ic).sort_values()

    # Compute k2-statistic
    k_square, p = stats.normaltest(ic)

    # Iteratively remove gene w/ largest weight until k2-statistic < cutoff
    while k_square > dagostino_cutoff:
        i -= 1
        k_square, p = stats.normaltest(ic.loc[ordered_genes.index[:i]])

    # Select genes in iModulon
    comp_genes = ordered_genes.iloc[i:]

    # Slightly modify threshold to improve plotting visibility
    if len(comp_genes) == len(ic.index):
        return max(comp_genes) + 0.05
    else:
        return np.mean([ordered_genes.iloc[i], ordered_genes.iloc[i - 1]])


def dima(ica_data, sample1, sample2, threshold=5, fdr=0.1, alternate_A=None):
    """
    Creates DIMA table of differentially expressed iModulons

    Parameters
    ----------
    ica_data: ~pymodulon.core.IcaData
        :class:`~pymodulon.core.IcaData` data object
    sample1: str or list
        List of sample IDs or name of "project:condition"
    sample2: str or list
        List of sample IDs or name of "project:condition"
    threshold: float
        Minimum activity difference to determine DiMAs (default = 5)
    fdr: float
        False Detection Rate (default = .1)
    alternate_A: ~pandas.DataFrame
        Alternate A to use (default = None)

    Returns
    -------
    results: DataFrame
        Table of differentially expressed iModulons

    """

    # use the undocumented alternate_A option to allow custom-built DIMCA
    # activity matrix to be used in lieu of standard activty matrix
    if alternate_A is not None:
        A_to_use = alternate_A
    else:
        A_to_use = ica_data.A

    _diff = pd.DataFrame()

    sample1_list = _parse_sample(ica_data, sample1)
    sample2_list = _parse_sample(ica_data, sample2)

    for name, group in ica_data.sample_table.groupby(["project", "condition"]):
        for i1, i2 in combinations(group.index, 2):
            _diff[":".join(name)] = abs(A_to_use[i1] - A_to_use[i2])
    dist = {}

    for k in A_to_use.index:
        dist[k] = stats.lognorm(*stats.lognorm.fit(_diff.loc[k].values)).cdf

    res = pd.DataFrame(index=A_to_use.index)
    for k in res.index:
        a1 = A_to_use.loc[k, sample1_list].mean()
        a2 = A_to_use.loc[k, sample2_list].mean()
        res.loc[k, "difference"] = a2 - a1
        res.loc[k, "pvalue"] = 1 - dist[k](abs(a1 - a2))
    result = FDR(res, fdr)
    return result[(abs(result.difference) > threshold)].sort_values(
        "difference", ascending=False
    )


def _parse_sample(ica_data, sample):
    """
    Parses sample inputs into a list of sample IDs

    Parameters
    ----------
    ica_data: ~pymodulon.core.IcaData
        :class:`~pymodulon.core.IcaData` data object
    sample: list
        Sequence of sample IDs or "project:condition"

    Returns
    -------
    samples: list
        A list of `samples`
    """

    sample_table = ica_data.sample_table
    if isinstance(sample, str):
        proj, cond = re.search("(.*):(.*)", sample).groups()
        samples = sample_table[
            (sample_table.project == proj) & (sample_table.condition == cond)
        ].index
        if len(samples) == 0:
            raise ValueError(
                f"No samples exist for project={proj} condition=" f"{cond}"
            )
        else:
            return samples
    else:
        return sample


def explained_variance(
    ica_data, genes=None, samples=None, imodulons=None, reference=None
):
    """
    Computes the fraction of variance explained by iModulons (from 0 to 1)

    Parameters
    ----------
    ica_data: ~pymodulon.core.IcaData
        :class:`~pymodulon.core.IcaData` data object
    genes: str or list, optional
        List of genes to use (default: all genes)
    samples: str or list, optional
        List of samples to use (default: all samples)
    imodulons: int or str or list, optional
        List of iModulons to use (default: all iModulons)
    reference: list, optional
        List of samples that represent the reference condition for the
        set. If none are provided, uses the dataset-specific reference
        condition.

    Returns
    -------
    float
        Fraction of variance explained by selected iModulons for selected
        genes/samples
    """

    # Check inputs
    if genes is None:
        genes = ica_data.X.index
    elif isinstance(genes, str):
        genes = [genes]

    gene_loci = set(genes) & set(ica_data.X.index)
    gene_names = set(genes) - set(ica_data.X.index)
    name_loci = [ica_data.name2num(gene) for gene in gene_names]
    genes = list(set(gene_loci) | set(name_loci))

    if samples is None:
        samples = ica_data.X.columns
    elif isinstance(samples, str):
        samples = [samples]

    if imodulons is None:
        imodulons = ica_data.M.columns
    elif isinstance(imodulons, str) or isinstance(imodulons, int):
        imodulons = [imodulons]

    if reference is None:
        centered = ica_data.X
    else:
        centered = ica_data.X.subtract(ica_data.X[reference].mean(axis=1), axis=0)

    # Account for normalization procedures before ICA (X=SA-x_mean)
    baseline = centered.subtract(centered.mean(axis=0), axis=1)
    baseline = baseline.loc[genes, samples]

    # Initialize variables
    base_err = np.linalg.norm(baseline) ** 2
    MA = np.zeros(baseline.shape)
    rec_var = [0]
    ma_arrs = {}
    ma_weights = {}

    # Get individual modulon contributions
    for k in imodulons:
        ma_arr = np.dot(
            ica_data.M.loc[genes, k].values.reshape(len(genes), 1),
            ica_data.A.loc[k, samples].values.reshape(1, len(samples)),
        )
        ma_arrs[k] = ma_arr
        ma_weights[k] = np.sum(ma_arr ** 2)

    # Sum components in order of most important component first
    sorted_mods = sorted(ma_weights, key=ma_weights.get, reverse=True)
    # Compute reconstructed variance
    for k in sorted_mods:
        MA = MA + ma_arrs[k]
        sa_err = np.linalg.norm(MA - baseline) ** 2
        rec_var.append((1 - sa_err / base_err))

    return np.clip(rec_var[-1], 0, 1)


def infer_activities(ica_data, data):
    """
    Infer iModulon activities for external data

    Parameters
    ----------
    ica_data: ~pymodulon.core.IcaData
        :class:`~pymodulon.core.IcaData` data object
    data: ~pandas.DataFrame
        External expression profiles (must be centered to a reference)

    Returns
    -------
    new_activities: ~pandas.DataFrame
        Inferred activities for the expression profiles
    """

    shared_genes = ica_data.M.index & data.index
    x = data.loc[shared_genes].values
    m = ica_data.M.loc[shared_genes].values
    m_inv = np.linalg.pinv(m)
    a = np.dot(m_inv, x)
    return pd.DataFrame(a, index=ica_data.imodulon_names, columns=data.columns)


def mutual_info_distance(x, y):
    x = np.asarray(x).reshape(x.shape[0], 1)
    y = np.asarray(y).reshape(x.shape[0], 1)
    h = entropy(np.hstack([x, y]))
    if h == 0:
        return 1
    else:
        return 1 - mi(x, y) / h


# the following code is taken from the NPEET package; it cannot be installed
# via pip, so the necessary functions are copied here; the package appears to
# be un-maintained, so updates are not very likely; this is the GitHub page:
# https://github.com/gregversteeg/NPEET


def mi(x, y, z=None, k=3, base=2, alpha=0):
    """Mutual information of x and y (conditioned on z if z is not None)
    x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
    if x is a one-dimensional scalar and we have four samples
    """
    assert len(x) == len(y), "Arrays should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x, y = np.asarray(x), np.asarray(y)
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    x = add_noise(x)
    y = add_noise(y)
    points = [x, y]
    if z is not None:
        z = np.asarray(z)
        z = z.reshape(z.shape[0], -1)
        points.append(z)
    points = np.hstack(points)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = build_tree(points)
    dvec = query_neighbors(tree, points, k)
    if z is None:
        a, b, c, d = (
            avgdigamma(x, dvec),
            avgdigamma(y, dvec),
            digamma(k),
            digamma(len(x)),
        )
        if alpha > 0:
            d += lnc_correction(tree, points, k, alpha)
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = (
            avgdigamma(xz, dvec),
            avgdigamma(yz, dvec),
            avgdigamma(z, dvec),
            digamma(k),
        )
    return max(0, (-a - b + c + d) / np.log(base))


def entropy(x, k=3, base=2):
    """The classic K-L k-nearest neighbor continuous entropy estimator
    x should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
    if x is a one-dimensional scalar and we have four samples
    """
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x = np.asarray(x)
    n_elements, n_features = x.shape
    x = add_noise(x)
    tree = build_tree(x)
    nn = query_neighbors(tree, x, k)
    const = digamma(n_elements) - digamma(k) + n_features * np.log(2)
    return max(0, (const + n_features * np.log(nn).mean()) / np.log(base))


def add_noise(x, intens=1e-10):
    """Small noise to break degeneracy, see doc."""
    return x + intens * np.random.random_sample(x.shape)


def build_tree(points):
    if points.shape[1] >= 20:
        return BallTree(points, metric="chebyshev")
    return KDTree(points, metric="chebyshev")


def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1)[0][:, k]


def avgdigamma(points, dvec):
    """This part finds number of neighbors in some radius in the marginal space
    returns expectation value of <psi(nx)>"""
    tree = build_tree(points)
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    return np.mean(digamma(num_points))


def lnc_correction(tree, points, k, alpha):
    e = 0
    n_sample = points.shape[0]
    for point in points:
        # Find k-nearest neighbors in joint space, p=inf means max norm
        knn = tree.query(point[None, :], k=k + 1, return_distance=False)[0]
        knn_points = points[knn]
        # Substract mean of k-nearest neighbor points
        knn_points = knn_points - knn_points[0]
        # Calculate covariance matrix of k-nearest neighbor points, obtain
        # eigen vectors
        covr = knn_points.T @ knn_points / k
        _, v = np.linalg.eig(covr)
        # Calculate PCA-bounding box using eigen vectors
        V_rect = np.log(np.abs(knn_points @ v).max(axis=0)).sum()
        # Calculate the volume of original box
        log_knn_dist = np.log(np.abs(knn_points).max(axis=0)).sum()

        # Perform local non-uniformity checking and update correction term
        if V_rect < log_knn_dist + np.log(alpha):
            e += (log_knn_dist - V_rect) / n_sample
    return e


def count_neighbors(tree, x, r):
    return tree.query_radius(x, r, count_only=True)
