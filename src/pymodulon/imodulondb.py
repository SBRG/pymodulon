"""
Functions for writing a directory for iModulonDB webpages
"""

import os
import re
import sys
from itertools import chain
from typing import List, Optional, Set, Union
from zipfile import ZipFile

import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from tqdm.notebook import tqdm

from pymodulon.core import IcaData
from pymodulon.plotting import _broken_line, _get_fit, _solid_line


##################
# User Functions #
##################


def imodulondb_compatibility(model, inplace=False):
    """
    Checks if data has the correct data and labeling for output to iModulonDB.
    If it fails, it will print what went wrong, and there is an option (write =
    True) to correct issues.

    'inplace = True' will assume default values for everything that is needed
    EXCEPT for the project and condition columns in the sample_table,
    which must be filled in as part of metadata curation. If these columns
    are missing, alternatives that would work with the code are described in
    print statements, but these are not recommended because metadata curation
    is an important prerequisite to iModulon analysis.

    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object to check
    inplace : bool, optional
        Modify the data in-place (default = False)

    Returns
    -------
    bool
        Whether the iModulon object is compatible with iModulonDB
    """

    major_errors = []

    # imodulon table index
    try:
        model.imodulon_table.index.astype(int)
        im_idx = "int"
    except TypeError:
        im_idx = "str"

    # imodulon table columns
    im_table_cols = [
        "name",
        "TF",
        "Regulator",
        "Function",
        "Category",
        "n_genes",
        "precision",
        "recall",
    ]
    for col in im_table_cols:
        if not (col in model.imodulon_table.columns):
            print("iModulon Table is missing a %s column" % col)
            if col == "TF":
                print(
                    (
                        "Note that TF is used for coding purposes, "
                        "so it uses '+' and '/' and must match the trn."
                    )
                )
            if col == "Regulator":
                print(
                    "Note that Regulator is displayed in the dataset page "
                    "so it can have spelled out 'and' and 'or' operators."
                    "If you have a TF column, a Regulator column will be"
                    "generated for you (but not vice versa)."
                )

            if inplace:
                if col == "name":
                    if im_idx == "int":
                        model.imodulon_table["name"] = [
                            "iModulon %i" % i for i in model.imodulon_table.index
                        ]
                    else:
                        model.imodulon_table["name"] = model.imodulon_table.index
                elif col == "n_genes":
                    model.imodulon_table["n_genes"] = model.M_binarized.sum().astype(
                        int
                    )
                elif (col == "Regulator") and ("TF" in model.imodulon_table.columns):
                    for idx, tf in zip(
                        model.imodulon_table.index, model.imodulon_table.TF
                    ):
                        try:
                            model.imodulon_table.loc[idx, "Regulator"] = (
                                model.imodulon_table.TF[idx]
                                .replace("/", " or ")
                                .replace("+", " and ")
                            )
                        except AttributeError:
                            model.imodulon_table.loc[
                                idx, "Regulator"
                            ] = model.imodulon_table.TF[idx]
                else:
                    model.imodulon_table[col] = np.nan
    if inplace and (im_idx == "str"):
        model.rename_imodulons(
            dict(zip(model.imodulon_names, range(len(model.imodulon_names))))
        )

    print()
    # gene table columns
    gene_table_cols = [
        "gene_name",
        "gene_product",
        "COG",
        "start",
        "length",
        "operon",
        "regulator",
    ]
    for col in gene_table_cols:
        if not (col in model.gene_table.columns):
            if col in ["gene_name", "gene_product"]:
                print("Gene Table is missing the %s column" % col)
                if inplace:
                    model.gene_table[col] = model.gene_table.index
            else:
                print("Gene Table is missing the optional %s column" % col)
                if col == "regulator":
                    print("Add a TRN file to automatically generate this column.")

    print()
    # sample table columns
    sample_table_cols = [
        "sample",
        "project",
        "condition",
        "Biological Replicates",
        "DOI",
    ]
    for col in sample_table_cols:
        if not (col in model.sample_table.columns):
            if col == "sample":
                if not (model.sample_table.index.name == "sample"):
                    print("Sample Table is missing the optional sample column")
                    print(
                        "The 0th column will be used to define the names of "
                        "samples in the activity bar graph unless you add "
                        "this column."
                    )
            elif col == "DOI":
                print("Sample Table is missing the optional DOI column")
                print(
                    "If you would like to be able to access papers by clicking"
                    " the activity bars, add this column and populate it with "
                    "links."
                )
            elif col == "Biological Replicates":
                print("Sample Table is missing a %s column" % col)
                if inplace:
                    try:
                        for name, group in model.sample_table.groupby(
                            ["project", "condition"]
                        ):
                            model.sample_table.loc[
                                group.index, "Biological Replicates"
                            ] = group.shape[0]
                    except KeyError:
                        print(
                            "Unable to write Biological Replicates "
                            "column (add project & condition columns first)"
                        )
            else:
                print("Sample Table is missing a %s column" % col)
                print(
                    "This is an essential column - complete "
                    "metadata curation and ensure proper naming."
                )
                major_errors += [col]
                if col == "project":
                    print(
                        "The project column defines separate groups of "
                        "conditions,"
                        " which will have lines between them in activity bar "
                        "graphs."
                    )
                    print(
                        "You may add a column containing the same project "
                        "name for all"
                        " samples if desired, but it is recommended to "
                        "differentiate"
                        " them."
                    )
                if col == "condition":
                    print(
                        "The condition column defines biological replicate "
                        "conditions."
                        " You may reuse condition names such as 'control' "
                        "only if the "
                        "separate conditions are also in separate projects."
                    )
                    print(
                        "You may add a column containing all unique condition "
                        "names (e.g. integers) if desired, but it is"
                        " recommended to curate condition names."
                    )

    # X
    if model.X is None:
        major_errors += ["X"]
        print()
        print(
            "X, the expression matrix, is required for the gene pages. "
            "Add it to the data!"
        )

    if len(major_errors) > 0:
        out_str = (
            "ERROR: Cannot export to iModulonDB without "
            "fixing the following: " + ", ".join(major_errors)
        )
        sys.exit(out_str)


def imodulondb_export(
    model,
    path=".",
    skip_check=False,
    cat_order=None,
    gene_scatter_x="start",
):
    """
    Generates the iModulonDB page for the data and exports to the path.
    If certain columns are unavailable but can be filled in automatically,
    they will be.

    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object to export
    path : str, optional
        Path to iModulonDB main hosting folder (default = ".")
    skip_check : bool, optional
        If true, skip compatibility check (default = "False")
    cat_order : list, optional
        List of categories in the imodulon_table, ordered as you would
        like them to appear in the dataset table (default = None)
    gene_scatter_x : str
        Option to pass to scatter plot function, determines the X axis
        on iModulon pages. Currently, only "start" is supported. (default =
        "start")

    Returns
    -------
    None: None
    """
    model1 = model.copy()
    if not skip_check:
        imodulondb_compatibility(model1, True)

    print("Writing main site files...")

    folder = imodulondb_main_site_files(model1, path, cat_order=cat_order)

    print(
        "Two progress bars will appear below. The second will take "
        "significantly longer than the first."
    )

    imdb_generate_im_files(model1, folder, gene_scatter_x)
    imdb_generate_gene_files(model1, folder)


###############################
# Major Outputs (Called Once) #
###############################


def imdb_iM_table(imodulon_table, cat_order=None):
    """
    Reformats the iModulon table according

    Parameters
    ----------
    imodulon_table : ~pandas.DataFrame
        Table formatted similar to IcaData.imodulon_table
    cat_order : list
        List of categories in imodulon_table.Category, ordered as desired

    Returns
    -------
    im_table: ~pandas.DataFrame
        New iModulon table with the columns expected by iModulonDB
    """

    im_table = imodulon_table[
        ["name", "Regulator", "Function", "Category", "n_genes", "precision", "recall"]
    ]
    im_table = im_table.rename(columns={"name": "Name"})
    im_table.index.name = "k"
    im_table.Category = im_table.Category.fillna("Uncharacterized")

    if cat_order is not None:
        cat_dict = {cat_order[i]: i for i in range(len(cat_order))}
        im_table["category_num"] = [
            cat_dict[im_table.Category[k]] for k in im_table.index
        ]
    else:
        try:
            im_table["category_num"] = imodulon_table["new_idx"]
        except KeyError:
            im_table["category_num"] = im_table.index

    return im_table


def imdb_gene_presence(model):
    """
    Generates the two versions of the gene presence file, one as a binary
    matrix, and one as a DataFrame

    Parameters
    ----------
    model: :class:`~pymodulon.core.IcaData`
        An IcaData object

    Returns
    -------
    mbin: ~pandas.DataFrame
        Binarized M matrix
    mbin_list: list
        Table mapping genes to iModulons
    """
    mbin = model.M_binarized.astype(bool)
    mbin_list = pd.DataFrame(columns=["iModulon", "Gene"])
    for k in mbin.columns:
        for g in mbin.index[mbin[k]]:
            mbin_list = mbin_list.append({"iModulon": k, "Gene": g}, ignore_index=True)
    return mbin, mbin_list


def imodulondb_main_site_files(
    model, path_prefix=".", rewrite_annotations=True, cat_order=None
):
    """
    Generates all parts of the site that do not require large iteration loops

    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object
    path_prefix : str, optional
        Main folder for iModulonDB files (default = ".")
    rewrite_annotations : bool, optional
        Set to False if the gene_table and trn are unchanged (default = True)
    cat_order : list, optional
        list of categories in data.imodulon_table.Category, ordered as you want
        them to appear on the dataset page (default = None)

    Returns
    -------
    main_folder: str
        Dataset folder, for use as the path_prefix in imdb_generate_im_files()
    """

    organism = model.splash_table["organism_folder"]
    dataset = model.splash_table["dataset_folder"]

    # create new folders
    organism_folder = os.path.join(path_prefix, "organisms", organism)
    if not (os.path.isdir(organism_folder)):
        os.makedirs(organism_folder)

    annot_folder = os.path.join(organism_folder, "annotation")
    if not (os.path.isdir(annot_folder)):
        rewrite_annotations = True
        os.makedirs(annot_folder)

    # save annotations
    if rewrite_annotations:

        # make the folder if necessary
        gene_folder = os.path.join(annot_folder, "gene_files")
        if not (os.path.isdir(gene_folder)):
            os.makedirs(gene_folder)

        # add files to the folder
        model.gene_table.to_csv(os.path.join(gene_folder, "gene_info.csv"))
        try:
            model.trn.to_csv(os.path.join(gene_folder, "trn.csv"))
        except FileNotFoundError:
            pass

        # zip the folder
        old_cwd = os.getcwd()
        os.chdir(gene_folder)
        with ZipFile("../gene_files.zip", "w") as z:
            z.write("gene_info.csv")
            z.write("trn.csv")
        os.chdir(old_cwd)

    main_folder = os.path.join(organism_folder, dataset)
    if not (os.path.isdir(main_folder)):
        os.makedirs(main_folder)

        # save the metadata files in the main folder
    pd.Series(model.dataset_table).to_csv(os.path.join(main_folder, "dataset_meta.csv"))
    # num_ims - used so that the 'next iModulon' button doesn't overflow
    file = open(main_folder + "/num_ims.txt", "w")
    file.write(str(model.M.shape[1]))
    file.close()

    # save the dataset files in the data folder
    data_folder = os.path.join(main_folder, "data_files")
    if not (os.path.isdir(data_folder)):
        os.makedirs(data_folder)

    model.X.to_csv(os.path.join(data_folder, "log_tpm.csv"))
    model.A.to_csv(os.path.join(data_folder, "A.csv"))
    model.M.to_csv(os.path.join(data_folder, "M.csv"))
    im_table = imdb_iM_table(model.imodulon_table, cat_order)
    im_table.to_csv(os.path.join(data_folder, "iM_table.csv"))
    model.sample_table.to_csv(os.path.join(data_folder, "sample_table.csv"))
    mbin, mbin_list = imdb_gene_presence(model)
    mbin.to_csv(os.path.join(data_folder, "gene_presence_matrix.csv"))
    mbin_list.to_csv(os.path.join(data_folder, "gene_presence_list.csv"))
    pd.Series(model.thresholds).to_csv(os.path.join(data_folder, "M_thresholds.csv"))

    # zip the data folder
    old_cwd = os.getcwd()
    os.chdir(data_folder)
    with ZipFile("../data_files.zip", "w") as z:
        z.write("log_tpm.csv")
        z.write("A.csv")
        z.write("M.csv")
        z.write("iM_table.csv")
        z.write("sample_table.csv")
        z.write("gene_presence_list.csv")
        z.write("gene_presence_matrix.csv")
        z.write("M_thresholds.csv")
    os.chdir(old_cwd)

    # make iModulons searchable
    enrich_df = model.imodulon_table.copy()
    enrich_df["component"] = enrich_df.index
    enrich_df = enrich_df[["component", "name", "Regulator", "Function"]]
    try:
        enrich_df = enrich_df.sort_values(by="name").fillna(value="N/A")
    except TypeError:
        enrich_df["name"] = enrich_df["name"].astype(str)
        enrich_df = enrich_df.sort_values(by="name").fillna(value="N/A")
    if not (os.path.isdir(main_folder + "/iModulon_files")):
        os.makedirs(main_folder + "/iModulon_files")
    enrich_df.to_json(main_folder + "/iModulon_files/im_list.json", orient="records")

    # make genes searchable
    gene_df = model.gene_table.copy()
    gene_df = gene_df[gene_df.index.isin(model.X.index)]
    gene_df["gene_id"] = gene_df.index
    gene_df = gene_df[["gene_name", "gene_id", "gene_product"]]
    gene_df = gene_df.sort_values(by="gene_name").fillna(value="not available")
    if not (os.path.isdir(main_folder + "/gene_page_files")):
        os.makedirs(main_folder + "/gene_page_files")
    gene_df.to_json(main_folder + "/gene_page_files/gene_list.json", orient="records")

    # make the html
    html = '<div class="col-md-4 col-lg-3">\n'
    html += '\t<a role="button" class="btn btn-primary btn-outline-light ' "btn-block"
    html += ' d-flex flex-column private_set" href="dataset.html?organism='
    html += organism
    html += "&dataset="
    html += dataset
    html += '">\n\t\t<h2>'
    html += model.splash_table["large_title"]
    html += "</h2>\n\t\t<p>"
    html += model.splash_table["subtitle"]
    html += '</p>\n\t\t<p class="mt-auto">'
    html += model.splash_table["author"]
    html += "</p>\n\t</a>\n</div>"

    file = open(main_folder + "/html_for_splash.html", "w")
    file.write(html)
    file.close()

    return main_folder


def imdb_generate_im_files(model, path_prefix=".", gene_scatter_x="start"):
    """
    Generates all files for all iModulons in data

    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object
    path_prefix : str, optional
        Dataset folder in which to store the files (default = ".")
    gene_scatter_x : str
        Column from the gene table that specificies what to use on the
        X-axis of the gene scatter plot (default = "start")

    Returns
    -------
    None: None
    """

    for k in tqdm(model.imodulon_table.index):
        make_im_directory(model, k, path_prefix, gene_scatter_x)


def imdb_generate_gene_files(model, path_prefix="."):
    """
    Generates all files for all iModulons in IcaData object

    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object
    path_prefix : str, optional
        Dataset folder in which to store the files (default = ".")

    Returns
    -------
    None
    """

    for g in tqdm(model.M.index):
        make_gene_directory(model, g, path_prefix)


###################################################
# iModulon-Related Outputs (and Helper Functions) #
###################################################

# Gene Table


def _parse_tf_string(model, tf_str, print_output=True):
    """
    Returns a list of relevant tfs from a string. Will ignore TFs not in the
    trn file.
    iModulonDB helper function.

    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object
    tf_str : str
        String of tfs joined by '+' and '/' operators
    print_output : bool, optional
        Whether or nor to print outputs

    Returns
    -------
    tfs: list
        List of relevant TFs
    """

    if not (type(tf_str) == str):
        return []
    tf_str = tf_str.replace(" ", "").replace("[", "").replace("]", "")
    tfs = re.split("[+/]", tf_str)

    # Check if there is an issue, just remove the issues for now.
    bad_tfs = []
    for tf in tfs:
        if (tf not in model.trn.regulator.unique()) & print_output:
            print("TF not in TRN:", tf)
            print(
                "To remedy this, add rows to the TRN for each gene associated "
                "with this TF"
            )
            bad_tfs += [tf]
    tfs = list(set(tfs) - set(bad_tfs))

    return tfs


def imdb_gene_table_df(model, k):
    """
    Creates the gene table dataframe for iModulonDB
    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object
    k : int or str
        iModulon name

    Returns
    -------
    res: ~pandas.DataFrame
        DataFrame of the gene table that is compatible with iModulonDB
    """

    # get TFs and large table
    row = model.imodulon_table.loc[k]
    tfs = _parse_tf_string(model, row.TF, print_output=True)
    res = model.view_imodulon(k)

    # sort
    columns = []
    for c in ["gene_weight", "gene_name", "gene_product", "COG", "operon", "regulator"]:
        if c in res.columns:
            columns += [c]
    res = res[columns]
    res = res.sort_values("gene_weight", ascending=False)

    # add TFs
    for tf in tfs:
        reg_genes = model.trn.gene_id[model.trn.regulator == tf].values
        res[tf] = [i in reg_genes for i in res.index]

    # add links
    res["link"] = [model.gene_links[g] for g in res.index]

    # clean up
    res.index.name = "locus"

    return res


# Gene Histogram


def _component_DF(model, k, tfs=None):
    """
    Helper function for imdb_gene_hist_df

    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object
    k : int or str
        iModulon name
    tfs : list
        List of TFs (default = None)

    Returns
    -------
    gene_table: ~pandas.DataFrame
        Gene table for the iModulon
    """

    df = pd.DataFrame(model.M[k].sort_values())
    df.columns = ["gene_weight"]
    if "gene_product" in model.gene_table.columns:
        df["gene_product"] = model.gene_table["gene_product"]
    if "gene_name" in model.gene_table.columns:
        df["gene_name"] = model.gene_table["gene_name"]
    if "operon" in model.gene_table.columns:
        df["operon"] = model.gene_table["operon"]
    if "length" in model.gene_table.columns:
        df["length"] = model.gene_table.length
    if "regulator" in model.gene_table.columns:
        df["regulator"] = model.gene_table.regulator.fillna("")

    if tfs is not None:
        for tf in tfs:
            df[tf] = [tf in regs.split(",") for regs in df["regulator"]]

    return df.sort_values("gene_weight")


def _tf_combo_string(row):
    """
    Creates a formatted string for the histogram legends. Helper function for
    imdb_gene_hist_df.

    Parameters
    ----------
    row : ~pandas.Series
        Boolean series indexed by TFs for a given gene

    Returns
    -------
    str
        A string formatted for display (i.e. "Regulated by ...")
    """
    if row.sum() == 0:
        return "unreg"
    if row.sum() == 1:
        return row.index[row][0]
    if row.sum() == 2:
        return " and ".join(row.index[row])
    else:
        return ", ".join(row.index[row][:-1]) + ", and " + row.index[row][-1]


def _sort_tf_strings(tfs, unique_elts):
    """
    Sorts TF strings for the legend of the histogram. Helper function for
    imdb_gene_hist_df.

    Parameters
    ----------
    tfs : list[str]
        Sequence of TFs in the desired order
    unique_elts : list[str]
        All combination strings made by _tf_combo_string

    Returns
    -------
    list
        A sorted list of combination strings that have a consistent ordering
    """

    # unreg always goes first
    unique_elts.remove("unreg")
    sorted_elts = ["unreg"]

    # then the individual TFs
    for tf in tfs:
        if tf in unique_elts:
            sorted_elts += [tf]
            unique_elts.remove(tf)

    # then pairs
    pairs = [i for i in unique_elts if "," not in i]
    for i in tfs:
        for j in tfs:
            name = i + " and " + j
            if name in pairs:
                sorted_elts += [name]
                unique_elts.remove(name)

    # then longer combos, which won't be sorted for now
    return sorted_elts + unique_elts


def imdb_gene_hist_df(
    model,
    k: Union[int, str],
    bins: Optional[int] = 20,
    tol: Optional[float] = 0.001,
):
    """
    Creates the gene histogram for an iModulon

    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object
    k : int or str
        iModulon name
    bins : int, optional
        Number of bins in the histogram (default = 20)
    tol : float, optional
        Distance to threshold for deciding if a bar is in the iModulon
        (default = .001)

    Returns
    -------
    gene_hist_table: ~pandas.DataFrame
        A dataframe for producing the histogram that is compatible with
        iModulonDB
    """

    # get TFs
    row = model.imodulon_table.loc[k]
    if not (type(row.TF) == str):
        tfs = []
    else:
        tfs = _parse_tf_string(model, row.TF, print_output=False)

    # get genes
    DF_gene = _component_DF(model, k, tfs)

    # add a tf_combo column
    if len(tfs) == 0:
        DF_gene["tf_combos"] = ["unreg"] * DF_gene.shape[0]
    else:
        tf_bools = DF_gene[tfs]
        DF_gene["tf_combos"] = [
            _tf_combo_string(tf_bools.loc[g]) for g in tf_bools.index
        ]

    # get the list of tf combos in the correct order
    tf_combo_order = _sort_tf_strings(tfs, list(DF_gene.tf_combos.unique()))

    # compute bins
    xmin = min(min(DF_gene.gene_weight), -model.thresholds[k])
    xmax = max(max(DF_gene.gene_weight), model.thresholds[k])
    width = (
        2
        * model.thresholds[k]
        / (np.floor(2 * model.thresholds[k] * bins / (xmax - xmin) - 1))
    )
    xmin = -model.thresholds[k] - width * np.ceil((-model.thresholds[k] - xmin) / width)
    xmax = xmin + width * bins

    # column headers: bin middles
    columns = np.arange(xmin + width / 2, xmax + width / 2, width)[:bins]
    index = ["thresh"] + tf_combo_order + [i + "_genes" for i in tf_combo_order]
    res = pd.DataFrame(index=index, columns=columns)

    # row 0: threshold indices and number of unique tf combos
    thresh1 = -model.thresholds[k]
    thresh2 = model.thresholds[k]
    num_combos = len(tf_combo_order)
    res.loc["thresh"] = [thresh1, thresh2, num_combos] + [np.nan] * (len(columns) - 3)

    # next set of rows: heights of bars
    for r in tf_combo_order:
        res.loc[r] = np.histogram(
            DF_gene.gene_weight[DF_gene.tf_combos == r], bins, (xmin, xmax)
        )[0]

    # last set of rows: gene names
    for b_mid in columns:

        # get the bin bounds
        b_lower = b_mid - width / 2
        b_upper = b_lower + width
        for r in tf_combo_order:
            # get the genes for this regulator and bin
            genes = DF_gene.index[
                (DF_gene.tf_combos == r)
                & (DF_gene.gene_weight < b_upper)
                & (DF_gene.gene_weight > b_lower)
            ]
            # use the gene names, and get them with num2name (more robust)
            genes = [model.num2name(g) for g in genes]

            res.loc[r, b_mid] = len(genes)

            gene_list = np.array2string(np.array(genes), separator=" ")

            # don't list unregulated genes unless they are in the i-modulon
            if r == "unreg":
                if (b_lower + tol >= model.thresholds[k]) or (
                    b_upper - tol <= -model.thresholds[k]
                ):
                    res.loc[r + "_genes", b_mid] = gene_list
                else:
                    res.loc[r + "_genes", b_mid] = "[]"
            else:
                res.loc[r + "_genes", b_mid] = gene_list
    return res


# Gene Scatter Plot


def _gene_color_dict(model):
    """
    Helper function to match genes to colors based on COG. Used by
    imdb_gene_scatter_df.

    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object

    Returns
    -------
    dict
        Dictionary associating gene names to colors

    """

    try:
        gene_cogs = model.gene_table.COG.to_dict()
    except AttributeError:
        gene_cogs = {k: np.nan for k in model.gene_table.index}

    try:
        return {k: model.cog_colors[v] for k, v in gene_cogs.items()}
    except (KeyError, AttributeError):
        # previously, this would call the setter using:
        # data.cog_colors = None
        cogs = sorted(model.gene_table.COG.unique())
        model.cog_colors = dict(
            zip(
                cogs,
                [
                    "red",
                    "pink",
                    "y",
                    "orchid",
                    "mediumvioletred",
                    "green",
                    "lightgray",
                    "lightgreen",
                    "slategray",
                    "blue",
                    "saddlebrown",
                    "turquoise",
                    "lightskyblue",
                    "c",
                    "skyblue",
                    "lightblue",
                    "fuchsia",
                    "dodgerblue",
                    "lime",
                    "sandybrown",
                    "black",
                    "goldenrod",
                    "chocolate",
                    "orange",
                ],
            )
        )
        return {k: model.cog_colors[v] for k, v in gene_cogs.items()}


def imdb_gene_scatter_df(model, k, gene_scatter_x="start"):
    """
    Generates a dataframe for the gene scatter plot in iModulonDB

    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object
    k : int or str
        iModulon name
    gene_scatter_x : str
        Determines x-axis of the scatterplot

    Returns
    -------
    res: ~pandas.DataFrame
        A dataframe for producing the scatterplot
    """

    columns = ["name", "x", "y", "cog", "color", "link"]
    res = pd.DataFrame(columns=columns, index=model.M.index)
    res.index.name = "locus"

    cutoff = model.thresholds[k]

    # x&y scatterplot points - do alternatives later
    if gene_scatter_x == "start":
        try:
            res.x = model.gene_table.loc[res.index, "start"]
        except KeyError:
            gene_scatter_x = "gene number"
            res.x = range(len(res.index))
    else:
        raise ValueError("Only 'start' is supported as a gene_scatter_x input.")
    # res.x = data.X[base_conds].mean(axis=1)
    res.y = model.M[k]

    # add other data
    res.name = [model.num2name(i) for i in res.index]
    try:
        res.cog = model.gene_table.COG[res.index]
    except AttributeError:
        res.cog = "Unknown"

    gene_colors = _gene_color_dict(model)
    res.color = [to_hex(gene_colors[gene]) for gene in res.index]

    # if the gene is in the iModulon, it is clickable
    in_im = res.index[res.y.abs() > cutoff]
    for g in in_im:
        res.loc[g, "link"] = model.gene_links[g]

    # add a row to store the threshold
    cutoff_row = pd.DataFrame(
        [gene_scatter_x, cutoff] + [np.nan] * 4, columns=["meta"], index=columns
    ).T
    res = pd.concat([cutoff_row, res])

    return res


# Activity Bar Graph


def imdb_activity_bar_df(model, k):
    """
    Generates a dataframe for the activity bar graph of iModulon k

    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object
    k : int or str
        iModulon name

    Returns
    -------
    res: ~pandas.DataFrame
        A dataframe for producing the activity bar graph for iModulonDB
    """

    samp_table = model.sample_table.reset_index(drop=True)

    # get the row of A
    A_k = model.A.loc[k]
    A_k = A_k.rename(dict(zip(A_k.index, samp_table.index)))

    # initialize the dataframe
    max_replicates = int(samp_table["Biological Replicates"].max())
    columns = ["A_avg", "A_std", "n"] + list(
        chain(*[["rep%i_idx" % i, "rep%i_A" % i] for i in range(1, max_replicates + 1)])
    )
    res = pd.DataFrame(columns=columns)

    # iterate through conditions and fill in rows
    for cond, group in samp_table.groupby(["project", "condition"], sort=False):

        # get condition name and A values
        cond_name = cond[0] + "__" + cond[1]  # project__cond
        vals = A_k[group.index]

        # compute statistics
        new_row = [vals.mean(), vals.std(), len(vals)]

        # fill in individual samples (indices and values)
        for idx in group.index:
            new_row += [idx, vals[idx]]
        new_row += [np.nan] * ((max_replicates - len(vals)) * 2)

        res.loc[cond_name] = new_row

    # clean up
    res.index.name = "condition"
    res = res.reset_index()

    return res


# Regulon Venn Diagram


def _parse_regulon_string(model, s):
    """
    The Bacillus microarray dataset uses [] to create unusually complicated
    TF strings.
    This function parses those, as a helper to _get_reg_genes for
    imdb_regulon_venn_df.

    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object
    s : str
        TF string

    Returns
    -------
    res: set
        Set of genes regulated by this string
    """

    res = set()
    if not (isinstance(s, str)):
        return res
    if "/" in s:
        union = s.split("] / [")
        union[0] = union[0][1:]
        union[-1] = union[-1][:-1]
    else:
        union = [s]
    for r in union:
        if "+" in r:
            intersection = r.split(" + ")
            genes = set(model.trn.gene_id[model.trn.regulator == intersection[0]])
            for i in intersection[1:]:
                genes = genes.intersection(
                    set(model.trn.gene_id[model.trn.regulator == i])
                )
        else:
            genes = set(model.trn.gene_id[model.trn.regulator == r])
        res = res.union(genes)
    return res


def _get_reg_genes(model: IcaData, tf: str) -> Set:
    """
    Finds the set of genes regulated by the boolean combination of regulators
    in a TF
    string

    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object
    tf : str
        string of TFs separated by +, /, and/or []

    Returns
    -------
    reg_genes: set
        Set of regulated genes
    """

    # the Bacillus tf strings use '[]' to make complicated boolean combinations
    if "[" in tf:
        reg_genes = _parse_regulon_string(model, tf)

    # other datasets can use this simpler code
    else:
        tf = tf.replace(" ", "")
        if "+" in tf:
            reg_list = []
            for tfx in tf.split("+"):
                reg_list.append(
                    set(model.trn[model.trn.regulator == tfx].gene_id.unique())
                )
            reg_genes = set.intersection(*reg_list)
        elif "/" in tf:
            reg_genes = set(
                model.trn[model.trn.regulator.isin(tf.split("/"))].gene_id.unique()
            )
        else:
            reg_genes = set(model.trn[model.trn.regulator == tf].gene_id.unique())

    # return result
    return reg_genes


def imdb_regulon_venn_df(model: IcaData, k: Union[str, int]):
    """
    Generates a dataframe for the regulon venn diagram of iModulon k. Returns
    None
    if there is no diagram to draw
    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object
    k : int or str
        iModulon name

    Returns
    -------
    res: ~pandas.DataFrame
        A DataFrame for producing the venn diagram in iModulonDB
    """

    row = model.imodulon_table.loc[k]
    tf = row["TF"]

    if not (type(tf) == str):
        return None

    # Take care of and/or enrichments
    reg_genes = _get_reg_genes(model, tf)

    # Get component genes
    comp_genes = set(model.view_imodulon(k).index)
    both_genes = set(reg_genes & comp_genes)

    # Get gene and operon counts
    reg_gene_count = len(reg_genes)
    comp_gene_count = len(comp_genes)
    both_gene_count = len(both_genes)

    # Add adjustments for venn plotting (add '2' for alternates)
    reg_gene_count2 = 0
    comp_gene_count2 = 0
    both_gene_count2 = 0
    if reg_genes == comp_genes:
        reg_gene_count = 0
        comp_gene_count = 0
        both_gene_count = 0
        reg_gene_count2 = 0
        comp_gene_count2 = 0
        both_gene_count2 = len(reg_genes)
    elif all(item in comp_genes for item in reg_genes):
        reg_gene_count = 0
        both_gene_count = 0
        reg_gene_count2 = len(reg_genes)
        comp_gene_count2 = 0
        both_gene_count2 = 0
    elif all(item in reg_genes for item in comp_genes):
        comp_gene_count = 0
        both_gene_count = 0
        reg_gene_count2 = 0
        comp_gene_count2 = len(comp_genes)
        both_gene_count2 = 0

    res = pd.DataFrame(
        [
            tf,
            reg_gene_count,
            comp_gene_count,
            both_gene_count,
            reg_gene_count2,
            comp_gene_count2,
            both_gene_count2,
        ],
        columns=["Value"],
        index=[
            "TF",
            "reg_genes",
            "comp_genes",
            "both_genes",
            "reg_genes2",
            "comp_genes2",
            "both_genes2",
        ],
    )

    # gene lists
    just_reg = reg_genes - both_genes
    just_comp = comp_genes - both_genes
    for i, l in zip(
        ["reg_genes", "comp_genes", "both_genes"], [just_reg, just_comp, both_genes]
    ):
        gene_list = np.array([model.num2name(g) for g in l])
        gene_list = np.array2string(gene_list, separator=" ")
        res.loc[i, "list"] = gene_list

    return res


# Regulon Scatter Plot


def _get_tfs_to_scatter(model: IcaData, tf_string: Union[str, float]) -> List:
    """

    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object
    tf_string : str or ~numpy.nan
        String of TFs, or np.nan
    Returns
    -------
    res: list
        List of gene loci
    """

    # hard-coded TF names
    # should just modify TRN/gene info so everything matches but ok
    rename_tfs = {
        "csqR": "yihW",
        "hprR": "yedW",
        "thi-box": "Thi-box",
        "flhD;flhC": "flhD",
        "rcsA;rcsB": "rcsA",
        "ntrC": "glnG",
        "gutR": "srlR",
    }
    res = []
    if type(tf_string) == str:

        tf_string = tf_string.replace(" ", "").replace("[", "").replace("]", "")

        tfs = re.split("[+/]", tf_string)

        for tf in tfs:

            if tf in rename_tfs.keys():
                tf = rename_tfs[tf]

            try:
                b_num = model.name2num(tf)
                if b_num in model.X.index:
                    res += [tf]
            except ValueError:
                print("TF has no associated expression profile:", tf)
                print("If %s is not a gene, this behavior is expected." % tf)
                print(
                    "If it is a gene, use consistent naming"
                    " between the TRN and gene_table."
                )

    res = list(set(res))  # remove duplicates
    return res


def imdb_regulon_scatter_df(model: IcaData, k: Union[str, int]):
    """

    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object
    k : int or str
        iModulon name

    Returns
    -------
    res: ~pandas.DataFrame
        A dataframe for producing the regulon scatter plots in iModulonDB
    """

    row = model.imodulon_table.loc[k]
    tfs = _get_tfs_to_scatter(model, row.TF)

    if len(tfs) == 0:
        return None

    # coordinates for points
    coord = pd.DataFrame(columns=["A"] + tfs, index=model.A.columns)
    coord["A"] = model.A.loc[k]

    # params for fit line
    param_df = pd.DataFrame(
        columns=["A"] + tfs, index=["R2", "xmin", "xmid", "xmax", "ystart", "yend"]
    )

    # fill in dfs
    for tf in tfs:

        # coordinates
        coord[tf] = model.X.loc[model.name2num(tf)]
        xlim = np.array([coord[tf].min(), coord[tf].max()])
        # fit line
        params, r2 = _get_fit(coord[tf], coord["A"])
        if len(params) == 2:  # unbroken
            y = _solid_line(xlim, *params)
            out = [xlim[0], np.nan, xlim[1], y[0], y[1]]
        else:  # broken
            xvals = np.array([xlim[0], params[2], xlim[1]])
            y = _broken_line(xvals, *params)
            out = [xlim[0], params[2], xlim[1], y[0], y[2]]

        param_df[tf] = [r2] + out

    res = pd.concat([param_df, coord], axis=0)
    res = res.sort_values("R2", axis=1, ascending=False)
    res = res[pd.Index(["A"]).append(res.columns.drop("A"))]

    return res


# iModulon Metadata
def _tf_with_links(model: IcaData, tf_str: Union[str, float]):
    """
    Adds links to the regulator string
    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object
    tf_str : str or float
        Regulator string for a given iModulon, or np.nan

    Returns
    -------
    res: str
        String with links added
    """

    tf_links = model.tf_links

    if not (type(tf_str) == str):
        return tf_str

    # get a list of transcription factors
    and_or = ""
    if "/" in tf_str:
        and_or = " or "
        tfs = tf_str.split("/")
    elif "+" in tf_str:
        and_or = " and "
        tfs = tf_str.split("+")
    else:
        tfs = [tf_str]

    # start building an html string
    tfs_html = []
    for tf in tfs:
        if tf in tf_links.keys():
            link = tf_links[tf]
            if type(link) == str:  # this tf has a link
                tf_ = '<a href="' + link + '" target="_blank">' + tf + "</a>"
                tfs_html += [tf_]
            else:  # this tf has no link
                tfs_html += [tf]
        # this tf isn't in the tf_links file
        else:
            tfs_html += [tf]
    res = and_or.join(tfs_html)
    return res


def imdb_imodulon_basics_df(
    model,
    k,
    reg_venn,
    reg_scatter,
):
    """
    Generates a dataframe for the metadata of iModulon k

    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object
    k : int or str
        iModulon name
    reg_venn : ~pandas.DataFrame or None
        Output of imdb_regulon_venn_df(data, k)
    reg_scatter : ~pandas.DataFrame or None
        Output of imdb_regulon_scatter_df(data, k)

    Returns
    -------
    res: ~pandas.DataFrame
        A dataframe of metadata for iModulon k in iModulonDB
    """

    row = model.imodulon_table.loc[k]

    res = pd.Series(
        index=[
            "name",
            "TF",
            "Regulator",
            "Function",
            "Category",
            "has_venn",
            "scatter",
            "has_meme",
            "precision",
            "recall",
        ],
        dtype=float,
    )
    res.loc["name"] = row.loc["name"]
    res.loc["TF"] = row.TF
    res.loc["Regulator"] = _tf_with_links(model, row.TF)
    res.loc["Function"] = row.Function
    res.loc["Category"] = row.Category
    res.loc["has_venn"] = not (reg_venn is None)
    if reg_scatter is None:
        res.loc["scatter"] = 0
    else:
        res.loc["scatter"] = reg_scatter.shape[1] - 1
    res.loc["has_meme"] = False  # memes will be added to the website later.

    res.loc["precision"] = row.precision
    res.loc["recall"] = row.recall

    return res


# Compute All iModulon Plots
def make_im_directory(model, k, path_prefix=".", gene_scatter_x="start"):
    """

    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object
    k : int or str
        iModulon name
    path_prefix : str, optional
        Path to the dataset folder. This function creates an 'iModulon_files/k/'
        subdirectory there to store everything. (default = ".")
    gene_scatter_x : str
        Passed to imdb_gene_scatter_df() to indicate
        the x axis type of that plot (default = "start")

    Returns
    -------
    None: None
    """

    # generate the plot files
    gene_table = imdb_gene_table_df(model, k)
    gene_hist = imdb_gene_hist_df(model, k)
    gene_scatter = imdb_gene_scatter_df(model, k, gene_scatter_x)
    act_bar = imdb_activity_bar_df(model, k)
    reg_venn = imdb_regulon_venn_df(model, k)
    reg_scatter = imdb_regulon_scatter_df(model, k)

    # generate a basic data df
    res = imdb_imodulon_basics_df(model, k, reg_venn, reg_scatter)

    # save output
    folder = path_prefix + "/iModulon_files/" + str(k) + "/"
    if not (os.path.isdir(path_prefix + "/iModulon_files")):
        os.makedirs(path_prefix + "/iModulon_files")
    if not (os.path.isdir(folder)):
        os.makedirs(folder)
    res.to_csv(folder + str(k) + "_meta.csv")
    gene_table.to_csv(folder + str(k) + "_gene_table.csv")
    gene_hist.to_csv(folder + str(k) + "_gene_hist.csv")
    gene_scatter.to_csv(folder + str(k) + "_gene_scatter.csv")
    act_bar.to_csv(folder + str(k) + "_activity_bar.csv")
    if not (reg_venn is None):
        reg_venn.to_csv(folder + str(k) + "_reg_venn.csv")
    if not (reg_scatter is None):
        reg_scatter.to_csv(folder + str(k) + "_reg_scatter.csv")
    model.M[k].to_csv(folder + str(k) + "_gene_weights.csv")
    model.A.loc[k].to_csv(folder + str(k) + "_activity.csv")


###############################################
# Gene-Related Outputs (and Helper Functions) #
###############################################

# Activity bar graph


def imdb_gene_activity_bar_df(model, gene_id):
    """


    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object
    gene_id : str
        Locus tag of gene

    Returns
    -------
    res: ~pandas.DataFrame
        A dataframe for the activity bar of gene in iModulonDB
    """

    # get the row of A
    X_gene_id = model.X.loc[gene_id]

    # initialize the dataframe
    samp_table = model.sample_table.reset_index(drop=True)
    X_gene_id = X_gene_id.rename(dict(zip(X_gene_id.index, samp_table.index)))
    max_replicates = int(samp_table["Biological Replicates"].max())
    columns = ["X_avg", "X_std", "n"] + list(
        chain(*[["rep%i_idx" % i, "rep%i_X" % i] for i in range(1, max_replicates + 1)])
    )
    res = pd.DataFrame(columns=columns)

    # iterate through conditions and fill in rows
    for cond, group in samp_table.groupby(["project", "condition"], sort=False):

        # get condition name and X values
        cond_name = cond[0] + "__" + cond[1]  # project__cond
        vals = X_gene_id[group.index]

        # compute statistics
        new_row = [vals.mean(), vals.std(), len(vals)]

        # fill in individual samples (indices and values)
        for idx in group.index:
            new_row += [idx, vals[idx]]
        new_row += [np.nan] * ((max_replicates - len(vals)) * 2)

        res.loc[cond_name] = new_row

    # clean up
    res.index.name = "condition"
    res = res.reset_index()

    return res


# iModulon Table


def imdb_gene_im_table_df(model, g, im_table, m_bin):
    """
    Generates a dataframe for the iModulon table of gene g

    Parameters
    ----------
    model: :class:`~pymodulon.core.IcaData`
        IcaData object
    g : str
        Gene locus tag
    im_table : ~pandas.DataFrame
        Pre-cleaned version of data.imodulon_table
    m_bin : ~pandas.DataFrame
        Boolean transpose version of data.M_binarized

    Returns
    -------
    perGene_table: ~pandas.DataFrame
        A dataframe for the iModulon table of gene g in iModulonDB
    """

    perGene_table = im_table.copy()
    perGene_table.insert(column="in_iM", value=m_bin.loc[:, g], loc=1)
    perGene_table.insert(column="gene_weight", value=model.M.loc[g, :], loc=2)

    perGene_table = (
        perGene_table.assign(A=perGene_table["gene_weight"].abs())
        .sort_values(["in_iM", "A"], ascending=[False, False])
        .drop("A", 1)
    )
    return perGene_table


# Gene Metadata


def imdb_gene_basics_df(model, g):
    """

    Parameters
    ----------
    model: :class:`~pymodulon.core.IcaData`
        IcaData object
    g : str
        Gene locus

    Returns
    -------
    res: ~pandas.DataFrame
        A dataframe for the metadata of gene g in iModulonDB
    """

    row = model.gene_table.loc[g]
    res = pd.Series(
        index=["gene_id", "name", "operon", "gene_product", "COG", "regulator", "link"],
        dtype=float,
    )
    res.loc["gene_id"] = g
    res.loc["name"] = row.gene_name

    for elt in ["gene_product", "COG", "operon", "regulator"]:
        try:
            res.loc[elt] = row[elt]
        except KeyError:
            res.loc[elt] = np.nan

    if type(model.gene_links[g]) == str:
        res.loc["link"] = (
            '<a href="' + str(model.gene_links[g]) + '">' + model.link_database + "</a>"
        )
    else:
        res.loc["link"] = np.nan

    res.fillna(value="<i>Not Available</i>", inplace=True)
    return res


# Compute All Gene Data


def make_gene_directory(model, g, path_prefix="."):
    """
    Generates all data for gene g, stores it in a subfolder of path_prefix

    Parameters
    ----------
    model : :class:`~pymodulon.core.IcaData`
        IcaData object
    g : str
        Gene locus
    path_prefix : str, optional
        Path to the dataset folder. This function creates
        a 'gene_page_files/k/' subdirectory there to store everything.
        (default = ".")

    Returns
    -------
    im_df: ~pandas.DataFrame
        Table containing iModulon information for the gene
    """

    im_table_short = model.imodulon_table[["name", "Regulator", "Function", "Category"]]
    im_table_short = im_table_short.rename(columns={"name": "Name"})
    im_table_short.index.name = "k"
    m_bin = model.M_binarized.astype(bool).T

    act_df = imdb_gene_activity_bar_df(model, g)
    im_df = imdb_gene_im_table_df(model, g, im_table_short, m_bin)
    g_df = imdb_gene_basics_df(model, g)

    folder = path_prefix + "/gene_page_files/" + str(g) + "/"
    if not (os.path.isdir(folder)):
        os.makedirs(folder)

    g_df.to_csv(folder + str(g) + "_meta.csv", header=True)
    act_df.to_csv(folder + str(g) + "_activity_bar.csv")
    model.X.loc[g].to_csv(folder + str(g) + "_expression.csv")
    im_df.to_csv(folder + str(g) + "_perGene_table.csv")

    return im_df
