import os
import subprocess
import warnings
from glob import glob
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from Bio import SeqIO
from graphviz import Digraph


def _get_orthologous_imodulons(
    M1: pd.DataFrame, M2: pd.DataFrame, method: Union[str, Callable], cutoff: float
):
    """
    Given two M matrices, returns the dot graph and name links of the various
    connected ICA components

    Parameters
    ----------
    M1 : pd.DataFrame
        M matrix from the first organism
    M2 : pd.DataFrame
        M matrix from the second organism
    method : str
        Correlation metric to use (see pd.DataFrame.corr)
    cutoff : float
        Cut off value for correlation metric

    Returns
    -------
    links
        Links and distances of connected iModulons
    """

    # Only keep genes found in both M matrices
    common = set(M1.index) & set(M2.index)

    if len(common) == 0:
        raise KeyError("No common genes")

    m1 = M1.reindex(common)
    m2 = M2.reindex(common)

    # Compute correlation matrix
    corr = (
        pd.concat([m1, m2], axis=1, keys=["df1", "df2"])
        .corr(method=method)
        .loc["df1", "df2"]
        .abs()
    )
    DF_corr = corr.loc[m1.columns, m2.columns]

    # Get positions where correlation is above cutoff
    pos = zip(*np.where(DF_corr > cutoff))
    links = [(m1.columns[i], m2.columns[j], DF_corr.iloc[i, j]) for i, j in pos]

    return links


def _make_dot_graph(
    links: Sequence,
    show_all: bool = True,
    names1: Optional[Sequence] = None,
    names2: Optional[Sequence] = None,
):
    """
    Given two M matrices, returns the dot graph and name links of the various
    connected ICA components

    Parameters
    ----------
    links : Sequence
        Names and distances of connected iModulons
    show_all : bool
        Show all iModulons regardless of their linkage (default: False)
    names1 : Sequence
        List of names in dataset 1 (required if show_all = True)
    names2 : Sequence
        List of names in dataset 1 (required if show_all = True)

    Returns
    -------
    dot
        Dot graph of connected iModulons
    links
        Links and distances of connected iModulons
    """

    link_names1 = [link[0] for link in links]
    link_names2 = [link[1] for link in links]

    if not show_all:
        # Get names of nodes
        names1 = link_names1
        names2 = link_names2

    # Split names in half if necessary for dataset1
    name_dict1 = {}
    for name in names1:
        val = str(name)
        if len(val) > 10:
            name_dict1[name] = val[: len(val) // 2] + "-\n" + val[len(val) // 2 :]
        else:
            name_dict1[name] = val

    # Split names in half if necessary for dataset2
    name_dict2 = {}
    for name in names2:
        val = str(name)
        if len(val) > 10:
            name_dict2[name] = val[: len(val) // 2] + "-\n" + val[len(val) // 2 :]
        else:
            name_dict2[name] = val

    # Initialize Graph
    dot = Digraph(
        engine="dot",
        graph_attr={
            "ranksep": "0.3",
            "nodesep": "0",
            "packmode": "array_u",
            "size": "10,10",
        },
        node_attr={"fontsize": "14", "shape": "none"},
        edge_attr={"arrowsize": "0.5"},
        format="png",
    )

    if len(links) == 0:
        warnings.warn("No components shared across runs")
        return dot

    # Initialize Nodes
    for k in sorted(names2):
        if k in link_names2:
            color = "black"
            font = "helvetica"
        else:
            color = "red"
            font = "helvetica-bold"
        dot.node(
            "data2_" + str(k),
            label=name_dict2[k],
            _attributes={"fontcolor": color, "fontname": font},
        )

    for k in sorted(names1):
        if k in link_names1:
            color = "black"
            font = "helvetica"
        else:
            color = "red"
            font = "helvetica-bold"
        dot.node(
            "data1_" + str(k),
            label=name_dict1[k],
            _attributes={"fontcolor": color, "fontname": font},
        )

    # Add links between related components
    for k1, k2, dist in links:
        width = dist * 5
        dot.edge(
            "data1_" + str(k1),
            "data2_" + str(k2),
            _attributes={"penwidth": "{:.2f}".format(width)},
        )

    return dot


def convert_gene_index(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    ortho_file: Optional[str] = None,
    keep_locus=False,
):
    """
    Reorganizes and renames genes in a dataframe to be consistent with
    another object/organism

    Parameters
    ----------
    df1 : pd.DataFrame
        Dataframe from the first object/organism
    df2 : pd.DataFrame
        Dataframe from the second object/organism
    ortho_file : str or pd.DataFrame
        Path to orthology file between organisms
    keep_locus : bool
        If True, keep old locus tags as a column

    Returns
    -------
    pd.DataFrame
        M matrix for organism 2 with indexes translated into orthologs
    """

    if ortho_file is None:
        common_genes = df1.index.intersection(df2.index)
        df1_new = df1.loc[common_genes]
        df2_new = df2.loc[common_genes]
    else:

        try:
            DF_orth = pd.read_csv(ortho_file)
        except (ValueError, TypeError):
            DF_orth = ortho_file

        DF_orth = DF_orth[
            DF_orth.gene.isin(df1.index) & DF_orth.subject.isin(df2.index)
        ]
        subject2gene = DF_orth.set_index("subject").gene.to_dict()
        df1_new = df1.loc[DF_orth.gene]
        df2_new = df2.loc[DF_orth.subject]

        # Reset index of df2 to conform with df1
        if keep_locus:
            df2_new.index.name = "locus_tag"
            df2_new.reset_index(inplace=True)
            df2_new.index = [subject2gene[idx] for idx in df2_new.locus_tag]
        else:
            df2_new.index = [subject2gene[idx] for idx in df2_new.index]

    if len(df1_new) == 0 or len(df2_new) == 0:
        raise ValueError(
            "No shared genes. Check that matrix 1 conforms to "
            "the 'gene' column of the BBH file and matrix 2 "
            "conforms to the 'subject' column"
        )
    return df1_new, df2_new


def compare_ica(
    M1: pd.DataFrame,
    M2: pd.DataFrame,
    ortho_file: Optional[str] = None,
    cutoff: float = 0.25,
    method: Union[str, Callable] = "pearson",
    plot: bool = True,
    show_all: bool = False,
):
    """
    Compares two M matrices between a single organism or across organisms and
    returns the connected iModulons

    Parameters
    ----------
    M1 : pd.DataFrame
        M matrix from the first organism
    M2 : pd.DataFrame
        M matrix from the second organism
    ortho_file : str
        Path to orthology file between organisms
    method : str
        Correlation metric to use (see pd.DataFrame.corr)
    cutoff : float
        Cut off value for correlation metric
    plot : bool
        Create dot plot of matches
    show_all : bool
        Show all iModulons regardless of their linkage

    Returns
    -------
    matches
        Links and distances of connected iModulons
    dot
        Dot graph of connected iModulons
    """

    new_M1, new_M2 = convert_gene_index(M1, M2, ortho_file)
    new_M1.columns = new_M1.columns.astype("str")
    new_M2.columns = new_M2.columns.astype("str")
    matches = _get_orthologous_imodulons(new_M1, new_M2, method=method, cutoff=cutoff)
    if plot:
        dot = _make_dot_graph(
            matches, show_all=show_all, names1=M1.columns, names2=M2.columns
        )
        return matches, dot
    else:
        return matches


####################
# BBH CSV Creation #
####################


def make_prots(gbk: str, out_path: str):
    """
    Makes protein files for all the genes in the genbank file. Adapted from
    code by Saugat Poudel
    Parameters
    ----------
    gbk : str
        Path to input genbank file
    out_path : str
        Path to the output FASTA file

    Returns
    -------
    None
    """

    with open(out_path, "w") as fa:
        for refseq in SeqIO.parse(gbk, "genbank"):
            for feats in [f for f in refseq.features if f.type == "CDS"]:
                lt = feats.qualifiers["locus_tag"][0]
                try:
                    seq = feats.qualifiers["translation"][0]
                except KeyError:
                    seq = feats.extract(refseq.seq).translate()
                if seq:
                    fa.write(">{}\n{}\n".format(lt, seq))


def make_prot_db(fasta_file: os.PathLike):
    """
    Creates GenBank Databases from Protein FASTA of an organism

    Parameters
    ----------
    fasta_file : str
        Path to protein FASTA file

    Returns
    -------
    None
    """

    if (
        os.path.isfile(fasta_file + ".phr")
        and os.path.isfile(fasta_file + ".pin")
        and os.path.isfile(fasta_file + ".psq")
    ):
        print("BLAST DB files already exist")
        return None

    cmd_line = ["makeblastdb", "-in", fasta_file, "-parse_seqids", "-dbtype", "prot"]

    print("running makeblastdb with following command line...")
    print(" ".join(cmd_line))
    try:
        subprocess.check_call(cmd_line)
        print("Protein DB files created successfully")
    except subprocess.CalledProcessError:
        print(
            "\nmakeblastdb run failed. Make sure makeblastdb is"
            " installed and working properly, and that the protein FASTA "
            "file contains no duplicate genes. View the output below to "
            "see what error occured:\n"
        )
        status = subprocess.run(cmd_line, capture_output=True)
        print(status)
        # raise err
    return None


# TODO: some genbanks put alternate start codon such as TTG as methionine while
# others label it as leucine.
# need to check and fix this.

# noinspection PyTypeChecker
def get_bbh(
    db1: os.PathLike,
    db2: os.PathLike,
    outdir: os.PathLike = "bbh",
    outname: os.PathLike = None,
    mincov: float = 0.8,
    evalue: float = 0.001,
    threads: int = 1,
    force: bool = False,
    savefiles=True,
):
    """
    Runs Bidirectional Best Hit BLAST to find orthologs utilizing two protein
    FASTA files. Outputs a CSV file of all orthologous genes.

    Parameters
    ----------
    db1 : str
        Path to protein FASTA file for organism 1
    db2 : str
        Path to protein FASTA file for organism 2
    outdir : str
        Path to output directory (default: "bbh")
    outname : str
        Name of output CSV file (default: <db1>_vs_<db2>_parsed.csv)
    mincov : float
        Minimum coverage to call hits in BLAST, must be between 0 and 1 (default: 0.8)
    evalue : float
        E-value threshold for BlAST hist (default: .001)
    threads : int
        Number of threads to use for BLAST (default: 1)
    force : bool
        If True, overwrite existing files (default: False)
    savefiles : bool
        If True, save files to <outdir> (default: True)

    Returns
    -------
    pd.DataFrame
        Table of bi-directional BLAST hits between the two organisms
    """

    # check if files exist, and vars are appropriate
    if not _all_clear(db1, db2, outdir, mincov):
        return None
    # get get the db names, will be used for outfile names
    on1 = ".".join(os.path.split(db1)[-1].split(".")[:-1])
    on2 = ".".join(os.path.split(db2)[-1].split(".")[:-1])

    # run and save BLAST results
    bres1 = os.path.join(outdir, "{}_vs_{}.txt".format(on2, on1))
    bres2 = os.path.join(outdir, "{}_vs_{}.txt".format(on1, on2))

    _run_blastp(db1, db2, bres1, evalue, threads, force)
    _run_blastp(db2, db1, bres2, evalue, threads, force)

    db1_lengths = _get_gene_lens(db1)
    db2_lengths = _get_gene_lens(db2)

    if not outname:
        outname = "{}_vs_{}_parsed.csv".format(on1, on2)

    out_file = os.path.join(outdir, outname)
    files = glob(os.path.join(outdir, "*_parsed.csv"))

    if not force and out_file in files:
        print("bbh already parsed for", on1, on2)
        out = pd.read_csv(out_file, index_col=0)
        return out
    print("parsing BBHs for", on1, on2)

    cols = [
        "gene",
        "subject",
        "PID",
        "alnLength",
        "mismatchCount",
        "gapOpenCount",
        "queryStart",
        "queryEnd",
        "subjectStart",
        "subjectEnd",
        "eVal",
        "bitScore",
    ]

    bbh_file1 = os.path.join(outdir, "{}_vs_{}.txt".format(on1, on2))
    bbh_file2 = os.path.join(outdir, "{}_vs_{}.txt".format(on2, on1))

    bbh = pd.read_csv(bbh_file1, sep="\t", names=cols)
    bbh = pd.merge(bbh, db1_lengths)

    bbh["COV"] = bbh["alnLength"] / bbh["gene_length"]

    bbh2 = pd.read_csv(bbh_file2, sep="\t", names=cols)
    bbh2 = pd.merge(bbh2, db2_lengths)
    bbh2["COV"] = bbh2["alnLength"] / bbh2["gene_length"]

    # Strip "lcl|" from protein files taken from NCBI
    bbh.gene = bbh.gene.str.strip("lcl|")
    bbh.subject = bbh.subject.str.strip("lcl|")
    bbh2.gene = bbh2.gene.str.strip("lcl|")
    bbh2.subject = bbh2.subject.str.strip("lcl|")

    # FILTER GENES THAT HAVE COVERAGE < mincov
    bbh = bbh[bbh.COV >= mincov]
    bbh2 = bbh2[bbh2.COV >= mincov]
    list2struct = []

    # find if genes are directionally best hits of each other
    for g in bbh.gene.unique():
        res = bbh[bbh.gene == g]
        if len(res) == 0:
            continue

        # find BLAST hit with highest percent identity (PID)
        best_hit = res.loc[res.PID.idxmax()].copy()
        res2 = bbh2[bbh2.gene == best_hit.subject]
        if len(res2) == 0:  # no match
            continue
        # find BLAST hit with higest PID in the reciprocal BLAST
        best_gene2 = res2.loc[res2.PID.idxmax(), "subject"]

        # if doing forward then reciprocal BLAST nets the same gene -> BBH
        if g == best_gene2:
            best_hit["BBH"] = "<=>"
        else:  # only best hit in one direction
            best_hit["BBH"] = "->"

        list2struct.append(best_hit)

    out = pd.DataFrame(list2struct)

    out = out[out["BBH"] == "<=>"]

    if savefiles:
        print("Saving results to: " + out_file)
        out.to_csv(out_file)
    else:
        os.remove(bbh_file1)
        os.remove(bbh_file2)
    return out


def _get_gene_lens(file_in):
    """
    Computes gene lengths

    Parameters
    ----------
    file_in : str
        Input file path

    Returns
    -------
    pd.DataFrame
        Table of gene lengths
    """

    handle = open(file_in)
    records = SeqIO.parse(handle, "fasta")
    out = []
    for record in records:
        out.append({"gene": record.name, "gene_length": len(record.seq)})

    out = pd.DataFrame(out)
    return out


def _run_blastp(db1, db2, out, evalue, threads, force):
    """
    Runs BLASTP between two organisms

    Parameters
    ----------
    db1 : str
        Path to protein FASTA file for organism 1
    db2 : str
        Path to protein FASTA file for organism 2
    out : str
        Path for BLASTP output
    evalue : float
        E-value threshold for BlAST hist=
    threads : int
        Number of threads to use for BLAST
    force : bool
        If True, overwrite existing files

    Returns
    -------
    out : str
        Path of BLASTP output
    """

    if not force and os.path.isfile(out):
        print(db1, " already blasted")
        return out

    print("blasting {} vs {}".format(db1, db2))
    cmd_line = [
        "blastp",
        "-db",
        db1,
        "-query",
        db2,
        "-out",
        out,
        "-evalue",
        str(evalue),
        "-outfmt",
        "6",
        "-num_threads",
        str(threads),
    ]

    print("running blastp with following command line...")
    print(" ".join(cmd_line))
    try:
        subprocess.check_call(cmd_line)
    except subprocess.CalledProcessError as err:
        print("BLAST run failed. Make sure BLAST is" " installed and working properly.")
        raise err
    return out


def _all_clear(db1, db2, outdir, mincov):
    if not 0 < mincov <= 1:
        print("Coverage must be greater than 0 and less than or equal to 1")
        return None

    if not (os.path.isfile(db1) and os.path.isfile(db2)):
        print("One of the fasta file is missing")
        return None

    for i in [".phr", ".pin", ".psq"]:
        if not os.path.isfile(db1 + i) or not os.path.isfile(db2 + i):
            print("Some of the BLAST db files are missing")
            return None

    if not os.path.isdir(outdir):
        print("Making the output directory: " + outdir)
        os.mkdir(outdir)
    return True


def _same_output(df1, df2):
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    if all(df1.eq(df2)):
        print("The two outputs are the same.")
        return True
    elif all(df1.eq(df2.rename(columns={"subject": "gene", "gene": "subject"}))):
        print(
            "The two outputs are the same, " "but the genes and subject are switched."
        )
        return True
    else:
        print("The two outputs are not the same.")
        return False