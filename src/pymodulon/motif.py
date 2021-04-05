# copy/pasted from the ICA version below

import os
import re
import subprocess
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from Bio import SeqIO
from bs4 import BeautifulSoup

import pymodulon.data.meme
from pymodulon.core import IcaData, MotifInfo


def _get_upstream_seqs(
    ica_data: IcaData,
    imodulon: Union[str, int],
    seq_dict: Dict,
    upstream: int,
    downstream: int,
):
    """
    Get upstream sequences for a table of operons

    Parameters
    ----------
    ica_data: IcaData
        IcaData object
    imodulon: Union[str,int]
        Name of iModulon
    seq_dict: Dict
        Dictionary mapping accession numbers to Biopython SeqRecords
    upstream: int
        Number of basepairs upstream from first gene in operon to include in motif
        search
    downstream: int
        Number of basepairs upstream from first gene in operon to include in motif
        search
    Returns
    -------
    pd.DataFrame
        DataFrame containing operon information
    List[SeqRecord]
        List of SeqRecords containing upstream sequences
    """

    # Get list of operons in component
    enriched_genes = ica_data.view_imodulon(imodulon).index
    enriched_operons = ica_data.gene_table.loc[enriched_genes]

    # Get upstream sequences
    list2struct = []
    seq_records = []
    for name, group in enriched_operons.groupby("operon"):
        genes = ",".join(group.gene_name)
        ids = ",".join(group.index)
        genome = seq_dict[group.accession[0]]
        if all(group.strand == "+"):
            pos = min(group.start)
            start_pos = max(0, pos - upstream)
            sequence = genome[start_pos : pos + downstream]
            seq = SeqIO.SeqRecord(seq=sequence, id=name)
            list2struct.append([name, genes, ids, start_pos, "+", str(seq.seq)])
            seq_records.append(seq)
        elif all(group.strand == "-"):
            pos = max(group.end)
            start_pos = max(0, pos - downstream)
            sequence = genome[start_pos : pos + upstream]
            seq = SeqIO.SeqRecord(seq=sequence, id=name)
            list2struct.append([name, genes, ids, start_pos, "-", str(seq.seq)])
            seq_records.append(seq)
        else:
            raise ValueError("Operon contains genes on both strands:", name)

    DF_seqs = pd.DataFrame(
        list2struct,
        columns=["operon", "genes", "locus_tags", "start_pos", "strand", "seq"],
    ).set_index("operon")
    return DF_seqs, seq_records


def find_motifs(
    ica_data: IcaData,
    imodulon: Union[int, str],
    fasta_file: Union[os.PathLike, List[os.PathLike]],
    outdir: Optional[os.PathLike] = None,
    palindrome: bool = False,
    nmotifs: int = 5,
    upstream: int = 500,
    downstream: int = 100,
    verbose: bool = True,
    force: bool = False,
    evt: float = 0.001,
    cores: int = 8,
    minw: int = 6,
    maxw: int = 40,
    minsites: Optional[int] = None,
):
    """
    Finds motifs upstream of genes in an iModulon

    Parameters
    ----------
    ica_data: IcaData
        IcaData object
    imodulon: Union[int, str]
        iModulon name
    fasta_file: Union[os.PathLike, List[os.PathLike]]
        Path or list of paths to fasta file(s) for organism
    outdir: os.PathLike
        Path to output directory
    palindrome: bool
        If True, limit search to palindromic motifs (default: False)
    nmotifs: int
        Number of motifs to search for (default: 5)
    upstream: int
        Number of basepairs upstream from first gene in operon to include in motif
        search (default: 500)
    downstream: int
        Number of basepairs upstream from first gene in operon to include in motif
        search (default: 100)
    verbose: bool
        Show steps in verbose output (default: True)
    force: bool
        Force execution of MEME even if output already exists (default: False)
    evt: float
        E-value threshold (default: 0.001)
    cores: int
        Number of cores to use (default: 8)
    minw: int
        Minimum motif width in basepairs (default: 6)
    maxw: int
        Maximum motif width in basepairs (default: 40)
    minsites: Optional[int]
        Minimum number of sites required for a motif. Default is the number of
        operons divided by 3.

    Returns
    -------
    # TODO: add documentation of return
    """

    # Handle output directory
    if outdir is None:
        outdir = "motifs"

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # Read in fasta sequence from file
    seq_dict = {}
    try:
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq_dict[record.id] = record.seq
    except AttributeError:
        for file in fasta_file:
            for record in SeqIO.parse(file, "fasta"):
                seq_dict[record.id] = record.seq

    # Ensure that all genome and plasmids have been loaded
    missing_acc = set(ica_data.gene_table.accession) - set(seq_dict.keys())
    if len(missing_acc) > 0:
        raise ValueError(f"Missing FASTA file for {missing_acc}")

    DF_seqs, seq_records = _get_upstream_seqs(
        ica_data, imodulon, seq_dict, upstream, downstream
    )

    n_seqs = len(DF_seqs)

    if n_seqs < 2:
        if verbose:
            print("Less than two sequences found for iModulon: {}".format(imodulon))
        return None

    # Handle options
    if verbose:
        print("Finding motifs for {:d} sequences".format(n_seqs))

    def full_path(name):
        return os.path.join(outdir, re.sub("/| ", "_", name))

    if palindrome:
        comp_dir = full_path("{}_pal".format(imodulon))
    else:
        comp_dir = full_path(str(imodulon))

    if minsites is None:
        minsites = max(2, int(n_seqs / 3))

    # Populate command
    fasta = full_path("{}.fasta".format(imodulon))

    cmd = [
        "meme",
        fasta,
        "-oc",
        comp_dir,
        "-dna",
        "-mod",
        "zoops",
        "-p",
        str(cores),
        "-nmotifs",
        str(nmotifs),
        "-evt",
        str(evt),
        "-minw",
        str(minw),
        "-maxw",
        str(maxw),
        "-allw",
        "-minsites",
        str(minsites),
    ]

    if palindrome:
        cmd.append("-pal")

    # Skip intensive tasks on rerun
    if force or not os.path.isdir(comp_dir):
        # Write sequence to file
        SeqIO.write(seq_records, fasta, "fasta")

        # print MEME command
        if verbose:
            print("Running command:", " ".join(cmd))

        # Run MEME
        subprocess.call(cmd)

    # Save results
    DF_motifs, DF_sites = _parse_meme(comp_dir, DF_seqs, verbose=verbose, evt=evt)

    if DF_motifs.empty:
        return None
    else:
        result = MotifInfo(
            DF_motifs, DF_sites, " ".join(cmd), os.path.join(comp_dir, "meme.txt")
        )

        ica_data.motif_info[imodulon] = result

    return result


def _parse_meme(directory, DF_seqs, verbose, evt):
    # Read MEME results
    with open(os.path.join(directory, "meme.xml"), "r") as f:
        result_file = BeautifulSoup(f.read(), "lxml")

    # Convert to motif XML file to dataframes: (overall,[individual_motif])
    DF_motifs = pd.DataFrame(columns=["e_value", "sites", "width", "consensus"])
    dfs = []
    for motif in result_file.find_all("motif"):

        try:
            idx = motif["alt"]
        except KeyError:
            idx = motif["id"]

        # Motif statistics
        DF_motifs.loc[idx, "e_value"] = np.float64(motif["e_value"])
        DF_motifs.loc[idx, "sites"] = motif["sites"]
        DF_motifs.loc[idx, "width"] = motif["width"]
        DF_motifs.loc[idx, "consensus"] = motif["name"]

        # Map Sequence to name

        list_to_struct = []
        for seq in result_file.find_all("sequence"):
            list_to_struct.append([seq["id"], seq["name"]])
        df_names = pd.DataFrame(list_to_struct, columns=["seq_id", "operon"])

        # Get motif sites

        list_to_struct = []
        for site in motif.find_all("contributing_site"):
            site_seq = "".join(
                [letter["letter_id"] for letter in site.find_all("letter_ref")]
            )
            data = [site["position"], site["pvalue"], site["sequence_id"], site_seq]
            list_to_struct.append(data)

        tmp_df = pd.DataFrame(
            list_to_struct, columns=["rel_position", "pvalue", "seq_id", "site_seq"]
        )

        # Combine motif sites with sequence to name mapper
        DF_meme = pd.merge(tmp_df, df_names)
        DF_meme = DF_meme.set_index("operon").sort_index().drop("seq_id", axis=1)
        DF_meme = pd.concat([DF_meme, DF_seqs], axis=1, sort=True)
        DF_meme.index.name = idx

        # Report number of sequences with motif
        DF_motifs.loc[idx, "motif_frac"] = np.true_divide(
            sum(DF_meme.rel_position.notnull()), len(DF_meme)
        )

        # Remove full upstream sequence
        DF_meme.drop(columns="seq", inplace=True)
        dfs.append(DF_meme)

    if len(dfs) == 0:
        if verbose:
            print("No motif found with E-value < {0:.1e}".format(evt))
        idx1 = pd.Series([], name="motif")
        idx2 = pd.MultiIndex.from_tuples([], names=["motif", "operon"])
        return (
            pd.DataFrame(
                columns=[
                    "e_value",
                    "sites",
                    "width",
                    "consensus",
                    "motif_name",
                    "motif_frac",
                ],
                index=idx1,
            ),
            pd.DataFrame(
                columns=[
                    "rel_position",
                    "pvalue",
                    "site_seq",
                    "genes",
                    "locus_tags",
                    "start_pos",
                    "strand",
                ],
                index=idx2,
            ),
        )

    DF_sites = pd.concat({df.index.name: df for df in dfs})

    # Add index labels
    DF_sites.index.names = ["motif", "operon"]
    DF_motifs.index.name = "motif"

    if verbose:
        if len(DF_motifs) == 1:
            motif_str = "motif"
        else:
            motif_str = "motifs"

        if len(DF_sites) == 1:
            site_str = "site"
        else:
            site_str = "sites"
        print(
            f"Found {len(DF_motifs)} {motif_str} across "
            f"{sum(DF_sites.site_seq.notnull())} {site_str}"
        )

    return DF_motifs, DF_sites


def _parse_tomtom(tomtom_dir):
    DF_tomtom = pd.read_csv(
        os.path.join(tomtom_dir, "tomtom.tsv"), sep="\t", skipfooter=3, engine="python"
    )

    with open(os.path.join(tomtom_dir, "tomtom.xml"), "r") as f:
        result = BeautifulSoup(f.read(), "lxml")

    # Convert ID to names (for the PRODORIC database)
    id2name = {}
    for motif in result.find_all("motif"):
        try:
            id2name[motif["id"]] = motif["alt"]
        except KeyError:
            id2name[motif["id"]] = motif["id"]

    # Switch to correct name
    DF_tomtom["motif"] = DF_tomtom["Query_ID"].apply(lambda x: id2name[x])
    DF_tomtom["target"] = DF_tomtom["Target_ID"].apply(lambda x: id2name[x])

    # Get database names
    db_names = [db["name"] for db in result.find("target_dbs").find_all("db")]

    id2db = {}
    for motif in result.find("targets").find_all("motif"):
        id2db[motif["id"]] = db_names[int(motif["db"])]

    DF_tomtom["database"] = DF_tomtom["Target_ID"].apply(lambda x: id2db[x])

    # Reorder columns
    DF_tomtom = DF_tomtom[
        [
            "motif",
            "target",
            "Target_ID",
            "p-value",
            "E-value",
            "q-value",
            "Overlap",
            "Optimal_offset",
            "Orientation",
            "database",
        ]
    ]
    DF_tomtom.columns = [
        "motif",
        "target",
        "target_id",
        "pvalue",
        "Evalue",
        "qvalue",
        "overlap",
        "optimal_offset",
        "orientation",
        "database",
    ]
    return DF_tomtom


def compare_motifs(
    motif_info: Optional[MotifInfo] = None,
    motif_file: Optional[str] = None,
    motif_db: Optional[str] = None,
    outdir: Optional[str] = None,
    force=False,
    verbose=True,
    evt=0.001,
):
    """
    Compare a MEME motif against external motif databases

    Parameters
    ----------
    motif_info: Optional[MotifInfo]
        MotifInfo object. Either 'motif_info' or 'motif_file' must be provided.
        'motif_info' takes precedence over 'motif_file'.
    motif_file: Optional[os.PathLike]
        Txt file generated by MEME from a motif search.
    motif_db: Optional[Union[List,str]]
        Name or path to MEME formatted databases
    outdir: Optional[os.PathLike]
        Output directory for TOMTOM comparisons
    force: bool
        Force execution of TOMTOM even if output already exists (default: False)
    verbose: bool
        Show steps in verbose output (default: True)
    evt: float
        E-value threshold (default: 0.001)

    Returns
    -------

    """

    if motif_info:
        motif_file = motif_info.file
    elif motif_file is None:
        raise ValueError("Either 'motif_info' or 'motif_file' must be provided")

    # Ensure motif file exists
    if not os.path.isfile(motif_file):
        raise FileNotFoundError(f"File does not exist: '{motif_file}'")

    # Get motif DB locations
    motif_db_dir = pymodulon.data.meme.__path__[0]
    default_dbs = [
        "collectf",
        "dpinteract",
        "prodoric",
        "regtransbase",
        "SwissRegulon_e_coli",
    ]

    if motif_db is None:
        motif_db = default_dbs
        motif_db = [os.path.join(motif_db_dir, db + ".meme") for db in motif_db]
    elif isinstance(motif_db, str):
        motif_db = [motif_db]

    for db in motif_db:
        if db in default_dbs:
            motif_db = [os.path.join(motif_db_dir, db + ".meme") for db in motif_db]
        elif not os.path.isfile(db):
            raise FileNotFoundError(f"File does not exist: '{db}'")

    # Handle output directory
    if outdir is None:
        outdir = os.path.join(os.path.dirname(motif_file), "tomtom")

    # Run TOMTOM
    if not os.path.isdir(outdir) or force:
        cmd = [
            "tomtom",
            "-oc",
            outdir,
            "-thresh",
            str(evt),
            "-incomplete-scores",
            "-png",
            motif_file,
        ] + motif_db

        if verbose:
            print("Running command:", " ".join(cmd))

        res = subprocess.check_call(cmd)
        if res != 0:
            cmd_str = " ".join(cmd)
            raise RuntimeError(
                f"TOMTOM failed to execute the following command: {cmd_str}"
            )
    result = _parse_tomtom(outdir)

    # Save result
    if motif_info:
        motif_info.matches = result
    return result
