import tqdm
import warnings
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
from typing import *
from graphviz import Digraph
from scipy import stats
from glob import glob
from pymodulon.visualization import scatterplot



def _make_dot_graph(M1: pd.DataFrame, M2: pd.DataFrame, metric: str,
                    cutoff: float, show_all: bool):
    """
    Given two M matrices, returns the dot graph and name links of the various
    connected ICA components
    Args:
        M1: M matrix from the first organism
        M2: M matrix from the second organism
        metric: Statistical test to use (default:'pearson')
        cutoff: Float cut off value for pearson statistical test
        show_all: Show all iModulons regardless of their linkage

    Returns: Dot graph and name links of connected ICA components

    """

    # Only keep genes found in both S matrices
    common = set(M1.index) & set(M2.index)

    if len(common) == 0:
        raise KeyError("No common genes")

    s1 = M1.reindex(common)
    s2 = M2.reindex(common)

    # Split names in half if necessary for comp1
    col_dict1 = {}
    for col in s1.columns:
        val = str(col)
        if len(val) > 10:
            col_dict1[col] = val[:len(val) // 2] + '-\n' + val[len(val) // 2:]
        else:
            col_dict1[col] = val
    s1.columns = [col_dict1[x] for x in s1.columns]

    # Split names in half if necessary for comp2
    col_dict2 = {}
    for col in s2.columns:
        val = str(col)
        if len(val) > 10:
            col_dict2[col] = val[:len(val) // 2] + '-\n' + val[len(val) // 2:]
        else:
            col_dict2[col] = val
    s2.columns = [col_dict2[x] for x in s2.columns]

    # Calculate correlation matrix
    corr = np.zeros((len(s1.columns), len(s2.columns)))

    for i, k1 in tqdm.tqdm(enumerate(s1.columns), total=len(s1.columns)):
        for j, k2 in enumerate(s2.columns):
            if metric == 'pearson':
                corr[i, j] = abs(stats.pearsonr(s1[k1], s2[k2])[0])
            elif metric == 'spearman':
                corr[i, j] = abs(stats.spearmanr(s1[k1], s2[k2])[0])

    # Only keep genes found in both S matrices
    DF_corr = pd.DataFrame(corr, index=s1.columns, columns=s2.columns)

    # Initialize Graph
    dot = Digraph(engine='dot', graph_attr={'ranksep': '0.3', 'nodesep': '0',
                                            'packmode': 'array_u',
                                            'size': '10,10'},
                  node_attr={'fontsize': '14', 'shape': 'none'},
                  edge_attr={'arrowsize': '0.5'}, format='png')

    # Set up linkage and designate terminal nodes
    # noinspection PyTypeChecker
    loc1, loc2 = np.where(DF_corr > cutoff)
    links = list(zip(s1.columns[loc1], s2.columns[loc2]))

    if len(links) == 0:
        warnings.warn('No components shared across runs')
        return None, None
    if show_all is True:
        # Initialize Nodes
        for k in sorted(s2.columns):
            if k in s2.columns[loc2]:
                color = 'black'
                font = 'helvetica'
            else:
                color = 'red'
                font = 'helvetica-bold'
            dot.node('data2_' + str(k), label=k,
                     _attributes={'fontcolor': color, 'fontname': font})

        for k in s1.columns:
            if k in s1.columns[loc1]:
                color = 'black'
                font = 'helvetica'
            else:
                color = 'red'
                font = 'helvetica-bold'
            dot.node('data1_' + str(k), label=k,
                     _attributes={'fontcolor': color, 'fontname': font})

        # Add links between related components
        for k1, k2 in links:
            width = DF_corr.loc[k1, k2] * 5
            dot.edge('data1_' + str(k1), 'data2_' + str(k2),
                     _attributes={'penwidth': '{:.2f}'.format(width)})
    else:
        # Initialize Nodes
        for k in sorted(s2.columns):
            if k in s2.columns[loc2]:
                color = 'black'
                font = 'helvetica'
                dot.node('data2_' + str(k), label=k,
                         _attributes={'fontcolor': color, 'fontname': font})

        for k in s1.columns:
            if k in s1.columns[loc1]:
                color = 'black'
                font = 'helvetica'
                dot.node('data1_' + str(k), label=k,
                         _attributes={'fontcolor': color, 'fontname': font})

        # Add links between related components
        for k1, k2 in links:
            if k1 in s1.columns[loc1] and k2 in s2.columns[loc2]:
                width = DF_corr.loc[k1, k2] * 5
                dot.edge('data1_' + str(k1), 'data2_' + str(k2),
                         _attributes={'penwidth': '{:.2f}'.format(width)})

    # Reformat names back to normal
    name1, name2 = list(zip(*links))
    inv_cols1 = {v: k for k, v in col_dict1.items()}
    inv_cols2 = {v: k for k, v in col_dict2.items()}

    name_links = list(zip([inv_cols1[x] for x in name1],
                          [inv_cols2[x] for x in name2],
                          [DF_corr.loc[name1[x]][name2[x]]
                           for x in range(0, len(name1))]))
    return dot, name_links


def _pull_bbh_csv(ortho_file: str, S1: pd.DataFrame):
    """
    Receives an the S matrix for an organism and returns the same S matrix
    with index genes translated into the orthologs in organism 2
    Args:
        ortho_file: String path to the bbh CSV file in the
        "modulome_compare_data" repository.
        Ex. "../../modulome_compare_data/bbh_csv/
        bSubtilis_full_protein_vs_sAureus_full_protein_parsed.csv"
        S1: Pandas DataFrame of the S matrix for organism 1

    Returns:
        Pandas DataFrame of the S matrix for organism 1 with indexes
        translated into orthologs

    """

    bbh_DF = pd.read_csv(ortho_file, index_col="gene")

    S1_copy = S1.copy()
    S1_index = list(S1_copy.index)

    for i in range(0, len(S1_index)):
        try:
            S1_index[i] = bbh_DF.loc[S1_index[i]]["subject"]
        except KeyError:
            continue
    S1_copy.index = S1_index

    return S1_copy


def compare_ica(S1: pd.DataFrame, S2: pd.DataFrame,
                ortho_file: Optional[str] = None, cutoff: float = 0.2,
                metric='pearson', show_all: bool = False,
                plot_express: bool = False):
    """
    Compares two S matrices between a single organism or across organisms and
    returns the connected ICA components
    Args:
        S1: Pandas Dataframe of S matrix 1
        S2: Pandas Dataframe of S Matrix 2
        ortho_file: String of the location where organism data can be found
            (can be found under modulome/data)
        cutoff: Float cut off value for pearson statistical test
        metric: A string of what statistical test to use (standard is 'pearson')
        show_all: True will show all nodes of the digraph matrix

    Returns: Dot graph and name links of connected ICA components between the
    two runs or organisms.

    """
    if ortho_file is None:
        dot, name_links = _make_dot_graph(S1, S2, metric=metric,
                                          cutoff=cutoff,
                                          show_all=show_all)
        return dot, name_links

    else:
        warnings.warn("Please ensure that the order of S1 and S2 match the "
                      "order of the BBH CSV file")
        translated_S = _pull_bbh_csv(ortho_file, S1)
        dot, name_links = _make_dot_graph(translated_S, S2, metric,
                                          cutoff, show_all=show_all)

        if plot_express is True:
            common = set(translated_S.index) & set(S2.index)
            reduced_S1 = translated_S.reindex(common)
            reduced_S2 = S2.reindex(common)

            print(len(name_links))

            if len(name_links) == 1:
                subplot_dims = (1, 1)
            elif len(name_links) == 2:
                subplot_dims = (1, 2)
            elif len(name_links) <= 4:
                subplot_dims = (2, 2)
            elif len(name_links) <= 6:
                subplot_dims = (2, 3)
            elif len(name_links) <= 9:
                subplot_dims = (3, 3)
            elif len(name_links) <= 12:
                subplot_dims = (3, 4)
            elif len(name_links) <= 16:
                subplot_dims = (4, 4)

            _, axs = plt.subplots(*subplot_dims, figsize=(12, 12))
            if len(name_links) != 1:
                axs = axs.flatten()
            for num , (compare, ax) in enumerate(zip(name_links, axs)):
                scatterplot(reduced_S1[compare[0]],
                            reduced_S2[compare[1]],
                            fit_line=True,
                            xlabel="Organism 1: "+str(compare[0]),
                            ylabel="Organism 2: "+str(compare[1]),
                            ax=ax)

            return dot, name_links

        else:
            return dot, name_links


####################
# BBH CSV Creation #
####################

#

def make_prots(gbk: os.PathLike, out_path: os.PathLike):
    """
    Makes protein files for all the genes in the genbank file. Adapted from
    code by Saguat Poudel
    :param gbk: path to input genbank file
    :param out_path: path to the output FASTA file
    :return: None
    """
    with open(out_path, 'w') as fa:
        for refseq in SeqIO.parse(gbk, 'genbank'):
            for feats in [f for f in refseq.features if f.type == 'CDS']:
                lt = feats.qualifiers['locus_tag'][0]
                try:
                    seq = feats.qualifiers['translation'][0]
                except KeyError:
                    seq = feats.extract(refseq.seq).translate()
                if seq:
                    fa.write('>{}\n{}\n'.format(lt, seq))


def make_prot_db(fasta_file: os.PathLike):
    """
    Creates GenBank Databases from Protein FASTA of an organism (Output from
    make_prot function)
    Args:
        fasta_file: String path to protein FASTA file
    Returns: None

    """
    if os.path.isfile(fasta_file + ".phr") \
            and os.path.isfile(fasta_file + ".pin") \
            and os.path.isfile(fasta_file + ".psq"):
        print('BLAST DB files already exist')
        return None

    cmd_line = ['makeblastdb', "-in", fasta_file, "-parse_seqids", "-dbtype",
                "prot"]

    print('running makeblastdb with following command line...')
    print(' '.join(cmd_line))
    try:
        subprocess.check_call(cmd_line)
        print("Protein DB files created successfully")
    except subprocess.CalledProcessError:
        print('\nmakeblastdb run failed. Make sure makeblastdb is'
              ' installed and working properly, and that the protein FASTA '
              'file contains no duplicate genes. View the output below to '
              'see what error occured:\n')
        status = subprocess.run(cmd_line, capture_output=True)
        print(status)
        # raise err
    return None


# TODO: some genbanks put alternate start codon such as TTG as methionine while
# others label it as leucine.
# need to check and fix this.

# noinspection PyTypeChecker
def get_bbh(db1: os.PathLike, db2: os.PathLike, outdir: os.PathLike = 'bbh',
            outname: os.PathLike = None, mincov: float = 0.8,
            evalue: float = 0.001, threads: int = 1, force: bool = False,
            savefiles=True):
    """
    Runs Bidirectional Best Hit BLAST to find orthologs utilizing two protein
    FASTA files. Outputs a CSV file of all orthologous genes. Adopted from code
    Args:
        db1: String path to protein FASTA file (output of make_prots
        function) for organism 1
        db2: String path to protein FASTA file (output of make_prots
        function) for organism 2
        outdir: String path to output directory, default is "bbh" and will
        create the directory if it does not exist
        outname: Default db1_vs_db2_parsed.csv where db[1-2] are the passed
        arguments name of the csv file where that will save the results
        mincov: Minimum coverage to call hits in BLAST, must be between 0 and 1
        evalue: evalue thershold for BLAST hits, Default .001
        threads: Number of threads to run BLAST, Default 1
        force: Whether to overwrite existing files or not
        savefiles: Whether to save files to outdir

    Returns:

    """
    # check if files exist, and vars are appropriate
    if not _all_clear(db1, db2, outdir, mincov):
        return None
    # get get the db names, will be used for outfile names
    on1 = '.'.join(os.path.split(db1)[-1].split('.')[:-1])
    on2 = '.'.join(os.path.split(db2)[-1].split('.')[:-1])

    # run and save BLAST results
    bres1 = os.path.join(outdir, '{}_vs_{}.txt'.format(on2, on1))
    bres2 = os.path.join(outdir, '{}_vs_{}.txt'.format(on1, on2))

    _run_blastp(db1, db2, bres1, evalue, threads, force)
    _run_blastp(db2, db1, bres2, evalue, threads, force)

    db1_lengths = _get_gene_lens(db1)
    db2_lengths = _get_gene_lens(db2)

    if not outname:
        outname = '{}_vs_{}_parsed.csv'.format(on1, on2)

    out_file = os.path.join(outdir, outname)
    files = glob(os.path.join(outdir, '*_parsed.csv'))

    if not force and out_file in files:
        print('bbh already parsed for', on1, on2)
        out = pd.read_csv(out_file)
        return out
    print('parsing BBHs for', on1, on2)

    cols = ['gene', 'subject', 'PID', 'alnLength', 'mismatchCount',
            'gapOpenCount', 'queryStart',
            'queryEnd', 'subjectStart', 'subjectEnd', 'eVal', 'bitScore']

    bbh_file1 = os.path.join(outdir, '{}_vs_{}.txt'.format(on1, on2))
    bbh_file2 = os.path.join(outdir, '{}_vs_{}.txt'.format(on2, on1))

    bbh = pd.read_csv(bbh_file1, sep='\t', names=cols)
    bbh = pd.merge(bbh, db1_lengths)

    bbh['COV'] = bbh['alnLength'] / bbh['gene_length']

    bbh2 = pd.read_csv(bbh_file2, sep='\t', names=cols)
    bbh2 = pd.merge(bbh2, db2_lengths)
    bbh2['COV'] = bbh2['alnLength'] / bbh2['gene_length']

    # FILTER GENES THAT HAVE COVERAGE < mincov
    bbh = bbh[bbh.COV >= mincov]
    bbh2 = bbh2[bbh2.COV >= mincov]
    out = pd.DataFrame()

    # find if genes are directionally best hits of each other
    for g in bbh.gene.unique():
        res = bbh[bbh.gene == g]
        if len(res) == 0:
            continue

        # find BLAST hit with highest percent identity (PID)
        best_hit = res.loc[res.PID.idxmax()]
        res2 = bbh2[bbh2.gene == best_hit.subject]
        if len(res2) == 0:  # no match
            continue
        # find BLAST hit with higest PID in the reciprocal BLAST
        best_gene2 = res2.loc[res2.PID.idxmax(), 'subject']

        # if doing forward then reciprocal BLAST nets the same gene -> BBH
        if g == best_gene2:
            best_hit.loc[:]['BBH'] = '<=>'
        else:  # only best hit in one direction
            best_hit.loc[:]['BBH'] = '->'
        out = pd.concat([out, pd.DataFrame(best_hit).transpose()])

    out = out[out['BBH'] == '<=>']

    if savefiles:
        print('Saving results to: ' + out_file)
        out.to_csv(out_file)
    else:
        os.remove(bbh_file1)
        os.remove(bbh_file2)
    return out


def _get_gene_lens(file_in):
    """

    Args:
        file_in:

    Returns:

    """
    handle = open(file_in)
    records = SeqIO.parse(handle, "fasta")
    out = []
    for record in records:
        out.append({'gene': record.name, 'gene_length': len(record.seq)})

    out = pd.DataFrame(out)
    return out


def _run_blastp(db1, db2, out, evalue, threads, force):
    """

    Args:
        db1:
        db2:
        out:
        evalue:
        threads:
        force:

    Returns:

    """

    if not force and os.path.isfile(out):
        print(db1, ' already blasted')
        return out

    print('blasting {} vs {}'.format(db1, db2))
    cmd_line = ['blastp', '-db', db1, '-query', db2, '-out', out, '-evalue',
                str(evalue), '-outfmt', '6',
                '-num_threads', str(threads)]

    print('running blastp with following command line...')
    print(' '.join(cmd_line))
    try:
        subprocess.check_call(cmd_line)
    except subprocess.CalledProcessError as err:
        print('BLAST run failed. Make sure BLAST is'
              ' installed and working properly.')
        raise err
    return out


def _all_clear(db1, db2, outdir, mincov):
    if not 0 < mincov <= 1:
        print('Coverage must be greater than 0 and less than or equal to 1')
        return None

    if not (os.path.isfile(db1) and os.path.isfile(db2)):
        print('One of the fasta file is missing')
        return None

    for i in ['.phr', '.pin', '.psq']:
        if not os.path.isfile(db1 + i) or not os.path.isfile(db2 + i):
            print('Some of the BLAST db files are missing')
            return None

    if not os.path.isdir(outdir):
        print('Making the output directory: ' + outdir)
        os.mkdir(outdir)
    return True


def _same_output(df1, df2):
    """

    Args:
        df1:
        df2:

    Returns:

    """
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    if all(df1.eq(df2)):
        print('The two outputs are the same.')
        return True
    elif all(df1.eq(df2.rename(columns={'subject': 'gene',
                                        'gene': 'subject'}))):
        print('The two outputs are the same, '
              'but the genes and subject are switched.')
        return True
    else:
        print('The two outputs are not the same.')
        return False
