"""
Utility functions for gene annotation
"""

import logging
import re
import urllib
from io import StringIO

import pandas as pd


def cog2str(cog):
    """
    Get the full description for a COG category letter

    Parameters
    ----------
    cog : str
        COG category letter

    Returns
    -------
    str
        Description of COG category
    """

    cog_dict = {
        "A": "RNA processing and modification",
        "B": "Chromatin structure and dynamics",
        "C": "Energy production and conversion",
        "D": "Cell cycle control, cell division, chromosome partitioning",
        "E": "Amino acid transport and metabolism",
        "F": "Nucleotide transport and metabolism",
        "G": "Carbohydrate transport and metabolism",
        "H": "Coenzyme transport and metabolism",
        "I": "Lipid transport and metabolism",
        "J": "Translation, ribosomal structure and biogenesis",
        "K": "Transcription",
        "L": "Replication, recombination and repair",
        "M": "Cell wall/membrane/envelope biogenesis",
        "N": "Cell motility",
        "O": "Post-translational modification, protein turnover, and chaperones",
        "P": "Inorganic ion transport and metabolism",
        "Q": "Secondary metabolites biosynthesis, transport, and catabolism",
        "R": "General function prediction only",
        "S": "Function unknown",
        "T": "Signal transduction mechanisms",
        "U": "Intracellular trafficking, secretion, and vesicular transport",
        "V": "Defense mechanisms",
        "W": "Extracellular structures",
        "X": "No COG annotation",
        "Y": "Nuclear structure",
        "Z": "Cytoskeleton",
    }

    return cog_dict[cog]


def _get_attr(attributes, attr_id, ignore=False):
    """
    Helper function for parsing GFF annotations

    Parameters
    ----------
    attributes : str
        Attribute string
    attr_id : str
        Attribute ID
    ignore : bool
        If true, ignore errors if ID is not in attributes (default: False)

    Returns
    -------
    str, optional
        Value of attribute
    """

    try:
        return re.search(attr_id + "=(.*?)(;|$)", attributes).group(1)
    except AttributeError:
        if ignore:
            return None
        else:
            raise ValueError("{} not in attributes: {}".format(attr_id, attributes))


def gff2pandas(gff_file, feature="CDS", index=None):
    """
    Converts GFF file(s) to a Pandas DataFrame
    Parameters
    ----------
    gff_file : str or list
        Path(s) to GFF file
    feature: str or list
        Name(s) of features to keep (default = "CDS")
    index : str, optional
        Column or attribute to use as index

    Returns
    -------
    df_gff: ~pandas.DataFrame
        GFF formatted as a DataFrame
    """

    # Argument checking
    if isinstance(gff_file, str):
        gff_file = [gff_file]

    if isinstance(feature, str):
        feature = [feature]

    result = []

    for gff in gff_file:
        with open(gff, "r") as f:
            lines = f.readlines()

        # Get lines to skip
        skiprow = sum([line.startswith("#") for line in lines])

        # Read GFF
        names = [
            "accession",
            "source",
            "feature",
            "start",
            "end",
            "score",
            "strand",
            "phase",
            "attributes",
        ]
        DF_gff = pd.read_csv(gff, sep="\t", skiprows=skiprow, names=names, header=None)

        # Filter for CDSs
        DF_cds = DF_gff[DF_gff.feature.isin(feature)]

        # Also filter for genes to get old_locus_tag
        DF_gene = DF_gff[DF_gff.feature == "gene"].reset_index()
        DF_gene["locus_tag"] = DF_gene.attributes.apply(
            _get_attr, attr_id="locus_tag", ignore=True
        )
        DF_gene["old_locus_tag"] = DF_gene.attributes.apply(
            _get_attr, attr_id="old_locus_tag", ignore=True
        )
        DF_gene = DF_gene[["locus_tag", "old_locus_tag"]]
        DF_gene = DF_gene[DF_gene.locus_tag.notnull()]

        # Sort by start position
        DF_cds = DF_cds.sort_values("start")

        # Extract attribute information
        DF_cds["locus_tag"] = DF_cds.attributes.apply(_get_attr, attr_id="locus_tag")

        DF_cds["gene_name"] = DF_cds.attributes.apply(
            _get_attr, attr_id="gene", ignore=True
        )

        DF_cds["gene_product"] = DF_cds.attributes.apply(
            _get_attr, attr_id="product", ignore=True
        )

        DF_cds["ncbi_protein"] = DF_cds.attributes.apply(
            _get_attr, attr_id="protein_id", ignore=True
        )

        # Merge in old_locus_tag
        DF_cds = pd.merge(DF_cds, DF_gene, how="left", on="locus_tag", sort=False)

        result.append(DF_cds)

    DF_gff = pd.concat(result)

    if index:
        if DF_gff[index].duplicated().any():
            logging.warning("Duplicate {} detected. Dropping duplicates.".format(index))
            DF_gff = DF_gff.drop_duplicates(index)
        DF_gff.set_index("locus_tag", drop=True, inplace=True)

    return DF_gff


def reformat_biocyc_tu(tu):
    """

    Parameters
    ----------
    tu: str
        Biocyc-formatted transcription unit (i.e. 'thrL // thrA // thrB //
        thrC')

    Returns
    -------
    formatted_tu : str
        Semicolon-separated sorted gene list
    """
    try:
        return ";".join(sorted(tu.split(" // ")))
    except AttributeError:
        return None


##############
# ID Mapping #
##############


def uniprot_id_mapping(
    prot_list,
    input_id="ACC+ID",
    output_id="P_REFSEQ_AC",
    input_name="input_id",
    output_name="output_id",
):
    """
    Python wrapper for the uniprot ID mapping tool (See
    https://www.uniprot.org/uploadlists/)

    Parameters
    ----------
    prot_list : list
        List of proteins to be mapped
    input_id : str
        ID type for the mapping input (default: "ACC+ID")
    output_id : str
        ID type for the mapping output (default: "P_REFSEQ_AC")
    input_name : str
        Column name for input IDs
    output_name : str
        Column name for output IDs

    Returns
    -------
    mapping : ~pandas.DataFrame
        Table containing two columns, one listing the inputs, and one listing
        the mapped outputs. Column names are defined by input_name and
        output_name.

    """

    url = "https://www.uniprot.org/uploadlists/"

    params = {
        "from": input_id,
        "to": output_id,
        "format": "tab",
        "query": " ".join(prot_list),
    }

    # Send mapping request to uniprot
    data = urllib.parse.urlencode(params)
    data = data.encode("utf-8")
    req = urllib.request.Request(url, data)
    with urllib.request.urlopen(req) as f:
        response = f.read()

    # Load result to pandas dataframe
    text = StringIO(response.decode("utf-8"))
    mapping = pd.read_csv(text, sep="\t", header=0, names=[input_name, output_name])

    # Only keep one uniprot ID per gene
    mapping = mapping.sort_values(output_name).drop_duplicates(input_name)
    return mapping
