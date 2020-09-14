"""
Utility functions for gene annotation
"""

import re
import pandas as pd
from io import StringIO
import urllib


def _get_attr(attributes, attr_id, ignore=False):
    """
    Helper function for parsing GFF annotations
    Args:
        attributes: Dictionary of attributes
        attr_id: Attribute ID
        ignore: Ignore error if ID not in attributes

    Returns:
        Value of attribute
    """

    try:
        return re.search(attr_id + '=(.*?)(;|$)', attributes).group(1)
    except AttributeError:
        if ignore:
            return None
        else:
            raise ValueError('{} not in attributes: {}'
                             .format(attr_id, attributes))


def gff2pandas(gff_file):
    """
    Converts GFF file to pandas DataFrame
    Args:
        gff_file: Path to GFF file

    Returns:
        Pandas DataFrame containing GFF information
    """
    with open(gff_file, 'r') as f:
        lines = f.readlines()

    # Get lines to skip
    skiprow = sum([line.startswith('#') for line in lines])

    # Read GFF
    names = ['refseq', 'source', 'feature', 'start', 'end', 'score', 'strand',
             'phase', 'attributes']
    DF_gff = pd.read_csv(gff_file, sep='\t', skiprows=skiprow,
                         names=names, header=None)

    # Filter for CDSs
    DF_cds = DF_gff[DF_gff.feature == 'CDS']

    # Also filter for genes to get old_locus_tag
    DF_gene = DF_gff[DF_gff.feature == 'gene'].reset_index()
    DF_gene['locus_tag'] = DF_gene.attributes.apply(_get_attr,
                                                    attr_id='locus_tag')
    DF_gene['old_locus_tag'] = DF_gene.attributes.apply(_get_attr,
                                                        attr_id='old_locus_tag',
                                                        ignore=True)
    DF_gene = DF_gene[['locus_tag', 'old_locus_tag']]

    # Sort by start position
    DF_cds = DF_cds.sort_values('start')

    # Extract attribute information
    DF_cds['locus_tag'] = DF_cds.attributes.apply(_get_attr,
                                                  attr_id='locus_tag')

    DF_cds['gene_name'] = DF_cds.attributes.apply(_get_attr, attr_id='gene',
                                                  ignore=True)

    DF_cds['gene_product'] = DF_cds.attributes.apply(_get_attr,
                                                     attr_id='product')

    DF_cds['ncbi_protein'] = DF_cds.attributes.apply(_get_attr,
                                                     attr_id='protein_id',
                                                     ignore=True)

    # Merge in old_locus_tag
    DF_cds = pd.merge(DF_cds, DF_gene, how='left', on='locus_tag', sort=False)

    return DF_cds


##############
# ID Mapping #
##############

def uniprot_id_mapping(prot_list, input_id='ACC+ID', output_id='P_REFSEQ_AC',
                       input_name='input_id', output_name='output_id'):
    """
    Convert a list of uniprot IDs to Refseq or Genbank IDs
    Args:
        prot_list:
        input_id:
        output_id:
        input_name:
        output_name:

    Returns:

    """

    url = 'https://www.uniprot.org/uploadlists/'

    params = {
        'from': input_id,
        'to': output_id,
        'format': 'tab',
        'query': ' '.join(prot_list)
    }

    # Send mapping request to uniprot
    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')
    req = urllib.request.Request(url, data)
    with urllib.request.urlopen(req) as f:
        response = f.read()

    # Load result to pandas dataframe
    text = StringIO(response.decode('utf-8'))
    mapping = pd.read_csv(text, sep='\t', header=0,
                          names=[input_name, output_name])

    # Only keep one uniprot ID per gene
    mapping = mapping.sort_values(output_name).drop_duplicates(input_name)
    return mapping
