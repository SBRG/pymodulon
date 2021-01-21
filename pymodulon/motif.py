# copy/pasted from the ICA version below

import os
import pandas as pd
import re
from Bio import SeqIO
from Bio.Seq import Seq
import subprocess
from bs4 import BeautifulSoup
import numpy as np

###################
## MOTIF FINDING ##
###################

def find_motifs(ica_data,fasta_file,k,palindrome=False,nmotifs=5,upstream=500,downstream=100,
                verbose=True,force=False,evt=0.001,maxw=40):
    
    if not os.path.isdir('motifs'):
        os.mkdir('motifs')
    
    # Read in fasta sequence from file
    fasta_sequence = ''
    f = open(fasta_file, 'r')
    lines = f.readlines()
    f.close()
    for line in lines[1:]:
        fasta_sequence += line.replace('\n','')
    
    # Get list of operons in component
    enriched_genes = ica_data.gene_table.index[0:100] # xxx zzz delete the range
    enriched_operons = ica_data.gene_table.loc[enriched_genes]
    n_operons = len(enriched_operons.operon.unique())
    print(n_operons)
    
    # Return empty dataframe if under 4 operons or over 200 operons exist
    if n_operons <= 4 or n_operons > 200:
        return pd.DataFrame(columns = ['motif_frac']),pd.DataFrame()
    
    # Get upstream sequences
    list2struct = []
    seqs = []
    for name,group in enriched_operons.groupby('operon'):
        genes = ','.join(group.gene_name)
        ids = ','.join(group.index)
        if all(group.strand == '+'):
            pos = min(group.start)
            start_pos = max(0,pos-upstream)
            sequence = fasta_sequence[start_pos:pos+downstream]
            seq = SeqIO.SeqRecord(seq = Seq(sequence), id = name)
            list2struct.append([name,genes,ids,
                                start_pos,'+',str(seq.seq)])
            seqs.append(seq)
        elif all(group.strand == '-'):
            pos = max(group.stop)
            start_pos = max(0,pos-downstream)
            sequence = fasta_sequence[start_pos:pos+upstream]
            seq = SeqIO.SeqRecord(seq = Seq(sequence), id = name)
            list2struct.append([name,genes,ids,
                                start_pos,'-',str(seq.seq)])
            seqs.append(seq)
        else:
            raise ValueError('Operon contains genes on both strands:',name)
            
    DF_seqs = pd.DataFrame(list2struct,columns=['operon','genes','locus_tags','start_pos','strand','seq']).set_index('operon')

    # Add TRN info
    #tf_col = []
    #for genes in DF_seqs.locus_tags:
    #    tfs = []
    #    for gene in genes.split(','):
    #        tfs += ica_data.trn[ica_data.trn.gene_id == gene].TF.unique().tolist()
    #    tf_col.append(','.join(list(set(tfs))))
    #DF_seqs.loc[:,'TFs'] = tf_col

    # Run MEME
    if verbose:
        print('Finding motifs for {:d} sequences'.format(len(seqs)))
    if palindrome:
        comp_dir = 'motifs/' + re.sub('/','_','{}_pal'.format(k))
    else:
        comp_dir = 'motifs/' + re.sub('/','_',str(k))
    
    # Skip intensive tasks on rerun
    if force or not os.path.isdir(comp_dir):
    
        # Write sequence to file
        fasta = 'motifs/' + re.sub('/','_','{}.fasta'.format(k))
        SeqIO.write(seqs,fasta,'fasta')

        # Minimum number of total sites to find
        minsites = max(2,int(n_operons/3)) 
        
        cmd = ['meme',fasta,'-oc',comp_dir,
               '-dna','-mod','zoops','-p','8','-nmotifs',str(nmotifs),
               '-evt',str(evt),'-minw','6','-maxw',str(maxw),'-allw',
               '-minsites',str(minsites)]
        if palindrome:
            cmd.append('-pal')
        subprocess.call(cmd)

    # Save results
    result = parse_meme_output(comp_dir,DF_seqs,verbose=verbose,evt=evt)
    ica_data.motif_info[k] = result
    return(result)

def parse_meme_output(directory,DF_seqs,verbose=True,evt=0.001):

    # Read MEME results
    with open(directory+'/meme.xml','r') as f:
        result_file = BeautifulSoup(f.read(),'lxml')

    # Convert to motif XML file to dataframes: (overall,[individual_motif])
    DF_overall = pd.DataFrame(columns=['e_value','sites','width','consensus'])
    dfs = []
    for motif in result_file.find_all('motif'):

        # Motif statistics
        DF_overall.loc[motif['id'],'e_value'] = np.float64(motif['e_value'])
        DF_overall.loc[motif['id'],'sites']  = motif['sites']
        DF_overall.loc[motif['id'],'width']  = motif['width']
        DF_overall.loc[motif['id'],'consensus']  = motif['name']
        DF_overall.loc[motif['id'],'motif_name'] = motif['alt']

        # Map Sequence to name

        list_to_struct = []
        for seq in result_file.find_all('sequence'):
            list_to_struct.append([seq['id'],seq['name']])
        df_names = pd.DataFrame(list_to_struct,columns=['seq_id','operon'])

        # Get motif sites

        list_to_struct = []
        for site in motif.find_all('contributing_site'):

            site_seq = ''.join([letter['letter_id'] 
                                for letter in site.find_all('letter_ref')
                               ])
            data = [site['position'],site['pvalue'],site['sequence_id'],
                    site.left_flank.string,site_seq,site.right_flank.string]
            list_to_struct.append(data)
            
        tmp_df = pd.DataFrame(list_to_struct,columns=['rel_position','pvalue','seq_id',
                                                      'left_flank','site_seq','right_flank'])  

        # Combine motif sites with sequence to name mapper
        DF_meme = pd.merge(tmp_df,df_names)
        DF_meme = DF_meme.set_index('operon').sort_index().drop('seq_id',axis=1)
        DF_meme = pd.concat([DF_meme,DF_seqs],axis=1,sort=True)
        DF_meme.index.name = motif['id']
        
        # Report number of sequences with motif
        DF_overall.loc[motif['id'],'motif_frac'] = np.true_divide(sum(DF_meme.rel_position.notnull()),len(DF_meme))
        dfs.append(DF_meme)
        
    if len(dfs) == 0:
        if verbose:
            print('No motif found with E-value < {0:.1e}'.format(evt))
        return pd.DataFrame(columns=['e_value','sites','width','consensus','motif_frac']),[]
       
    return DF_overall,pd.concat({df.index.name:df for df in dfs})

def compare_motifs(k,motif_db,force=False,evt=.001):
    motif_file = 'motifs/' + re.sub('/','_',str(k)) + '/meme.txt'
    out_dir = 'motifs/' + re.sub('/','_',str(k))+ '/tomtom_out/'
    if not os.path.isdir(out_dir) or force:
        subprocess.call(['tomtom','-oc',out_dir,'-thresh',str(evt),'-incomplete-scores','-png',motif_file,motif_db])
    DF_tomtom = pd.read_csv(os.path.join(out_dir,'tomtom.tsv'),sep='\t',skipfooter=3,engine='python')
    
    if len(DF_tomtom) > 0:
        row = DF_tomtom.iloc[0]
        print(row['Target_ID'])
        tf_name = row['Target_ID'][:4].strip('_')
        lines = 'Motif similar to {} (E-value: {:.2e})'.format(tf_name,row['E-value'])
        files = out_dir+'align_'+row['Query_ID']+'_0_-'+row['Target_ID']+'.png'
        if not os.path.isfile(files):
            files = out_dir+'align_'+row['Query_ID']+'_0_+'+row['Target_ID']+'.png'
        with open(out_dir+'/tomtom.xml','r') as f:
            result_file = BeautifulSoup(f.read(),'lxml')
        motif_names = [motif['alt'] for motif in result_file.find('queries').find_all('motif')]
        idx = int(result_file.find('matches').query['idx'])
        
        return motif_names[idx],lines,files
    else:
        return -1,'',''
