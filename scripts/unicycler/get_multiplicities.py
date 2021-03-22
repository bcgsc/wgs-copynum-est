import array
import argparse
import os
import pandas as pd
import re
import sys

if os.getenv('WGS_COPYNUM_EST_HOME'):
  sys.path.insert(0, os.path.join(os.getenv('WGS_COPYNUM_EST_HOME'), 'scripts'))
else:
  raise RuntimeError('Please set environment variable WGS_COPYNUM_EST_HOME before running script')

import utils.utils as utils


def set_sequence_multiplicity(seq_multiplicity, seqs):
    parsed = re.match('(\d+)\((\d+)x\)', seq_multiplicity)
    if parsed:
        seqs.loc[int(parsed[1]), 'likeliest_copynum'] = int(parsed[2])


argparser = argparse.ArgumentParser(description="Write SPAdes contig data (multiplicities from unicycler.log, the rest from FASTA file) into CSV format consistent with that used in this project.")
argparser.add_argument("fasta_file", type=str, help="Contig FASTA file")
argparser.add_argument("multiplicities_file", type=str, help="Contig multiplicities text file")
argparser.add_argument("output_dir", type=str, help="Output folder")
args = argparser.parse_args()

seqs = utils.seqs_from_abyss_contigs(args.fasta_file)
seqs.sort_values(by=['length', 'mean_kmer_depth'], inplace=True)

mult_file = open(args.multiplicities_file)
for row in mult_file:
    row = row.strip()
    junction = row.split(' \u2192 ')
    if len(junction) == 1:
        junction = list(map(lambda paths: paths.strip(), row.split(' -> ')))
    if len(junction) == 1: # split over 2 lines, e.g. '414(5x) + 406(4x) + 418(7x)\n + 423(9x) → 278(25x)'
        junction = [row]
    if re.match('\+ ', junction[0]):
        junction[0] = junction[0][2:]
    for contig_multiplicity in junction[0].split(' + '):
        set_sequence_multiplicity(contig_multiplicity, seqs)
    if len(junction) > 1:
        set_sequence_multiplicity(junction[1], seqs)

seqs.loc[seqs.likeliest_copynum < 0, 'likeliest_copynum'] = 0
seqs.loc[:, 'length':].to_csv(args.output_dir + '/sequence-labels.csv',
    header=['Length', 'Average k-mer depth', '1st Mode X', 'GC %', 'Estimation length group', 'Likeliest copy #'], index_label='ID')
