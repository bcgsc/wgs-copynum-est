import array
import argparse
import os
import numpy as np
import pandas as pd
import re
import sys
from scipy import stats

if os.getenv('WGS_COPYNUM_EST_HOME'):
  sys.path.insert(0, os.path.join(os.getenv('WGS_COPYNUM_EST_HOME'), 'scripts'))
else:
  raise RuntimeError('Please set environment variable WGS_COPYNUM_EST_HOME before running script')

import utils


argparser = argparse.ArgumentParser(description='Parse GenomeScope parameter estimates, assign implied genomic copy number to k-length assembly sequences, and write output')
argparser.add_argument('kmer_len', type=int, help='Value of k used by GenomeScope, and in assembly that output sequences to be classified')
argparser.add_argument('--max_cpnum_3', action='store_true', help='Classify up to copy number 3 (instead of default of 2)')
argparser.add_argument('unitigs_file', type=str, help='FASTA file listing sequences to be classified')
argparser.add_argument('ntcard_hist_file', type=str, help='ntCard output histogram file, needed for determining boundary between copy numbers 2 and 3')
argparser.add_argument('gs_output_dir', type=str, help='GenomeScope output folder')
argparser.add_argument('output_dir_name', type=str, help='Folder (within GenomeScope output folder) to which output files should be written')
args = argparser.parse_args()

ntcard_stats = pd.read_csv(args.ntcard_hist_file, sep='\t', header=None)
hist = ntcard_stats.iloc[2:]
hist.rename(columns = { 0: 'cvg', 1: 'freq' }, inplace=True)
hist['cvg'] = hist.cvg.astype(int)
hist.set_index('cvg', inplace=True)

model_file = open(args.gs_output_dir + '/model.txt', newline='')
row = model_file.readline()
while not(re.match('^Parameters', row)):
    row = model_file.readline()
row = model_file.readline()
d = float(model_file.readline().split()[1])
r = float(model_file.readline().split()[1])
mu = float(model_file.readline().split()[1])
bias = float(model_file.readline().split()[1])
length = float(model_file.readline().split()[1])
n = mu / bias
p = 1 / (1 + bias)

copynums = [0, 0.5, 1, 2, 3]
cvg_frequencies = pd.DataFrame(0, index=copynums, columns=hist.index)
k = 1.0
half_wt = ((2 * (1-d) * (1 - (1-r)**k)) + (2 * d * (1 - (1-r)**k) ** 2) + (2 * d * ((1-r)**k) * (1 - (1-r)**k)))
cvg_frequencies.loc[0.5] = hist.index.map(lambda k: half_wt * stats.nbinom.pmf(k, n, p) * length)
cvg_frequencies.loc[1] = hist.index.map(lambda k: (((1-d) * (1-r)**k) + (d * (1 - (1-r)**k) ** 2)) * stats.nbinom.pmf(k, 2*n, p) * length)
wt2 = (2 * d * ((1-r)**k) * (1 - (1-r)**k)) + (d * (1-r)**(2*k))
cvg_frequencies.loc[2] = hist.index.map(lambda k: ((2 * d * ((1-r)**k) * (1 - (1-r)**k)) * stats.nbinom.pmf(k, 3*n, p) + (d * (1-r)**(2*k)) * stats.nbinom.pmf(k, 4*n, p)) * length)

copynum_assnmts, copynum_lbs, copynum_ubs = utils.get_cpnums_and_bounds(cvg_frequencies, copynums)
cvg_frequencies.loc[0] = hist.freq - cvg_frequencies.loc[0.5:].sum()
utils.impute_lowest_cpnum_and_bds(cvg_frequencies, 0, copynum_assnmts, copynum_lbs, copynum_ubs,
    stats.nbinom.mean(n, p) - stats.nbinom.std(n, p) - half_wt)

if args.max_cpnum_3:
  cvg_frequencies.loc[3] = hist.freq - cvg_frequencies.loc[0.5:2].sum()
  # Assume copy #s 1.5 and 2 uncorrelated for simplicity
  utils.impute_highest_cpnum_and_bds(cvg_frequencies, 3, copynum_assnmts, copynum_lbs, copynum_ubs,
      stats.nbinom.mean(3*n, p) + stats.nbinom.mean(4*n, p) + 2 * (stats.nbinom.var(3*n, p) + stats.nbinom.var(4*n, p)) ** 0.5 + wt2)

seq_IDs = array.array('L')
seq_lens = array.array('L')
seq_mean_kmer_depths = array.array('d')
seq_gc_contents = array.array('d')

with open(args.unitigs_file) as unitigs:
    line = unitigs.readline()
    while line:
        if re.search('^>[0-9]', line):
            row = list(map(int, line[1:].split()))
            seq_IDs.append(row[0])
            seq_lens.append(row[1])
            kmers = row[1] - args.kmer_len + 1
            seq_mean_kmer_depths.append(row[2] / kmers)
        else:
            seq_gc_contents.append(utils.compute_gc_content(line))
        line = unitigs.readline()

numseqs = len(seq_mean_kmer_depths)
seqs = pd.DataFrame(columns=['ID', 'length', 'mean_kmer_depth', 'modex', 'gc', 'est_gp', 'likeliest_copynum'])
seqs['ID'] = seq_IDs
seqs['length'] = seq_lens
seqs['mean_kmer_depth'] = seq_mean_kmer_depths
seqs['gc'] = seq_gc_contents
seqs['est_gp'] = -1
seqs['likeliest_copynum'] = -1.0
seqs.set_index('ID', inplace=True)
seqs.sort_values(by=['length', 'mean_kmer_depth'], inplace=True)
seqs = seqs.loc[seqs.length == args.kmer_len]

seqs['est_gp'] = 0
# Note: copynum_lbs[i] == copynum_ubs[i-1]
if len(copynum_assnmts) > 1:
    seqs.loc[(seqs.mean_kmer_depth < copynum_lbs[copynum_assnmts[1]]), 'likeliest_copynum'] = copynum_assnmts[0]
else:
    seqs['likeliest_copynum'] = copynum_assnmts[0]
for i in range(1, len(copynum_assnmts)):
    depth_condition = (seqs.mean_kmer_depth >= copynum_lbs[copynum_assnmts[i]]) & (seqs.mean_kmer_depth < copynum_ubs[copynum_assnmts[i]])
    seqs.loc[depth_condition, 'likeliest_copynum'] = copynum_assnmts[i]

seqs.loc[:, 'length':].to_csv(args.gs_output_dir + '/' + args.output_dir_name + '/sequence-labels.csv',
    header=['Length', 'Average k-mer depth', '1st Mode X', 'GC %', 'Estimation length group', 'Likeliest copy #'], index_label='ID')
