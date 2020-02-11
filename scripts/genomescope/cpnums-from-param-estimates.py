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
cvg_frequencies.loc[0.5] = hist.index.map(lambda k: ((2 * (1-d) * (1 - (1-r)**k)) + (2 * d * (1 - (1-r)**k) ** 2) + (2 * d * ((1-r)**k) * (1 - (1-r)**k))) * stats.nbinom.pmf(k, n, p) * length)
cvg_frequencies.loc[1] = hist.index.map(lambda k: (((1-d) * (1-r)**k) + (d * (1 - (1-r)**k) ** 2)) * stats.nbinom.pmf(k, 2*n, p) * length)
cvg_frequencies.loc[2] = hist.index.map(lambda k: ((2 * d * ((1-r)**k) * (1 - (1-r)**k)) * stats.nbinom.pmf(k, 3*n, p) + (d * (1-r)**(2*k)) * stats.nbinom.pmf(k, 4*n, p)) * length)

copynum_assnmts, copynum_lbs, copynum_ubs = utils.get_cpnums_and_bounds(cvg_frequencies, copynums)
cvg_frequencies.loc[0] = hist.freq - cvg_frequencies.loc[0.5:].sum()
ub0 = copynum_ubs[copynum_ubs < np.inf].iloc[0]
cvg_frequencies.loc[0, ub0:] = 0
maxdensity_cpnums = cvg_frequencies.loc[:, :ub0].idxmax()
likeliest_cpnum_ub_idxs = utils.compute_likeliest_copynum_indices(maxdensity_cpnums)
likeliest_cpnums = utils.compute_likeliest_copynums(maxdensity_cpnums, likeliest_cpnum_ub_idxs)
zero_to_next = ((likeliest_cpnums[:-1] == 0).values & (likeliest_cpnums[1:] == copynum_assnmts[0]).values)
if (np.argwhere(zero_to_next).size == 0) and (likeliest_cpnums[0] == 0):
    zero_to_next = ((likeliest_cpnums[:-1] > copynum_assnmts[0]).values & (likeliest_cpnums[1:] == copynum_assnmts[0]).values)
if np.argwhere(zero_to_next).size:
    boundary = maxdensity_cpnums.index[likeliest_cpnum_ub_idxs[np.argwhere(zero_to_next)[0][0]]]
    copynum_lbs[0], copynum_ubs[0], copynum_lbs[copynum_assnmts[0]] = 0, boundary, boundary
    copynum_assnmts.insert(0, 0)

if args.max_cpnum_3:
  lb3 = copynum_lbs[copynum_lbs < np.inf].iloc[-1]
  cvg_frequencies.loc[3] = hist.freq - cvg_frequencies.loc[0.5:2].sum()
  cvg_frequencies.loc[3, :lb3] = 0
  maxdensity_cpnums = cvg_frequencies.loc[:, lb3:].idxmax()
  first_3_idx = maxdensity_cpnums.index[maxdensity_cpnums == 3][0]
  the_rest = maxdensity_cpnums[first_3_idx:]
  the_rest[the_rest == 0] = 3
  likeliest_cpnum_ub_idxs = utils.compute_likeliest_copynum_indices(maxdensity_cpnums)
  likeliest_cpnums = utils.compute_likeliest_copynums(maxdensity_cpnums, likeliest_cpnum_ub_idxs)
  prev_to_three = ((likeliest_cpnums[:-1] == copynum_assnmts[-1]).values & (likeliest_cpnums[1:] == 3).values)
  if (np.argwhere(prev_to_three).size == 0) and (3 in likeliest_cpnums.values):
      prev_to_three = ((likeliest_cpnums[:-1] < copynum_assnmts[-1]).values & (likeliest_cpnums[1:] == copynum_assnmts[0]).values)
  if np.argwhere(prev_to_three).size:
      boundary = maxdensity_cpnums.index[likeliest_cpnum_ub_idxs[np.argwhere(prev_to_three)[0][0]]]
      copynum_ubs[copynum_assnmts[-1]], copynum_lbs[3] = boundary, boundary
      copynum_assnmts.append(3)

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
    seqs.loc['likeliest_copynum'] = copynum_assnmts[0]
for i in range(1, len(copynum_assnmts)):
    depth_condition = (seqs.mean_kmer_depth >= copynum_lbs[copynum_assnmts[i]]) & (seqs.mean_kmer_depth < copynum_ubs[copynum_assnmts[i]])
    seqs.loc[depth_condition, 'likeliest_copynum'] = copynum_assnmts[i]

seqs.loc[:, 'length':].to_csv(args.gs_output_dir + '/' + args.output_dir_name + '/sequence-labels.csv',
    header=['Length', 'Average k-mer depth', '1st Mode X', 'GC %', 'Estimation length group', 'Likeliest copy #'], index_label='ID')
