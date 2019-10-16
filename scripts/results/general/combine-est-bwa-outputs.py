import argparse
import csv
import math as m
import numpy as np
import pandas as pd
import sys

argparser = argparse.ArgumentParser(description="Combine BWA alignment and copy number estimator outputs for classified sequences")
argparser.add_argument('--use_est_len_gps', action="store_true", help='Divide sequences by length groups used in estimation')
argparser.add_argument("est_output", type=str, help="CSV file listing sequence data and classifications")
argparser.add_argument("est_len_gp_stats", type=str, help="CSV file listing sequence length groups used in classification, with summary statistics")
argparser.add_argument("bwa_parse_output", type=str, help="Parsed contig data from BWA reference alignment output SAM file")
args = argparser.parse_args()

def count_and_write(max_components_est, seqs, filename):
  est_components = [0]
  if HALF and (max_components_est > 0):
    est_components.append(0.5)
  if max_components_est >= 1:
    est_components = est_components + list(range(1, max_components_est + 1))
  cols = [0, 'avg_avg_depths_' + str(0), 'avg_gc_' + str(0)]
  for i in est_components[1:]:
    cols.extend([i, 'avg_avg_depths_' + str(i), 'avg_gc_' + str(i)])
  aln_est = pd.DataFrame(None, index=est_components, columns=cols)
  for i in est_components:
    for j in est_components:
      seqs_ij = seqs[(((seqs.len_gp_est_components >= i) & (seqs.aln_match_count == i)) | ((seqs.len_gp_est_components == i) & (seqs.aln_match_count > i))) & (seqs.copynum_est == j)]
      aln_est.loc[i, j] = seqs_ij.shape[0]
      if aln_est.loc[i, j] > 0:
        aln_est.loc[i, 'avg_avg_depths_' + str(j)] = seqs_ij.avg_depth.sum() / aln_est.loc[i, j]
        aln_est.loc[i, 'avg_gc_' + str(j)] = seqs_ij.GC.sum() / aln_est.loc[i, j]
  assert aln_est[est_components].sum().sum() == seqs.shape[0], "Number of enumerated sequences [in length group] differs from number of input sequences!"
  header = []
  for i in est_components:
    header.extend([str(i) + ' (+)', 'Avg avg depth ' + str(i), 'Avg GC content ' + str(i)])
  idx_dict = {}
  for i in est_components[1:]:
    idx_dict[i] = str(i) + ' (+)'
  aln_est.rename(index=idx_dict)
  aln_est.to_csv(filename, header=header, index_label='Aln\Est')

def get_best_alnmt_count(dist_counts_str):
  if type(dist_counts_str) is str:
    return int(dist_counts_str.split(',')[0].split(':')[1])
  return 0


seqs = pd.read_csv(args.est_output)
cols_from_to = { 'Length': 'length', 'Average k-mer depth': 'avg_depth', '1st Mode X': 'modex', 'GC %': 'GC', 'Estimation length group': 'len_gp', 'Likeliest copy #': 'copynum_est' }
seqs.rename(columns=cols_from_to, inplace=True)
seqs.set_index('ID', inplace=True)
HALF = ((seqs.copynum_est == 0.5).sum() > 0)

len_gp_stats = pd.read_csv(args.est_len_gp_stats)
len_gp_est_component_counts = len_gp_stats['Largest copy # estimated'].tolist()
max_components_est = max(len_gp_est_component_counts)
if max_components_est > 0.5:
  max_components_est = int(max_components_est)

seqs['len_gp_est_components'] = seqs.len_gp.apply(lambda gp: len_gp_est_component_counts[m.floor(gp)])
del len_gp_est_component_counts

seq_alns = pd.read_csv(args.bwa_parse_output, delimiter='\t')
seq_alns.drop(['Length', 'GC content'], axis=1, inplace=True)
cols_from_to = { 'Length': 'length', 'Matches': 'matches', 'Clipped': 'clipped', 'MAPQ sum': 'mapq_sum', 'Edit distance': 'edit_dist' }
seq_alns.rename(columns=cols_from_to, inplace=True)
seq_alns.set_index('ID', inplace=True)
seqs = seqs.join(seq_alns)

seqs['aln_match_count'] = seqs.edit_dist.apply(lambda dist_counts_str: get_best_alnmt_count(dist_counts_str))
seqs['aln_match_count'] = seqs.aln_match_count.apply(lambda i: m.ceil(i/2.0) if (i != 1 or not(HALF)) else 0.5) # alnmt counts still diploid
seqs.sort_values(by=['length', 'avg_depth'], inplace=True)

write_cols = ['length', 'avg_depth', 'GC', 'copynum_est', 'aln_match_count', 'mapq_sum']
header = ['Length', 'Average depth', 'GC %', 'Likeliest copy #', 'Alignments (alns)', 'MAPQ sum']
seqs.loc[:, write_cols].to_csv('seq-est-and-aln.csv', header=header, index_label='ID')

if args.use_est_len_gps:
  mins, maxs = len_gp_stats['Min. len.'].values, len_gp_stats['Max. len.'].values
  len_gp_conditions = list(map(lambda bds: (seqs.length >= bds[0]) & (seqs.length <= bds[1]), zip(mins, maxs)))
  count_files = ['aln-est_counts_'] * len_gp_stats.shape[0]
  for i in range(len(mins)):
    ub_str = 'lt' + (maxs[i] < np.inf) * 'e' + str(maxs[i])
    count_files[i] = count_files[i] + 'gte' + str(mins[i]) + ub_str + '.csv'
else:
  mins = [0, 100, 1000, 10000]
  maxs = [99, 999, 9999, np.inf]
  len_gp_conditions = list(map(lambda bds: (seqs.length >= bds[0]) & (seqs.length <= bds[1]), zip(mins, maxs)))
  count_files = ['aln-est_counts_gte0lte99.csv', 'aln-est_counts_gte100lte999.csv', 'aln-est_counts_gte1000lte9999.csv', 'aln-est_counts_gte10000ltinf.csv']

for i in range(len(len_gp_conditions)):
  count_and_write(max_components_est, seqs.loc[len_gp_conditions[i]], count_files[i])
count_and_write(max_components_est, seqs, 'aln-est_counts.csv')

