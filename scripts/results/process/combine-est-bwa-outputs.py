import csv
import math as m
import numpy as np
import pandas as pd
import sys

EST_OUTPUT = sys.argv[1]
EST_LEN_GP_STATS = sys.argv[2]
BWA_PARSE_OUTPUT = sys.argv[3]

def count_and_write(max_components_est, seqs, cat=None):
  est_components = list(range(max_components_est + 1))
  cols = [0, 'avg_avg_depths_' + str(0), 'avg_gc_' + str(0)]
  for i in est_components[1:]:
    cols.extend([i, 'avg_avg_depths_' + str(i), 'avg_gc_' + str(i)])
  aln_est = pd.DataFrame(None, index=est_components, columns=cols)
  for i in est_components:
    for j in est_components:
      if cat is not None:
        seqs_ij = seqs[(seqs.length >= mins[cat]) & (seqs.length < ubs[cat]) & (seqs.apply(lambda seq: min(seq['aln_match_count'], seq['len_gp_est_components']), axis=1) == i) & (seqs.copynum_est == j)]
      else:
        seqs_ij = seqs[(seqs.apply(lambda seq: min(seq['aln_match_count'], seq['len_gp_est_components']), axis=1) == i) & (seqs.copynum_est == j)]
      aln_est.loc[i, j] = seqs_ij.shape[0]
      if aln_est.loc[i, j] > 0:
        aln_est.loc[i, 'avg_avg_depths_' + str(j)] = seqs_ij.avg_depth.sum() / aln_est.loc[i, j]
        aln_est.loc[i, 'avg_gc_' + str(j)] = seqs_ij.GC.sum() / aln_est.loc[i, j]
  header = [0, 'Avg avg depth ' + str(0), 'Avg GC content ' + str(0)]
  for i in est_components[1:]:
    header.extend([str(i) + ' (+)', 'Avg avg depth ' + str(i), 'Avg GC content ' + str(i)])
  idx_dict = {}
  for i in est_components[1:]:
    idx_dict[i] = str(i) + ' (+)'
  aln_est.rename(index=idx_dict)
  if cat is not None:
    aln_est.to_csv(count_files[cat], header=header, index_label='Aln/Est')
  else:
    aln_est.to_csv('aln-est_counts.csv', header=header, index_label='Aln/Est')


len_gp_est_component_counts = pd.read_csv(EST_LEN_GP_STATS)['Largest diploid copy # estimated'].apply(lambda i: m.ceil(i/2.0)).tolist()
max_components_est = int(max(len_gp_est_component_counts))

seqs = pd.read_csv(EST_OUTPUT)
cols_from_to = { 'Length': 'length', 'Average k-mer depth': 'avg_depth', '1st Mode X': 'modex', 'GC %': 'GC', 'Estimation length group': 'len_gp', 'Likeliest copy #': 'copynum_est' }
seqs.rename(columns=cols_from_to, inplace=True)
seqs.set_index('ID', inplace=True)
seqs['len_gp_est_components'] = seqs.len_gp.apply(lambda gp: len_gp_est_component_counts[gp])

seq_alns = pd.read_csv(BWA_PARSE_OUTPUT, delimiter='\t')
seq_alns.drop('GC content', axis=1, inplace=True)
cols_from_to = { 'Mapped': 'aln_mapped', 'Matches': 'aln_match_count', 'Others': 'aln_other_count', 'Other CIGARs': 'aln_other_cigars', 'MAPQ (unique match only)': 'aln_mapq' }
seq_alns.rename(columns=cols_from_to, inplace=True)
seq_alns.set_index('ID', inplace=True)
seq_alns['aln_match_count'] = seq_alns.aln_match_count.apply(lambda i: m.ceil(i/2.0))
seqs = seqs.join(seq_alns)
seqs.sort_values(by=['length', 'avg_depth'], inplace=True)

write_cols = ['length', 'avg_depth', 'GC', 'copynum_est', 'aln_match_count', 'aln_other_count', 'aln_other_cigars', 'aln_mapq']
header = ['Length', 'Average depth', 'GC %', 'Likeliest copy #', 'Alignments (alns)', 'Other alns', 'Other-aln CIGARs', 'MAPQ (unique aln only)']
seqs.loc[:, write_cols].to_csv('seq-est-and-aln.csv', header=header, index_label='ID')

mins = [0, 100, 1000, 10000]
ubs = [100, 1000, 10000, np.inf]
count_files = ['aln-est_counts_lt100.csv', 'aln-est_counts_lt1000.csv', 'aln-est_counts_lt10000.csv', 'aln-est_counts_gte10000.csv']

for cat in range(len(mins)):
  count_and_write(max_components_est, seqs, cat)

count_and_write(max_components_est, seqs)

