import argparse
import csv
import math as m
import numpy as np
import pandas as pd
import sys

argparser = argparse.ArgumentParser(description="Combine BWA alignment and copy number estimator outputs for classified sequences")
argparser.add_argument("est_output", type=str, help="CSV file listing sequence data and classifications")
argparser.add_argument("est_len_gp_stats", type=str, help="CSV file listing sequence length groups used in classification, with summary statistics")
argparser.add_argument("bwa_parse_output", type=str, help="BWA alignment output SAM file")
args = argparser.parse_args()

def count_and_write(max_components_est, seqs, total_seqs, cat=None):
  est_components = [0]
  if HALF and (max_components_est > 0):
    est_components.append(0.5)
  if max_components_est >= 1:
    est_components = est_components + list(range(1, max_components_est + 1))
  cols = [0, 'avg_avg_depths_' + str(0), 'avg_gc_' + str(0)]
  for i in est_components[1:]:
    cols.extend([i, 'avg_avg_depths_' + str(i), 'avg_gc_' + str(i)])
  aln_est = pd.DataFrame(None, index=est_components, columns=cols)
  total = 0
  if cat is not None:
    seqs = seqs[(seqs.length >= mins[cat]) & (seqs.length < ubs[cat])]
  for i in est_components:
    count = 0
    for j in est_components:
      seqs_ij = seqs[(((seqs.len_gp_est_components >= i) & (seqs.aln_match_count == i)) | ((seqs.len_gp_est_components == i) & (seqs.aln_match_count > i))) & (seqs.copynum_est == j)]
      aln_est.loc[i, j] = seqs_ij.shape[0]
      count += aln_est.loc[i, j]
      if aln_est.loc[i, j] > 0:
        aln_est.loc[i, 'avg_avg_depths_' + str(j)] = seqs_ij.avg_depth.sum() / aln_est.loc[i, j]
        aln_est.loc[i, 'avg_gc_' + str(j)] = seqs_ij.GC.sum() / aln_est.loc[i, j]
    total += count
  assert aln_est[est_components].sum().sum() == seqs.shape[0], "Number of enumerated sequences [in length group] differs from number of input sequences!"
  header = []
  for i in est_components:
    header.extend([str(i) + ' (+)', 'Avg avg depth ' + str(i), 'Avg GC content ' + str(i)])
  idx_dict = {}
  for i in est_components[1:]:
    idx_dict[i] = str(i) + ' (+)'
  aln_est.rename(index=idx_dict)
  if cat is not None:
    aln_est.to_csv(count_files[cat], header=header, index_label='Aln\Est')
  else:
    aln_est.to_csv('aln-est_counts.csv', header=header, index_label='Aln\Est')


seqs = pd.read_csv(args.est_output)
TOTAL_SEQUENCES = seqs.shape[0]
cols_from_to = { 'Length': 'length', 'Average k-mer depth': 'avg_depth', '1st Mode X': 'modex', 'GC %': 'GC', 'Estimation length group': 'len_gp', 'Likeliest copy #': 'copynum_est' }
seqs.rename(columns=cols_from_to, inplace=True)
seqs.set_index('ID', inplace=True)
HALF = ((seqs.copynum_est == 0.5).sum() > 0)

len_gp_est_component_counts = pd.read_csv(args.est_len_gp_stats)['Largest diploid copy # estimated'].apply(lambda i: m.ceil(i/2.0) if (i > 1 or not(HALF)) else 0.5 if i == 0.5 else 0).tolist()
max_components_est = max(len_gp_est_component_counts)
if max_components_est > 0.5:
  max_components_est = int(max_components_est)

seqs['len_gp_est_components'] = seqs.len_gp.apply(lambda gp: len_gp_est_component_counts[gp])

seq_alns = pd.read_csv(args.bwa_parse_output, delimiter='\t')
seq_alns.drop('GC content', axis=1, inplace=True)
cols_from_to = { 'Mapped': 'aln_mapped', 'Matches': 'aln_match_count', 'Others': 'aln_other_count', 'Other CIGARs': 'aln_other_cigars', 'MAPQ (unique match only)': 'aln_mapq' }
seq_alns.rename(columns=cols_from_to, inplace=True)
seq_alns.set_index('ID', inplace=True)

seq_alns['aln_match_count'] = seq_alns.aln_match_count.apply(lambda i: m.ceil(i/2.0) if (i > 1 or not(HALF)) else 0.5 if i == 1 else 0)
seqs = seqs.join(seq_alns)
seqs.loc[seqs.aln_match_count.isna(), 'aln_match_count'] = 0
seqs.sort_values(by=['length', 'avg_depth'], inplace=True)

write_cols = ['length', 'avg_depth', 'GC', 'copynum_est', 'aln_match_count', 'aln_other_count', 'aln_other_cigars', 'aln_mapq']
header = ['Length', 'Average depth', 'GC %', 'Likeliest copy #', 'Alignments (alns)', 'Other alns', 'Other-aln CIGARs', 'MAPQ (unique aln only)']
seqs.loc[:, write_cols].to_csv('seq-est-and-aln.csv', header=header, index_label='ID')

mins = [0, 100, 1000, 10000]
ubs = [100, 1000, 10000, np.inf]
count_files = ['aln-est_counts_lt100.csv', 'aln-est_counts_lt1000.csv', 'aln-est_counts_lt10000.csv', 'aln-est_counts_gte10000.csv']

for cat in range(len(mins)):
  count_and_write(max_components_est, seqs, TOTAL_SEQUENCES, cat)

est_components = [0]
if HALF and (max_components_est > 0):
  est_components.append(0.5)
if max_components_est >= 1:
  est_components = est_components + list(range(1, max_components_est + 1))
count = 0
for i in est_components:
  count += (seqs.aln_match_count == i).sum()
count += (seqs.aln_match_count > est_components[-1]).sum()

count_and_write(max_components_est, seqs, TOTAL_SEQUENCES)

