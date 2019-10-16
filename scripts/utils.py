# utils module

import math as m
import numpy as np
from scipy import stats

def get_contig_len_gp_min_quantile_size(lengp_minsize, numseqs, min_quantile = 0.0025):
  quantile = max(lengp_minsize/numseqs, min_quantile) * 100
  if quantile > min_quantile * 100:
      quantile = 100 / m.floor(100 / quantile) # (Basically) the next smallest possible equally sized bins
  return quantile

def get_contig_length_gps(seqs, seqs_len):
  bin_minsize = min(500, seqs.shape[0])
  min_quantile_size = get_contig_len_gp_min_quantile_size(bin_minsize, seqs.shape[0])
  length_gps = []
  ub = np.Inf
  len_percentiles_uniq = np.unique(np.percentile(seqs_len.values, np.arange(min_quantile_size, 100, min_quantile_size), interpolation='higher'))
  lb_idx = len(len_percentiles_uniq) - 1
  while len(seqs[seqs_len < ub]) >= bin_minsize:
      while len(seqs[(seqs_len < ub) & (seqs_len >= len_percentiles_uniq[lb_idx])]) < bin_minsize:
          lb_idx -= 1
      i = 1
      base_gp = seqs[(seqs_len < ub) & (seqs_len >= len_percentiles_uniq[lb_idx])].mean_kmer_depth.values
      while stats.ks_2samp(base_gp, seqs[(seqs_len < ub) & (seqs_len >= len_percentiles_uniq[lb_idx - i])].mean_kmer_depth.values).pvalue > 0.05:
          i += 1
          if lb_idx < i:
              break
      if lb_idx < i:
          lb = -np.inf
      else:
          lb = len_percentiles_uniq[lb_idx - i + 1]
          if len(seqs[seqs_len < lb]) < len(seqs[(seqs_len < ub) & (seqs_len >= lb)]): # unlikely; hopefully never happens
              lb = -np.inf
      length_gp = seqs[(seqs_len < ub) & (seqs_len >= lb)]
      length_gps.append(length_gp)
      bin_minsize = len(length_gp)
      ub = lb
  length_gps.reverse()
  return length_gps

