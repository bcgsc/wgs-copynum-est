# utils module

from itertools import chain
import math as m
import numpy as np
import pandas as pd
from scipy import stats


def compute_gc_content(seq):
  gc_count = 0
  for b in seq:
    if b == 'G' or b == 'C':
      gc_count += 1
  return (gc_count * 100 / len(seq))

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
  if len(len_percentiles_uniq) < 1:
      return [seqs]
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

def compute_likeliest_copynum_indices(maxdensity_copynums):
  maxdens_cpnums_change_idxs = np.array([-1])
  maxdens_cpnums_change_idxs = np.append(maxdens_cpnums_change_idxs, np.where(np.diff(maxdensity_copynums))[0])
  maxdens_cpnums_change_idxs += 1
  return maxdens_cpnums_change_idxs

def compute_likeliest_copynums(maxdensity_copynums, maxdens_cpnums_change_idxs):
  return maxdensity_copynums.iloc[maxdens_cpnums_change_idxs].values

def assign_copynums_and_bounds(likeliest_copynums, likeliest_copynum_ubs, copynums):
  copynum_assnmts, copynums_unique = [likeliest_copynums[0]], { likeliest_copynums[0] }
  # Note: 0 does not have an entry in copynum_lbs and copynum_ubs.
  copynum_lbs = pd.Series(np.inf, index=copynums)
  copynum_ubs = pd.Series(np.inf, index=copynums)
  copynum_lbs[copynum_assnmts[0]] = 0
  # Initial assignments might be slightly out of order: have to infer orderly final assignments
  # Assume that 1. Larger copy#s don't occur in order before smaller copy#s, e.g. 1, 3, 4, 2 does not occur
  # (whereas 1, 4, 3, 2 could [copy#4 variance is larger than that of 3])
  # 2a. Out-of-order copy#s are always followed eventually by in-order copy#s, e.g.: 1, 3, 2 will be followed eventually by a copy# >= 3
  # b. With one possible exception at the end (accounted for by the last "if" clause)
  reserve_copynum, reserve_bd = 0, np.inf
  for i in range(1, len(likeliest_copynums)):
    if likeliest_copynums[i] not in copynums_unique:
      if likeliest_copynums[i] < copynum_assnmts[-1]:
        reserve_copynum, reserve_bd = copynum_assnmts[-1], copynum_lbs[copynum_assnmts[-1]]
        copynum_lbs[copynum_assnmts[-1]] = np.inf
        copynums_unique.remove(copynum_assnmts[-1])
        copynum_assnmts.pop()
      if len(copynum_assnmts) == 0:
        copynum_lbs[likeliest_copynums[i]] = 0
      else:
        copynum_lbs[likeliest_copynums[i]] = likeliest_copynum_ubs[i-1]
        copynum_ubs[copynum_assnmts[-1]] = likeliest_copynum_ubs[i-1]
      copynum_assnmts.append(likeliest_copynums[i])
      copynums_unique.add(likeliest_copynums[i])
  # To keep from putting upper bound of penultimate cpnum at intersection between higher copy number and tail of lower copy number.
  # Ideally, would record and check whether reserve_copynum and copynum_assnmts[-1] were assigned in the same iteration
  # (the current order seems likelier to be correct if multiple copy numbers were assigned after the reserve), but
  # this already works as well as or better than omitting it (in like 29/30 cases), and changing it would require a whole bunch more testing
  if len(copynum_assnmts) > 1 and (copynum_assnmts[-1] < reserve_copynum):
    copynum_lbs[copynum_assnmts[-1]] = np.inf
    copynum_assnmts.pop()
    copynum_lbs[reserve_copynum] = reserve_bd
    copynum_ubs[copynum_assnmts[-1]] = reserve_bd
    copynum_assnmts.append(reserve_copynum)
  return (copynum_assnmts, copynum_lbs, copynum_ubs)

def get_cpnums_and_bounds(copynum_densities, copynums):
  maxdensity_copynums = copynum_densities.idxmax()
  likeliest_copynum_bd_idxs = compute_likeliest_copynum_indices(maxdensity_copynums)
  likeliest_copynums = compute_likeliest_copynums(maxdensity_copynums, likeliest_copynum_bd_idxs)
  likeliest_copynum_bds = copynum_densities.columns[likeliest_copynum_bd_idxs]
  return assign_copynums_and_bounds(likeliest_copynums, likeliest_copynum_bds[1:], copynums)

def wide_to_long_from_listcol(df, variable_colname):
  obs_counts = [len(x) for x in chain.from_iterable(df[variable_colname])]
  df_long = pd.DataFrame(index=range(sum(obs_counts)), columns=df.columns)
  for col in df.columns.tolist():
    if col != variable_colname:
      df_long[col] = np.repeat(df[col].values, obs_counts)
    else:
      df_long[variable_colname] = list(chain.from_iterable(chain.from_iterable(df[variable_colname])))
  return df_long

def valcounts_str_to_ints_list(dist_counts_str):
  if type(dist_counts_str) is str:
    return [list(chain.from_iterable(list(map(lambda pair: [int(pair[0])] * int(pair[1]), map(lambda dcount: dcount.split(':'), dist_counts_str.split(','))))))]
  return [[np.nan]]

