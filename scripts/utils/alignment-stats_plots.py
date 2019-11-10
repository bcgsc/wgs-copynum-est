import array
import argparse
import csv
from itertools import chain
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
from scipy import stats
import seaborn as sns
import sys

if os.getenv('WGS_COPYNUM_EST_HOME'):
  sys.path.insert(0, os.path.join(os.getenv('WGS_COPYNUM_EST_HOME'), 'scripts'))
else:
  raise RuntimeError('Please set environment variable WGS_COPYNUM_EST_HOME before running script')

import utils


def max_pctl_for_plotting(obs):
  return np.percentile(obs, 100 - stats.variation(obs) * 1.5)

def plot_kde(obs, obs_maxval = np.inf):
  return sns.kdeplot(obs[obs <= obs_maxval])

def do_scatterplot(data, x, y, xlab, ylab, filepath):
  ax = sns.scatterplot(x = x, y = y, data = data)
  ax.set(xlabel = xlab, ylabel = ylab)
  ax.get_figure().savefig(filepath)
  plt.clf()


argparser = argparse.ArgumentParser(description='Plot alignment feature (e.g. edit distance, perfect match) distributions by sequence length and depth, and optionally by estimated copy number')
argparser.add_argument('kmer_len', type=int, help='Value of k used in assembly that produced contigs')
argparser.add_argument('bwa_parse_output', type=str, help='File name for parsed data from contig BWA reference alignment SAM output')
argparser.add_argument('contigs_file', type=str, help='FASTA file listing contigs')
argparser.add_argument('plots_folder', type=str, help='Folder name for output plot files')
argparser.add_argument("aln_est_counts", type=str, nargs='?', default=None, help="CSV file listing alignment counts and assignments for classified sequences")
args = argparser.parse_args()

seq_mean_kmer_depths = array.array('d')
with open(args.contigs_file) as unitigs:
  line = unitigs.readline()
  while line:
    if re.search('^>[0-9]', line):
      row = list(map(int, line[1:].split()))
      kmers = row[1] - args.kmer_len + 1
      seq_mean_kmer_depths.append(row[2] / kmers)
    line = unitigs.readline()
seqs = pd.DataFrame(index = range(len(seq_mean_kmer_depths)), columns = ['mean_kmer_depth'])
seqs['mean_kmer_depth'] = seq_mean_kmer_depths
seqs.index.name = 'ID'

seq_alns = pd.read_csv(args.bwa_parse_output, delimiter='\t')
cols_from_to = { 'Length': 'length', 'Matches': 'aln_match_count', 'Clipped': 'clipped', 'MAPQ sum': 'mapq_sum', 'GC content': 'gc', 'Edit distance': 'edit_dist_str' }
seq_alns.rename(columns=cols_from_to, inplace=True)
seq_alns.set_index('ID', inplace=True)
seq_alns['edit_dist_list'] = seq_alns.edit_dist_str.apply(lambda dcounts: utils.valcounts_str_to_ints_list(dcounts))

seq_alns = seq_alns.join(seqs)
if args.aln_est_counts:
  seq_aln_ests = pd.read_csv(args.aln_est_counts)
  seq_aln_ests = seq_aln_ests[['ID', 'Likeliest copy #']]
  seq_aln_ests.rename(columns = { 'Likeliest copy #': 'est_copynum' }, inplace = True)
  seq_aln_ests.set_index('ID', inplace=True)
  seq_alns = seq_alns.join(seq_aln_ests)
seq_alns.reset_index(inplace=True)

seq_alns.sort_values(by=['length', 'mean_kmer_depth'], inplace=True)
seq_alns_long = utils.wide_to_long_from_listcol(seq_alns, 'edit_dist_list')
seq_alns_long.rename(columns = { 'edit_dist_list': 'edit_dist' }, inplace = True)
seq_alns_long.sort_values(by=['length', 'mean_kmer_depth'], inplace=True)

# Plot (almost) whole-dataset edit distance densities
seq_alns['edit_dist_min'] = seq_alns_long.groupby('ID').edit_dist.agg('min')
seq_alns['edit_dist_quantile1'] = seq_alns_long.groupby('ID').edit_dist.agg(lambda dists: dists.quantile(0.25) if not(np.isnan(dists.iloc[0])) else np.nan)
seq_alns['edit_dist_median'] = seq_alns_long.groupby('ID').edit_dist.agg('median')
seq_alns['edit_dist_mean'] = seq_alns_long.groupby('ID').edit_dist.agg('mean')
seq_alns['edit_dist_mode'] = seq_alns_long.groupby('ID').edit_dist.agg(lambda dists: dists.value_counts().index[0] if not(np.isnan(dists.iloc[0])) else np.nan)
seq_alns_use = seq_alns.loc[~seq_alns.edit_dist_str.isna()]

ax = plot_kde(seq_alns_use.edit_dist_quantile1, max_pctl_for_plotting(seq_alns_use.edit_dist_quantile1))
ax = plot_kde(seq_alns_use.edit_dist_median, max_pctl_for_plotting(seq_alns_use.edit_dist_median))
ax = plot_kde(seq_alns_use.edit_dist_mean, max_pctl_for_plotting(seq_alns_use.edit_dist_mean))
mode_max_pctl = max_pctl_for_plotting(seq_alns_use.edit_dist_mode)
ax = plot_kde(seq_alns_use.edit_dist_mode, mode_max_pctl)
ax.set(xlabel = 'per-contig edit distance metric', ylabel = 'density', title = 'All contigs')
ax.get_figure().savefig(os.path.join(args.plots_folder, 'edit_dist_KDEs_all.png'))
plt.clf()

ax = sns.distplot(seq_alns_use.edit_dist_min[seq_alns_use.edit_dist_min <= max_pctl_for_plotting(seq_alns_use.edit_dist_min)], kde=False)
ax.set(xlabel = 'per-contig edit distance min', ylabel = '# observations', title = 'All contigs')
ax.get_figure().savefig(os.path.join(args.plots_folder, 'edit_dist_min_histogram_all.png'))
plt.clf()

ax = sns.distplot(seq_alns_use.edit_dist_mode[seq_alns_use.edit_dist_mode <= mode_max_pctl], kde=False)
ax.set(xlabel = 'per-contig edit distance mode', ylabel = '# observations', title = 'All contigs')
ax.get_figure().savefig(os.path.join(args.plots_folder, 'edit_dist_mode_histogram_all.png'))
plt.clf()

length_strata = utils.get_contig_length_gps(seq_alns_use, seq_alns_use.length)
strata_count = len(length_strata)

# Plot per-length-stratum edit distance densities

for lengp_seqs in length_strata:
  ax = plot_kde(lengp_seqs.edit_dist_quantile1, max_pctl_for_plotting(lengp_seqs.edit_dist_quantile1))
  ax = plot_kde(lengp_seqs.edit_dist_median, max_pctl_for_plotting(lengp_seqs.edit_dist_median))
  ax = plot_kde(lengp_seqs.edit_dist_mean, max_pctl_for_plotting(lengp_seqs.edit_dist_mean))
  mode_max_pctl = max_pctl_for_plotting(lengp_seqs.edit_dist_mode)
  ax = plot_kde(lengp_seqs.edit_dist_mode, mode_max_pctl)

  minlen_str, maxlen_str = str(lengp_seqs.length.min()), str(lengp_seqs.length.max())
  title = 'Contigs of length [' + minlen_str + ', ' + maxlen_str + '] bps'

  ax.set(xlabel = 'per-contig edit distance metric', ylabel = 'KDE', title = title)
  ax.get_figure().savefig(os.path.join(args.plots_folder, 'edit_dist_KDEs_gte' + minlen_str + 'lte' + maxlen_str + '.png'))
  plt.clf()

  ax = sns.distplot(lengp_seqs.edit_dist_min[lengp_seqs.edit_dist_min <= max_pctl_for_plotting(lengp_seqs.edit_dist_min)], kde=False)
  ax.set(xlabel = 'per-contig edit distance min', ylabel = '# observations', title = title)
  ax.get_figure().savefig(os.path.join(args.plots_folder, 'edit_dist_min_histogram_gte' + minlen_str + 'lte' + maxlen_str + '.png'))
  plt.clf()

  ax = sns.distplot(lengp_seqs.edit_dist_mode[lengp_seqs.edit_dist_mode <= mode_max_pctl], kde=False)
  ax.set(xlabel = 'per-contig edit distance mode', ylabel = '# observations', title = title)
  ax.get_figure().savefig(os.path.join(args.plots_folder, 'edit_dist_mode_histogram_gte' + minlen_str + 'lte' + maxlen_str + '.png'))
  plt.clf()


def get_seqs_in_gp_len_bds(seqs, seq_lens, gp_seq_lens):
  return seqs[(seq_lens >= gp_seq_lens.min()) & (seq_lens <= gp_seq_lens.max())]

# Plot per-length-stratum (total edit distances) / (total lengths) against median length, mean length, and group indices
long_seqs_length_strata = list(map(lambda gp_seqs: get_seqs_in_gp_len_bds(seq_alns_long, seq_alns_long.length, gp_seqs.length), length_strata))
df_cols = ['edit_dist_sum', 'length_sum', 'median_length', 'mean_length', 'est_copynum_mean', 'est_copynum_mean_plus_std']
lengp_dist_length_sums = pd.DataFrame(index = range(strata_count), columns = df_cols)
for i in range(strata_count):
  lengp_dist_length_sums.loc[i, 'edit_dist_sum'] = long_seqs_length_strata[i].edit_dist.sum()
  lengp_dist_length_sums.loc[i, 'length_sum'] = length_strata[i].length.sum()
  lengp_dist_length_sums.loc[i, 'median_length'] = length_strata[i].length.median()
  lengp_dist_length_sums.loc[i, 'mean_length'] = length_strata[i].length.mean()
  lengp_dist_length_sums.loc[i, 'est_copynum_mean'] = length_strata[i].est_copynum.mean()
  lengp_dist_length_sums.loc[i, 'est_copynum_mean_plus_std'] = lengp_dist_length_sums.loc[i, 'est_copynum_mean'] + length_strata[i].est_copynum.std()
lengp_dist_length_sums['gp_idx'] = lengp_dist_length_sums.index
lengp_dist_length_sums['dist_len_ratio'] = lengp_dist_length_sums.edit_dist_sum / lengp_dist_length_sums.length_sum

do_scatterplot(lengp_dist_length_sums, 'median_length', 'dist_len_ratio', 'per-stratum median contig length',
    'per-stratum contig (edit distance sum / length sum)', os.path.join(args.plots_folder, 'dist-len-ratio_vs_median-len.png'))
do_scatterplot(lengp_dist_length_sums, 'mean_length', 'dist_len_ratio', 'per-stratum mean contig length',
    'per-stratum contig (edit distance sum / length sum)', os.path.join(args.plots_folder, 'dist-len-ratio_vs_mean-len.png'))
do_scatterplot(lengp_dist_length_sums, 'gp_idx', 'dist_len_ratio', 'length stratum index',
    'per-stratum contig (edit distance sum / length sum)', os.path.join(args.plots_folder, 'dist-len-ratio_vs_len-gp-idx.png'))
do_scatterplot(lengp_dist_length_sums, 'est_copynum_mean', 'dist_len_ratio', 'length stratum estimated copy # mean',
    'per-stratum contig (edit distance sum / length sum)', os.path.join(args.plots_folder, 'dist-len-ratio_vs_est-copynum-mean.png'))
do_scatterplot(lengp_dist_length_sums, 'est_copynum_mean_plus_std', 'dist_len_ratio', 'length stratum estimated copy # mean plus stdev',
    'per-stratum contig (edit distance sum / length sum)', os.path.join(args.plots_folder, 'dist-len-ratio_vs_est-copynum-mean-plus-std.png'))
