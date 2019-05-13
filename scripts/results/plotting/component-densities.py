import matplotlib
matplotlib.use('agg')

import argparse
import matplotlib.pyplot as plt
import math as m
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys


argparser = argparse.ArgumentParser(description='Plot aligned and estimated copy-number component densities by sequence mean k-mer depth')
argparser.add_argument('seq_est_aln_file', type=str, help='CSV file listing sequences with aligned and estimated copy numbers')
argparser.add_argument('plots_file_prefix', type=str, help='Prefix for output plot file names')
args = argparser.parse_args()

def kde_and_count_copynum(vals, c, kdes, counts):
  if len(vals) > 0:
    kdes[c] = sm.nonparametric.KDEUnivariate(vals)
    kdes[c].fit(bw='scott')
    counts[c] = len(vals)

def fit_est_kdes(seqs, cpnums, kdes, counts):
  for c in cpnums[:-1]:
    kde_and_count_copynum(seqs.loc[seqs.likeliest_copynum == c,].mean_kmer_depth.values, c, kdes, counts)
  kde_and_count_copynum(seqs.loc[seqs.likeliest_copynum >= cpnums[-1],].mean_kmer_depth.values, cpnums[-1], kdes, counts)

def fit_aln_kdes(seqs, cpnums, kdes, counts):
  for c in cpnums[:-1]:
    kde_and_count_copynum(seqs.loc[seqs.alns == c,].mean_kmer_depth.values, c, kdes, counts)
  kde_and_count_copynum(seqs.loc[seqs.alns >= cpnums[-1],].mean_kmer_depth.values, cpnums[-1], kdes, counts)

def plot_kdes(cpnums, ax, kde_grid, est_kdes, aln_kdes, kde_all, est_counts, aln_counts, all_count):
  for c in cpnums:
    idx = int(c + 1)
    if c < 1:
      idx = m.ceil(c)
    if est_kdes[c] is not None:
      ax.plot(kde_grid, est_kdes[c].evaluate(kde_grid) * est_counts[c] / all_count, color = COLOURS[idx], linestyle = '--', lw = 1, label = 'Estimated copy # ' + str(c))
    if aln_kdes[c] is not None:
      ax.plot(kde_grid, aln_kdes[c].evaluate(kde_grid) * aln_counts[c] / all_count, color = COLOURS[idx], lw = 1, label = 'True copy # ' + str(c))
  ax.plot(kde_grid, kde_all.evaluate(kde_grid), 'k', lw = 1, label = "All sequences")

def compute_and_plot_kdes(seqs, title_suffix, filename_prefix):
  est_max = min(MAX_INT_COPYNUM, int(seqs.likeliest_copynum.max()))
  cpnums = range(est_max + 1)
  if HALF:
    cpnums = [0, 0.5] + list(range(1, est_max + 1))

  est_kdes = pd.Series([None] * len(cpnums), index = cpnums)
  est_counts = pd.Series([0] * len(cpnums), index = cpnums)
  aln_kdes = pd.Series([None] * len(cpnums), index = cpnums)
  aln_counts = pd.Series([0] * len(cpnums), index = cpnums)
  fit_est_kdes(seqs, cpnums, est_kdes, est_counts)
  fit_aln_kdes(seqs, cpnums, aln_kdes, aln_counts)

  depths = np.copy(seqs.mean_kmer_depth.values)
  depths.sort()
  kde_all = sm.nonparametric.KDEUnivariate(depths)
  kde_all.fit(bw='scott')

  grid_min = (1 - MIN_OFFSET_FRACTION) * depths[0]
  grid_max = min(seqs.loc[seqs.likeliest_copynum == est_max,].mean_kmer_depth.mean() * 2, depths[-1]) + MAX_OFFSET
  kde_grid = np.linspace(grid_min, grid_max, 100 * (grid_max - grid_min) + 1) # can replace 100 by some other density
  fig, ax = plt.subplots(figsize = (15, 10))
  plot_kdes(cpnums, ax, kde_grid, est_kdes, aln_kdes, kde_all, est_counts, aln_counts, len(depths))
  plt.legend()
  ax.set_title('Densities for estimated and true copy numbers, sequences ' + title_suffix)
  fig.savefig(filename_prefix + '.png')


all_seqs = pd.read_csv(args.seq_est_aln_file)
all_seqs.rename(index=str, inplace=True,
    columns = { 'Length': 'length', 'Average depth': 'mean_kmer_depth', 'GC %': 'GC', 'Likeliest copy #': 'likeliest_copynum',
                'Alignments (alns)': 'alns', 'Other alns': 'other_alns', 'Other-aln CIGARs': 'other_aln_cigars', 'MAPQ (unique aln only)': 'unique_aln_mapq' })

MAX_OFFSET, MIN_OFFSET_FRACTION = 1, 0.1
HALF = ((all_seqs.likeliest_copynum == 0.5).sum() > 0)
MAX_INT_COPYNUM = 8 + int(not(HALF))
COLOURS = [ 'xkcd:azure', 'xkcd:coral', 'xkcd:darkgreen', 'xkcd:gold', 'xkcd:plum', 'xkcd:darkblue', 'xkcd:olive', 'xkcd:magenta', 'xkcd:chocolate', 'xkcd:yellowgreen' ]

compute_and_plot_kdes(all_seqs, 'of all lengths', args.plots_file_prefix)

lb = 0
for ub in [100, 1000, 10000, np.Inf]:
  seqs = all_seqs.loc[(all_seqs.length >= lb) & (all_seqs.length < ub),]
  if seqs.shape[0] > 0:
    compute_and_plot_kdes(seqs, 'with length in [' + str(lb) + ', ' + str(ub) + ')', args.plots_file_prefix + '_len-gte' + str(lb) + 'lt' + str(ub))
  lb = ub
