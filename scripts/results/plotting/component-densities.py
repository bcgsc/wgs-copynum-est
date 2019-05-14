import matplotlib
matplotlib.use('agg')

import argparse
import matplotlib.pyplot as plt
import math as m
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import sys


argparser = argparse.ArgumentParser(description='Plot aligned and estimated copy-number component densities by sequence mean k-mer depth')
argparser.add_argument('seq_est_aln_file', type=str, help='CSV file listing sequences with aligned and estimated copy numbers')
argparser.add_argument('plots_file_prefix', type=str, help='Prefix for output plot file names')
argparser.add_argument('est_len_gp_stats', type=str, nargs='?', default=None, help='CSV file listing sequence length groups used in classification, with summary statistics')
argparser.add_argument('--copynum_stats_file', type=str, nargs=1, default=None, help='CSV file listing estimated copy number component parameters for all length groups')
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

def get_max_copynum(seq_likeliest_copynums):
  return min(MAX_INT_COPYNUM, int(seq_likeliest_copynums.max()))

def get_sorted_depth_vals(seq_depth_vals):
  depths = np.copy(seq_depth_vals)
  depths.sort()
  return depths

def get_whole_sample_kde(depths):
  kde_all = sm.nonparametric.KDEUnivariate(depths)
  kde_all.fit(bw='scott')
  return kde_all

def get_density_grid(seqs, depths, grid_min, est_max, grid_dens):
  grid_max = min(seqs.loc[seqs.likeliest_copynum == est_max,].mean_kmer_depth.mean() * 2, depths[-1]) + MAX_OFFSET
  return np.linspace(grid_min, grid_max, grid_dens * (grid_max - grid_min) + 1)

def finalise_figure(ax, fig, title, filename_prefix):
  plt.legend()
  ax.set_title(title)
  fig.savefig(filename_prefix + '.png')

def compute_and_plot_kdes(seqs, title_suffix, filename_prefix):
  est_max = get_max_copynum(seqs.likeliest_copynum)
  cpnums = range(est_max + 1)
  if HALF:
    cpnums = [0, 0.5] + list(range(1, est_max + 1))

  est_kdes = pd.Series([None] * len(cpnums), index = cpnums)
  est_counts = pd.Series([0] * len(cpnums), index = cpnums)
  aln_kdes = pd.Series([None] * len(cpnums), index = cpnums)
  aln_counts = pd.Series([0] * len(cpnums), index = cpnums)
  fit_est_kdes(seqs, cpnums, est_kdes, est_counts)
  fit_aln_kdes(seqs, cpnums, aln_kdes, aln_counts)

  depths = get_sorted_depth_vals(seqs.mean_kmer_depth.values)
  kde_grid = get_density_grid(seqs, depths, (1 - MIN_OFFSET_FRACTION) * depths[0], est_max, 100) # can replace 100 by some other density
  fig, ax = plt.subplots(figsize = (15, 10))
  plot_kdes(cpnums, ax, kde_grid, est_kdes, aln_kdes, get_whole_sample_kde(depths), est_counts, aln_counts, len(depths))
  finalise_figure(ax, fig, 'Densities for estimated and true copy numbers, sequences ' + title_suffix, filename_prefix)

def compute_and_plot_population_densities(components, seqs, title_suffix, filename_prefix):
  est_max = get_max_copynum(seqs.likeliest_copynum)
  depths = get_sorted_depth_vals(seqs.mean_kmer_depth.values)
  density_grid = get_density_grid(seqs, depths, 0, est_max, 50)
  densities = np.zeros([components.shape[0], density_grid.size])
  fig, ax = plt.subplots(figsize = (15, 10))
  for i in range(components.shape[0]):
    if np.isnan(components.iloc[i].gamma_shape):
      densities[i] = components.iloc[i].weight * stats.norm.pdf(density_grid, components.iloc[i].gauss_mean, components.iloc[i].gauss_stdev)
    else:
      densities[i] = components.iloc[i].weight * stats.gamma.pdf(density_grid, components.iloc[i].gamma_shape, components.iloc[i].gamma_loc, components.iloc[i].gamma_scale)
    ax.plot(density_grid, densities[i], color = COLOURS[i], linestyle = '--', lw = 1, label = 'Copy # ' + str(components.iloc[i].component_idx) + ' population density')
  ax.plot(density_grid, np.sum(densities, axis=0), 'k', linestyle = '--', lw = 1, label = 'Total population density')
  ax.plot(density_grid, get_whole_sample_kde(depths).evaluate(density_grid), 'k', lw = 1, label = 'Total sample density')
  finalise_figure(ax, fig, 'Population densities for estimated copy number components, sequences ' + title_suffix, filename_prefix)


all_seqs = pd.read_csv(args.seq_est_aln_file)
all_seqs.rename(index=str, inplace=True,
    columns = { 'Length': 'length', 'Average depth': 'mean_kmer_depth', 'GC %': 'GC', 'Likeliest copy #': 'likeliest_copynum',
                'Alignments (alns)': 'alns', 'Other alns': 'other_alns', 'Other-aln CIGARs': 'other_aln_cigars', 'MAPQ (unique aln only)': 'unique_aln_mapq' })
if args.copynum_stats_file is not None:
  all_components = pd.read_csv(args.copynum_stats_file[0])
  all_components.rename(columns = { 'Group #': 'group_idx', 'Group min. len.': 'group_min_len', 'Group max. len.': 'group_max_len',
                                    'Component #': 'component_idx', 'Component depth lower bound': 'depth_inf', 'Component max. depth': 'depth_max',
                                    'Weight': 'weight', 'Mean': 'gauss_mean', 'Std. deviation': 'gauss_stdev', 'Location': 'gamma_loc', 'Shape': 'gamma_shape', 'Scale': 'gamma_scale' }, inplace = True)

MAX_OFFSET, MIN_OFFSET_FRACTION = 1, 0.1
HALF = ((all_seqs.likeliest_copynum == 0.5).sum() > 0)
MAX_INT_COPYNUM = 8 + int(not(HALF))
COLOURS = [ 'xkcd:azure', 'xkcd:coral', 'xkcd:darkgreen', 'xkcd:gold', 'xkcd:plum', 'xkcd:darkblue', 'xkcd:olive', 'xkcd:magenta', 'xkcd:chocolate', 'xkcd:yellowgreen' ]

if args.copynum_stats_file is None:
  compute_and_plot_kdes(all_seqs, 'of all lengths', args.plots_file_prefix)

lbs, ubs = [0, 100, 1000, 10000], [99, 999, 9999, np.Inf]
if args.est_len_gp_stats is not None:
  len_gp_stats = pd.read_csv(args.est_len_gp_stats)
  lbs, ubs = len_gp_stats['Min. len.'], len_gp_stats['Max. len.']
for i in range(len(ubs)):
  lb, ub = lbs[i], ubs[i]
  seqs = all_seqs.loc[(all_seqs.length >= lb) & (all_seqs.length <= ub),]
  if seqs.shape[0] > 0:
    if args.copynum_stats_file is not None:
      curr_components = all_components.loc[(all_components.group_min_len == lb) & (all_components.group_max_len == ub),] # TODO: fix +1
      compute_and_plot_population_densities(curr_components, seqs, 'with length in [' + str(lb) + ', ' + str(ub) + ')', args.plots_file_prefix + '_len-gte' + str(lb) + 'lte' + str(ub))
    else:
      compute_and_plot_kdes(seqs, 'with length in [' + str(lb) + ', ' + str(ub) + ')', args.plots_file_prefix + '_len-gte' + str(lb) + 'lte' + str(ub))
