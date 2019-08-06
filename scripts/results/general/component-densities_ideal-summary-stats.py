import matplotlib
matplotlib.use('agg')

import argparse
import csv
import matplotlib.pyplot as plt
import math as m
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import sys


argparser = argparse.ArgumentParser(description='Plot aligned and estimated copy-number component densities by sequence mean k-mer depth')
argparser.add_argument('seq_est_aln_file', type=str, help='CSV file listing sequences with aligned and estimated copy numbers')
argparser.add_argument('plots_folder', type=str, help='Folder name for output plot files')
argparser.add_argument('plots_file_prefix', type=str, help='Prefix for output plot file names')
argparser.add_argument('est_len_gp_stats', type=str, nargs='?', default=None, help='CSV file listing sequence length groups used in classification, with summary statistics')
argparser.add_argument('stats_folder', type=str, nargs='?', default=None, help='Folder name for ideal (best-possible) summary statistics output files')
argparser.add_argument('stats_file_prefix', type=str, nargs='?', default=None, help='Prefix for ideal (best-possible) summary statistics output file names')
argparser.add_argument('--copynum_stats_file', type=str, nargs=1, default=None, help='CSV file listing estimated copy number component parameters for all length groups')
args = argparser.parse_args()


def get_max_copynum(seq_likeliest_copynums):
  return min(MAX_INT_COPYNUM, int(seq_likeliest_copynums.max()))

def get_copynums_list(est_max_cpnum):
  if HALF:
    return ([0, 0.5] + list(range(1, est_max_cpnum + 1)))
  return list(range(est_max_cpnum + 1))

def get_kde(vals):
  if len(vals) > 0:
    kdes_c = sm.nonparametric.KDEUnivariate(vals)
    kdes_c.fit(bw='scott')
    return kdes_c
  return None

def get_kdes(seqs_by_cpnum):
  return seqs_by_cpnum.apply(lambda cpnum_seqs: get_kde(cpnum_seqs.mean_kmer_depth.values))

def get_counts(seqs_by_cpnum):
  return seqs_by_cpnum.apply(lambda cpnum_seqs: cpnum_seqs.shape[0])

def get_next_nonnull(array, idx, i):
  while (i < idx.size) and (array[idx[i]] is None):
    i += 1
  if i < idx.size:
    return i
  return np.inf

def get_aln_copynum_bounds(kdes_normed, kde_grid):
  lbs = pd.Series([np.inf] * len(kdes_normed), index = kdes_normed.keys())
  prev_idx = get_next_nonnull(kdes_normed, lbs.index, 0)
  idx = get_next_nonnull(kdes_normed, lbs.index, prev_idx + 1)
  lbs[lbs.index[prev_idx]] = 0
  while prev_idx < np.inf and idx < np.inf:
    intersections = np.argwhere(np.diff(np.sign(kdes_normed[lbs.index[idx]] - kdes_normed[lbs.index[prev_idx]])) > 0)
    separatrices = intersections[(intersections > np.argmax(kdes_normed[lbs.index[prev_idx]])) & (intersections < np.argmax(kdes_normed[lbs.index[idx]]))]
    if separatrices.shape[0] > 0:
      lbs[lbs.index[idx]] = kde_grid[separatrices[0]]
    prev_idx, idx = idx, get_next_nonnull(kdes_normed, lbs.index, idx + 1)
  return lbs

def compute_ideal_positives(seqs, seqs_aln, lbs):
  tps, positives = pd.Series(index = lbs.index), pd.Series(index = lbs.index)
  for i in range(lbs.size - 1):
    cp = lbs.index[i]
    seqs_aln_cpnum = seqs_aln[cp]
    tps[cp] = seqs_aln_cpnum[(seqs_aln_cpnum.mean_kmer_depth >= lbs[cp]) & (seqs_aln_cpnum.mean_kmer_depth < lbs[lbs.index[i+1]])].shape[0]
    positives[cp] = seqs[(seqs.mean_kmer_depth >= lbs[cp]) & (seqs.mean_kmer_depth < lbs[lbs.index[i+1]])].shape[0]
  cp, seqs_aln_cpnum = lbs.index[lbs.size - 1], seqs_aln[lbs.index.max()]
  tps[cp], positives[cp] = seqs_aln_cpnum[seqs_aln_cpnum.mean_kmer_depth >= lbs[cp]].shape[0], seqs[seqs.mean_kmer_depth >= lbs[cp]].shape[0]
  return (tps, positives)

def compute_ideal_stats(tps, counts_aln, positives):
  stats = pd.DataFrame(index=tps.index, columns=['tpr', 'fnr', 'ppv', 'fdr', 'f1'])
  stats['tpr'] = tps / counts_aln
  stats['fnr'] = 1 - stats.tpr
  stats['ppv'] = tps / positives
  stats['fdr'] = 1 - stats.ppv
  stats['f1'] = 2.0 * tps / (counts_aln + positives)
  return stats

def compute_stats(seqs, seqs_aln, counts_aln, lbs):
  tps, positives = pd.Series(index = lbs.index), pd.Series(index = lbs.index)
  for i in range(lbs.size - 1):
    cp = lbs.index[i]
    seqs_aln_cpnum = seqs_aln[cp]
    tps[cp] = seqs_aln_cpnum[(seqs_aln_cpnum.mean_kmer_depth >= lbs[cp]) & (seqs_aln_cpnum.mean_kmer_depth < lbs[lbs.index[i+1]])].shape[0]
    positives[cp] = seqs[(seqs.mean_kmer_depth >= lbs[cp]) & (seqs.mean_kmer_depth < lbs[lbs.index[i+1]])].shape[0]
  cp, seqs_aln_cpnum = lbs.index[lbs.size - 1], seqs_aln[lbs.index.max()]
  tps[cp], positives[cp] = seqs_aln_cpnum[seqs_aln_cpnum.mean_kmer_depth >= lbs[cp]].shape[0], seqs[seqs.mean_kmer_depth >= lbs[cp]].shape[0]
  stats = pd.DataFrame(index=lbs.index, columns=['tpr', 'fnr', 'ppv', 'fdr', 'f1'])
  stats['tpr'] = tps / counts_aln
  stats['fnr'] = 1 - stats.tpr
  stats['ppv'] = tps / positives
  stats['fdr'] = 1 - stats.ppv
  stats['f1'] = 2.0 * tps / (counts_aln + positives)
  return stats

def get_sorted_depth_vals(seq_depth_vals):
  depths = np.copy(seq_depth_vals)
  depths.sort()
  return depths

def get_whole_sample_kde(depths):
  kde_all = sm.nonparametric.KDEUnivariate(depths)
  kde_all.fit(bw='scott')
  return kde_all

def get_density_grid(seqs, depths, grid_min, est_max, grid_dens):
  grid_max = min(seqs.loc[seqs.likeliest_copynum == est_max,].mean_kmer_depth.mean(), depths[-1]) + MAX_OFFSET
  return np.linspace(grid_min, grid_max, grid_dens * (grid_max - grid_min) + 1)

def finalise_figure(ax, fig, title, filename_prefix):
  plt.legend()
  ax.set_title(title)
  fig.savefig(filename_prefix + '.png')

def plot_kdes(kde_grid, cpnums, est_kdes_normed, aln_kdes_normed, est_counts, aln_counts, depths, title_suffix, filename_prefix):
  fig, ax = plt.subplots(figsize = (15, 10))
  for c in cpnums:
    idx = int(c + 1)
    if c < 1:
      idx = m.ceil(c)
    if est_kdes_normed[c] is not None:
      ax.plot(kde_grid, est_kdes_normed[c], color = COLOURS[idx], linestyle = '--', lw = 1, label = 'Estimated copy # ' + str(c))
    if aln_kdes_normed[c] is not None:
      ax.plot(kde_grid, aln_kdes_normed[c], color = COLOURS[idx], lw = 1, label = 'True copy # ' + str(c))
  ax.plot(kde_grid, get_whole_sample_kde(depths).evaluate(kde_grid), 'k', lw = 1, label = "All sequences")
  finalise_figure(ax, fig, 'Densities for estimated and true copy numbers, sequences ' + title_suffix, filename_prefix)

def compute_and_plot(seqs, title_suffix, plots_filepath_prefix, stats_filepath_prefix = None):
  est_max_cpnum = get_max_copynum(seqs.likeliest_copynum)
  cpnums = get_copynums_list(est_max_cpnum)
  seqs_est = { c: seqs.loc[seqs.likeliest_copynum == c,] for c in cpnums[:-1] }
  seqs_est[cpnums[-1]] = seqs.loc[seqs.likeliest_copynum >= cpnums[-1],]
  seqs_aln = { c: seqs.loc[seqs.alns == c,] for c in cpnums[:-1] }
  seqs_aln[cpnums[-1]] = seqs.loc[seqs.alns >= cpnums[-1],]
  seqs_est, seqs_aln = pd.Series(seqs_est), pd.Series(seqs_aln)
  est_kdes, est_counts = get_kdes(seqs_est), get_counts(seqs_est)
  aln_kdes, aln_counts = get_kdes(seqs_aln), get_counts(seqs_aln)
  depths = get_sorted_depth_vals(seqs.mean_kmer_depth.values)
  kde_grid = get_density_grid(seqs, depths, (1 - MIN_OFFSET_FRACTION) * depths[0], est_max_cpnum, 100) # can replace 100 by some other density
  est_kdes_normed, aln_kdes_normed = {}, {}
  for i in est_kdes.index:
    est_kdes_normed[i] = est_kdes[i] and (est_kdes[i].evaluate(kde_grid) * est_counts[i] / len(depths))
  for i in aln_kdes.index:
    aln_kdes_normed[i] = aln_kdes[i] and (aln_kdes[i].evaluate(kde_grid) * aln_counts[i] / len(depths))
  if stats_filepath_prefix:
    tps, positives = compute_ideal_positives(seqs, seqs_aln, get_aln_copynum_bounds(aln_kdes_normed, kde_grid))
    compute_ideal_stats(tps, aln_counts, positives).to_csv(stats_filepath_prefix + '.csv', index_label='Copy #', header=['TPR', 'FNR', 'PPV', 'FDR', 'F1'])
    overall = tps.sum() / seqs.shape[0]
    csv.writer(open(stats_filepath_prefix + '.csv', 'a')).writerow(['Overall', overall, 1 - overall, '', '', ''])
  plot_kdes(kde_grid, cpnums, est_kdes_normed, aln_kdes_normed, est_counts, aln_counts, depths, title_suffix, plots_filepath_prefix)

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


if (args.stats_folder is not None) and (args.stats_file_prefix is None):
  parser.error("Folder for ideal summary statistics output files provided, but filename prefix isn't: latter required with former")

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
  if args.stats_folder is None:
    compute_and_plot(all_seqs, 'of all lengths', args.plots_folder + '/' + args.plots_file_prefix)
  else:
    compute_and_plot(all_seqs, 'of all lengths', args.plots_folder + '/' + args.plots_file_prefix, args.stats_folder + '/' + args.stats_file_prefix)

lbs, ubs = [0, 100, 1000, 10000], [99, 999, 9999, np.Inf]
if args.est_len_gp_stats is not None:
  len_gp_stats = pd.read_csv(args.est_len_gp_stats)
  lbs, ubs = len_gp_stats['Min. len.'], len_gp_stats['Max. len.']
for i in range(len(ubs)):
  lb, ub = lbs[i], ubs[i]
  seqs = all_seqs.loc[(all_seqs.length >= lb) & (all_seqs.length <= ub),]
  if seqs.shape[0] > 0:
    if (args.est_len_gp_stats is not None) and (args.copynum_stats_file is not None): # The latter needs the former to make sense (be defined)
      curr_components = all_components.loc[(all_components.group_min_len == lb) & (all_components.group_max_len == ub),] # TODO: fix +1
      compute_and_plot_population_densities(curr_components, seqs, 'with length in [' + str(lb) + ', ' + str(ub) + ')', args.plots_file_prefix + '_len-gte' + str(lb) + 'lte' + str(ub))
    else:
      if args.stats_folder is None:
        compute_and_plot(seqs, 'with length in [' + str(lb) + ', ' + str(ub) + ')', args.plots_folder + '/' + args.plots_file_prefix + '_len-gte' + str(lb) + 'lte' + str(ub))
      else:
        compute_and_plot(seqs, 'with length in [' + str(lb) + ', ' + str(ub) + ')', args.plots_folder + '/' + args.plots_file_prefix + '_len-gte' + str(lb) + 'lte' + str(ub),
            args.stats_folder + '/' + args.stats_file_prefix + '_len-gte' + str(lb) + 'lte' + str(ub))
