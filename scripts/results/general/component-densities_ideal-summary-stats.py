import matplotlib
matplotlib.use('agg')

import argparse
import csv
import matplotlib.pyplot as plt
import math as m
import numpy as np
import os
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import sys


argparser = argparse.ArgumentParser(description='Plot aligned and estimated copy-number component densities by sequence mean k-mer depth')
argparser.add_argument('--use_oom_len_gps', action="store_true",
    help='Plot and compute stats/counts for sequences partitioned by order of magnitude ([0, 99], [100, 999], ...) instead of length strata used during copy # estimation.')
argparser.add_argument('--plot_est_population_densities', action="store_true", help='Plot population (ideal) densities using estimated copy number component parameters')
argparser.add_argument('seq_est_aln_file', type=str, help='CSV file listing sequences with aligned and estimated copy numbers')
argparser.add_argument('est_len_gp_stats', type=str, help='CSV file listing sequence length groups used in classification, with summary statistics')
argparser.add_argument('copynum_stats_file', type=str, help='CSV file listing estimated copy number component parameters for all length groups')
argparser.add_argument('plots_folder', type=str, help='Folder name for output plot files')
argparser.add_argument('plots_file_prefix', type=str, help='Prefix for output plot file names')
# Use presence of following argument to indicate that best-possible summary figures should be computed and output
argparser.add_argument('ideal_summary_folder', type=str, nargs='?', default=None, help='Folder name for ideal (best-possible) summary statistics and counts output files')
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

def get_next_nonnull(array, idx, i):
    while (i < idx.size) and (array[idx[i]] is None):
        i += 1
    if i < idx.size:
        return i
    return np.inf

def get_aln_copynum_bounds(kdes_normed, kde_grid):
    copynum_kdes = pd.DataFrame(index = kdes_normed.keys(), columns = kde_grid)
    for cpnum in kdes_normed.keys():
        copynum_kdes.loc[cpnum] = kdes_normed[cpnum]
    copynum_kdes = copynum_kdes.astype(float)
    maxdensity_copynums = copynum_kdes.idxmax()
    maxdens_cpnums_change_idxs = np.where(np.diff(maxdensity_copynums))[0]
    if maxdens_cpnums_change_idxs.size == 0:
        maxdens_cpnums_change_idxs = np.array([-1])
    maxdens_cpnums_change_idxs += 1
    maxdens_change_cpnums = pd.Series([0] * (maxdens_cpnums_change_idxs.size + 1))
    maxdens_change_cpnums[0] = maxdensity_copynums.iloc[0]
    if maxdens_change_cpnums.size > 1:
        maxdens_change_cpnums[1:] = maxdensity_copynums.iloc[maxdens_cpnums_change_idxs]
        # Want indices where copy number is followed by a larger copy number, plus the last cp # preceded by a smaller one
        cpnum_assnmt_idxs = np.where(np.diff(maxdens_change_cpnums) > 0)[0]
        if cpnum_assnmt_idxs.size > 0:
            cpnum_assnmts_tmp = pd.Series([np.nan] * cpnum_assnmt_idxs.size)
            cpnum_assnmts_tmp = maxdens_change_cpnums[cpnum_assnmt_idxs]
            # To eliminate all assignments preceded by any larger one(s), and consolidate duplicates
            cpnum_assnmts_tmp = cpnum_assnmts_tmp[cpnum_assnmts_tmp >= cpnum_assnmts_tmp.cummax()].unique()
            cpnum_assnmts = pd.Series([np.nan] * (cpnum_assnmts_tmp.size + 1))
            cpnum_assnmts[:-1] = cpnum_assnmts_tmp
            cpnum_assnmts.iloc[-1] = maxdens_change_cpnums[cpnum_assnmt_idxs[np.argwhere(maxdens_change_cpnums[cpnum_assnmt_idxs] == cpnum_assnmts_tmp[-1])[0][0]] + 1]
        elif maxdens_change_cpnums.iloc[-1] > 0:
            cpnum_assnmts = pd.Series([maxdens_change_cpnums.iloc[-1]])
        else:
            cpnum_assnmts = pd.Series([maxdens_change_cpnums.iloc[-2]])
    else:
        cpnum_assnmts = pd.Series([maxdens_change_cpnums[0]])
    lbs = pd.Series([np.inf] * len(kdes_normed), index = kdes_normed.keys())
    lbs[cpnum_assnmts[0]] = 0
    for i in range(1, cpnum_assnmts.size):
        intersections = np.array([])
        if kdes_normed[cpnum_assnmts.iloc[i-1]] is not None:
            intersection_idxs = np.argwhere(np.diff(np.sign(kdes_normed[cpnum_assnmts.iloc[i]] - kdes_normed[cpnum_assnmts.iloc[i-1]])) > 0)
            # between peaks
            intersections = intersection_idxs[(intersection_idxs > np.argmax(kdes_normed[cpnum_assnmts.iloc[i-1]])) & (intersection_idxs < np.argmax(kdes_normed[cpnum_assnmts.iloc[i]]))]
            if intersections.size == 0:
                # before peaks
                intersections = intersection_idxs[(intersection_idxs < np.argmax(kdes_normed[cpnum_assnmts.iloc[i-1]])) & (intersection_idxs < np.argmax(kdes_normed[cpnum_assnmts.iloc[i]]))]
                if intersections.size == 0:
                    # after peaks
                    intersections = intersection_idxs[(intersection_idxs > np.argmax(kdes_normed[cpnum_assnmts.iloc[i-1]])) & (intersection_idxs > np.argmax(kdes_normed[cpnum_assnmts.iloc[i]]))]
            if intersections.size:
                lbs[cpnum_assnmts.iloc[i]] = kde_grid[intersections[0]]
        else:
            lbs[cpnum_assnmts.iloc[i]] = kde_grid[0]
    return lbs

def compute_ideal_stats(tps, counts_aln, positives):
    stats = pd.DataFrame(index=tps.index, columns=['tpr', 'fnr', 'ppv', 'fdr', 'f1'])
    stats['tpr'] = tps / counts_aln
    stats['fnr'] = 1 - stats.tpr
    stats['ppv'] = tps / positives
    stats['fdr'] = 1 - stats.ppv
    stats['f1'] = 2.0 * tps / (counts_aln + positives)
    return stats

def get_whole_sample_kde(depths):
    kde_all = sm.nonparametric.KDEUnivariate(depths)
    kde_all.fit(bw='scott')
    return kde_all

def init_lengp_cpnum_ideal_stats_table(lengps_count, cpnums, lengp_min_lens, lengp_max_lens):
    len_gp_cpnum_table = pd.DataFrame(0, index=range(lengps_count), columns=['minlen', 'maxlen'] + cpnums + ['total'])
    len_gp_cpnum_table.loc[:(lengps_count - 2), 'minlen'], len_gp_cpnum_table.loc[:(lengps_count - 2), 'maxlen'] = lengp_min_lens, lengp_max_lens
    len_gp_cpnum_table.loc[lengps_count - 1, 'minlen'], len_gp_cpnum_table.loc[lengps_count - 1, 'maxlen'] = lengp_min_lens.iloc[0], lengp_max_lens.iloc[-1]
    return len_gp_cpnum_table

def get_seqs_attr_per_cpnum(seqs, attr, cpnums):
    seqs_attr = { c: seqs.loc[getattr(seqs, attr) == c,] for c in cpnums[:-1] }
    seqs_attr[cpnums[-1]] = seqs.loc[getattr(seqs, attr) >= cpnums[-1],]
    return pd.Series(seqs_attr)

def get_counts(seqs_by_cpnum):
    return seqs_by_cpnum.apply(lambda cpnum_seqs: cpnum_seqs.shape[0])

def get_sorted_depth_vals(seq_depth_vals):
    depths = np.copy(seq_depth_vals)
    depths.sort()
    return depths

def get_density_grid(seqs, depths, grid_min, est_max, grid_dens, mode):
    depths_98th_pctile, est_max_rel_ub = np.quantile(depths, 0.98), 1.75 * est_max * mode
    if depths[-1] < est_max_rel_ub:
        grid_max = depths[-1] + MAX_OFFSET
        if depths[-1] - depths_98th_pctile > 0.5 * mode:
            grid_max = depths_98th_pctile + MAX_OFFSET
    elif (depths_98th_pctile >= (est_max + 1) * mode) and (depths_98th_pctile <= est_max_rel_ub):
        grid_max = depths_98th_pctile + MAX_OFFSET
    elif seqs.loc[seqs.likeliest_copynum == est_max,].mean_kmer_depth.mean() > est_max_rel_ub:
        grid_max = est_max_rel_ub + MAX_OFFSET
    else:
        grid_max = (((est_max + 1) * mode) or depths_98th_pctile) + MAX_OFFSET
    return np.linspace(grid_min, grid_max, grid_dens * (grid_max - grid_min) + 1)

def get_normalised_kdes(seqs_attr, kde_grid, counts, total_count):
    kdes = get_kdes(seqs_attr)
    kdes_normed = {}
    for i in kdes.index:
        kdes_normed[i] = kdes[i] and (kdes[i].evaluate(kde_grid) * counts[i] / total_count)
    return kdes_normed

def finalise_figure(ax, fig, title, filename_prefix):
    plt.legend()
    ax.set_title(title)
    fig.savefig(filename_prefix + '.png')

def plot_kdes(kde_grid, cpnums, est_kdes_normed, aln_kdes_normed, depths, title_suffix, filename_prefix):
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

def plot_aln_kdes_with_est_bds(kde_grid, cpnums, aln_kdes_normed, est_cpnum_data, depths, title_suffix, filename_prefix):
    fig, ax = plt.subplots(figsize = (15, 10))
    for i in range(len(cpnums)):
        c, idx = cpnums[i], int(cpnums[i] + 1)
        if c < 1:
            idx = m.ceil(c)
        if aln_kdes_normed[c] is not None:
            ax.plot(kde_grid, aln_kdes_normed[c], color = COLOURS[idx], lw = 1, label = 'True copy # ' + str(c))
        cpnum_metadata = est_cpnum_data[est_cpnum_data.copynum == c]
        if len(cpnum_metadata) > 0:
            plt.axvline(x = cpnum_metadata.depth_min.iloc[0], color = COLOURS[idx], linestyle = '--', lw = 1, label = 'Estimated copy # ' + str(c) + ' lower bound')
    if len(cpnum_metadata) == 0:
        cpnum_metadata = est_cpnum_data[est_cpnum_data.copynum == cpnums[i-1]]
        plt.axvline(x = cpnum_metadata.depth_ub.iloc[0], color = COLOURS[idx], linestyle = '--', lw = 1, label = 'Estimated copy # ' + str(cpnums[i-1]+1) + ' lower bound')
    ax.plot(kde_grid, get_whole_sample_kde(depths).evaluate(kde_grid), 'k', lw = 1, label = "All sequences")
    finalise_figure(ax, fig, 'Densities for estimated and true copy numbers, sequences ' + title_suffix, filename_prefix)

def compute_ideal_tps(seqs_aln, lbs):
    tps = pd.Series(index = lbs.index)
    for i in range(lbs.size - 1):
        cp = lbs.index[i]
        tps[cp] = seqs_aln[cp][(seqs_aln[cp].mean_kmer_depth >= lbs[cp]) & (seqs_aln[cp].mean_kmer_depth < lbs[lbs.index[i+1]])].shape[0]
    cp, seqs_aln_cpnum = lbs.index[lbs.size - 1], seqs_aln[lbs.index.max()]
    tps[cp] = seqs_aln_cpnum[seqs_aln_cpnum.mean_kmer_depth >= lbs[cp]].shape[0]
    return tps

def compute_ideal_positives(seqs, lbs, assigned_cpnums):
    positives = pd.Series(0, index = lbs.index)
    for i in range(len(assigned_cpnums) - 1):
        positives[assigned_cpnums[i]] = seqs[(seqs.mean_kmer_depth >= lbs[assigned_cpnums[i]]) & (seqs.mean_kmer_depth < lbs[assigned_cpnums[i+1]])].shape[0]
    positives[assigned_cpnums[-1]] = seqs[seqs.mean_kmer_depth >= lbs[assigned_cpnums[-1]]].shape[0]
    return positives

def write_stats(stats, filepath_prefix, overall):
    stats.to_csv(filepath_prefix + '.csv', index_label='Copy #', header=['TPR', 'FNR', 'PPV', 'FDR', 'F1'])
    csv.writer(open(filepath_prefix + '.csv', 'a')).writerow(['Overall', overall, 1 - overall, '', '', ''])

def write_counts(counts, rowmax, cpnums_all, filepath_prefix):
    counts.loc[:(rowmax - 1)].to_csv(filepath_prefix + '.csv', index_label = 'Len. gp.',
        header = ['Min. len.', 'Max. len'] + list(map(lambda n: 'copy # ' + str(n), cpnums_all)) + ['Len. gp. total'])
    csv.writer(open(filepath_prefix + '.csv', 'a')).writerow(['All'] + counts.loc[rowmax].tolist())

def compute_and_plot_population_densities(components, seqs, mode, title_suffix, filename_prefix):
    est_max = get_max_copynum(seqs.likeliest_copynum)
    depths = get_sorted_depth_vals(seqs.mean_kmer_depth.values)
    density_grid = get_density_grid(seqs, depths, 0, est_max, 50, mode)
    densities = np.zeros([components.shape[0], density_grid.size])
    fig, ax = plt.subplots(figsize = (15, 10))
    for i in range(components.shape[0]):
        if np.isnan(components.iloc[i].gamma_shape):
            if components.iloc[i].copynum > 0:
                densities[i] = components.iloc[i].weight * stats.norm.pdf(density_grid, components.iloc[i].cpnum_mean, components.iloc[i].cpnum_stdev)
            else:
                densities[i] = components.iloc[i].weight * stats.expon.pdf(density_grid, 0, components.iloc[i].cpnum_mean)
        else:
            densities[i] = components.iloc[i].weight * stats.gamma.pdf(density_grid, components.iloc[i].gamma_shape, components.iloc[i].gamma_loc, components.iloc[i].gamma_scale)
        ax.plot(density_grid, densities[i], color = COLOURS[i], linestyle = '--', lw = 1, label = 'Copy # ' + str(components.iloc[i].copynum) + ' population density')
    ax.plot(density_grid, np.sum(densities, axis=0), 'k', linestyle = '--', lw = 1, label = 'Total population density')
    ax.plot(density_grid, get_whole_sample_kde(depths).evaluate(density_grid), 'k', lw = 1, label = 'Total sample density')
    finalise_figure(ax, fig, 'Population densities for estimated copy number components, sequences ' + title_suffix, filename_prefix)


all_seqs = pd.read_csv(args.seq_est_aln_file)
if all_seqs.shape[0] < 2:
  print('Trivial assembly and estimation output: only 1 sequence.\nExiting.')
  sys.exit()

all_seqs.rename(index=str, inplace=True,
    columns = { 'Length': 'length', 'Average depth': 'mean_kmer_depth', 'GC %': 'GC', 'Likeliest copy #': 'likeliest_copynum',
                'Alignments (alns)': 'alns', 'Other alns': 'other_alns', 'Other-aln CIGARs': 'other_aln_cigars', 'MAPQ (unique aln only)': 'unique_aln_mapq' })
all_components = pd.read_csv(args.copynum_stats_file)
all_components.rename(columns = { 'Group #': 'group_idx', 'Group min. len.': 'group_min_len', 'Group max. len.': 'group_max_len',
                                  'Component #': 'copynum', 'Component depth lower bound': 'depth_min', 'Component max. depth': 'depth_ub',
                                  'Weight': 'weight', 'Mean': 'cpnum_mean', 'Std. deviation': 'cpnum_stdev', 'Location': 'gamma_loc', 'Shape': 'gamma_shape', 'Scale': 'gamma_scale' }, inplace = True)

MAX_OFFSET, MIN_OFFSET_FRACTION = 1, 0.1
HALF = ((all_seqs.likeliest_copynum == 0.5).sum() > 0)
MAX_INT_COPYNUM = 8 + int(not(HALF))
COLOURS = [ 'xkcd:azure', 'xkcd:coral', 'xkcd:darkgreen', 'xkcd:gold', 'xkcd:plum', 'xkcd:darkblue', 'xkcd:olive', 'xkcd:magenta', 'xkcd:chocolate', 'xkcd:yellowgreen' ]

smallest_cpnum = all_components.loc[(all_components.copynum > 0) & (all_components.depth_min < np.inf)].iloc[0]
MODE = smallest_cpnum.cpnum_mean / smallest_cpnum.copynum

lbs, ubs = pd.Series([0, 100, 1000, 10000]), pd.Series([99, 999, 9999, np.Inf])
if not(args.use_oom_len_gps):
    len_gp_stats = pd.read_csv(args.est_len_gp_stats)
    lbs, ubs = len_gp_stats['Min. len.'], len_gp_stats['Max. len.']

if not(args.plot_est_population_densities):
    est_max_cpnum_all = get_max_copynum(all_seqs.likeliest_copynum)
    cpnums_all = get_copynums_list(est_max_cpnum_all)
    seqs_est, seqs_aln = get_seqs_attr_per_cpnum(all_seqs, 'likeliest_copynum', cpnums_all), get_seqs_attr_per_cpnum(all_seqs, 'alns', cpnums_all)
    est_counts, aln_counts = get_counts(seqs_est), get_counts(seqs_aln)
    depths = get_sorted_depth_vals(all_seqs.mean_kmer_depth.values)
    kde_grid = get_density_grid(all_seqs, depths, (1 - MIN_OFFSET_FRACTION) * depths[0], est_max_cpnum_all, 100, MODE) # can replace 100 by some other density
    est_kdes_normed = get_normalised_kdes(seqs_est, kde_grid, est_counts, len(depths))
    aln_kdes_normed = get_normalised_kdes(seqs_aln, kde_grid, aln_counts, len(depths))
    plot_kdes(kde_grid, cpnums_all, est_kdes_normed, aln_kdes_normed, depths, 'of all lengths', os.path.join(args.plots_folder, args.plots_file_prefix))
    if args.ideal_summary_folder:
        ideal_summary_stats_folder, ideal_counts_folder = os.path.join(args.ideal_summary_folder, 'stats'), os.path.join(args.ideal_summary_folder, 'counts')
        if not(os.path.exists(ideal_summary_stats_folder)):
            os.mkdir(ideal_summary_stats_folder)
        if not(os.path.exists(ideal_counts_folder)):
            os.mkdir(ideal_counts_folder)
        len_gp_cpnum_lbs = pd.DataFrame(np.inf, index=range(len(ubs)), columns=['minlen', 'maxlen'] + cpnums_all)
        len_gp_cpnum_lbs['minlen'], len_gp_cpnum_lbs['maxlen'] = lbs, ubs
        len_gp_cpnum_alns = init_lengp_cpnum_ideal_stats_table(len(ubs) + 1, cpnums_all, lbs, ubs)
        len_gp_cpnum_tps = init_lengp_cpnum_ideal_stats_table(len(ubs) + 1, cpnums_all, lbs, ubs)
        len_gp_cpnum_positives = init_lengp_cpnum_ideal_stats_table(len(ubs) + 1, cpnums_all, lbs, ubs)

for i in range(len(ubs)):
    lb, ub = lbs[i], ubs[i]
    if ub < np.inf:
        ub = int(ub)
    seqs = all_seqs.loc[(all_seqs.length >= lb) & (all_seqs.length <= ub),]
    if seqs.shape[0] > 0:
        label_ub, filename_ub = str(ub) + ']', 'lte' + str(ub)
        if ub == np.inf:
            label_ub, filename_ub = str(ub) + ')', 'lt' + str(ub)
        if args.use_oom_len_gps:
            curr_components = all_components.loc[(all_components.group_min_len >= lb) & (all_components.group_max_len <= ub),]
        else:
            curr_components = all_components.loc[(all_components.group_min_len == lb) & (all_components.group_max_len == ub),]
        if args.plot_est_population_densities:
            compute_and_plot_population_densities(curr_components, seqs, MODE, 'with length in [' + str(lb) + ', ' + label_ub,
                os.path.join(args.plots_folder, args.plots_file_prefix + '_len-gte' + str(lb) + filename_ub))
        else:
            est_max_cpnum = get_max_copynum(seqs.likeliest_copynum)
            cpnums = get_copynums_list(est_max_cpnum)
            seqs_est, seqs_aln = get_seqs_attr_per_cpnum(seqs, 'likeliest_copynum', cpnums), get_seqs_attr_per_cpnum(seqs, 'alns', cpnums)
            est_counts, aln_counts = get_counts(seqs_est), get_counts(seqs_aln)
            depths = get_sorted_depth_vals(seqs.mean_kmer_depth.values)
            kde_grid = get_density_grid(seqs, depths, (1 - MIN_OFFSET_FRACTION) * depths[0], est_max_cpnum, 100, MODE) # can replace 100 by some other density
            aln_kdes_normed = get_normalised_kdes(seqs_aln, kde_grid, aln_counts, len(depths))
            if args.use_oom_len_gps:
                est_kdes_normed = get_normalised_kdes(seqs_est, kde_grid, est_counts, len(depths))
                plot_kdes(kde_grid, cpnums, est_kdes_normed, aln_kdes_normed, depths, 'with length in [' + str(lb) + ', ' + label_ub,
                    os.path.join(args.plots_folder, args.plots_file_prefix + '_len-gte' + str(lb) + filename_ub))
            else:
                plot_aln_kdes_with_est_bds(kde_grid, cpnums, aln_kdes_normed, curr_components, depths, 'with length in [' + str(lb) + ', ' + label_ub,
                    os.path.join(args.plots_folder, args.plots_file_prefix + '_len-gte' + str(lb) + filename_ub))
            if args.ideal_summary_folder:
                cpnum_lbs = get_aln_copynum_bounds(aln_kdes_normed, kde_grid)
                assigned_cpnums = cpnum_lbs[cpnum_lbs < np.inf].index
                tps, positives = compute_ideal_tps(seqs_aln, cpnum_lbs), compute_ideal_positives(seqs, cpnum_lbs, assigned_cpnums)
                for j in tps.index:
                    len_gp_cpnum_lbs.loc[i, j] = cpnum_lbs[j]
                    len_gp_cpnum_alns.loc[i, j], len_gp_cpnum_tps.loc[i, j], len_gp_cpnum_positives.loc[i, j] = aln_counts[j], tps[j], positives[j]
                write_stats(compute_ideal_stats(len_gp_cpnum_tps.loc[i, cpnums], len_gp_cpnum_alns.loc[i, cpnums], len_gp_cpnum_positives.loc[i, cpnums]),
                    os.path.join(ideal_summary_stats_folder, 'stats_len-gte' + str(lb) + filename_ub), tps.sum() / seqs.shape[0])

if not(args.plot_est_population_densities) and args.ideal_summary_folder:
    len_gp_cpnum_alns.loc[len(ubs), cpnums_all] = len_gp_cpnum_alns.loc[:(len(ubs)-1), cpnums_all].sum()
    len_gp_cpnum_tps.loc[len(ubs), cpnums_all] = len_gp_cpnum_tps.loc[:(len(ubs)-1), cpnums_all].sum()
    len_gp_cpnum_positives.loc[len(ubs), cpnums_all] = len_gp_cpnum_positives.loc[:(len(ubs)-1), cpnums_all].sum()
    len_gp_cpnum_alns['total'] = len_gp_cpnum_alns[cpnums_all].sum(axis=1)
    len_gp_cpnum_tps['total'] = len_gp_cpnum_tps[cpnums_all].sum(axis=1)
    len_gp_cpnum_positives['total'] = len_gp_cpnum_positives[cpnums_all].sum(axis=1)
    ideal_counts_path = os.path.join(ideal_counts_folder, 'est_len-gps')
    if args.use_oom_len_gps:
        ideal_counts_path = os.path.join(ideal_counts_folder, 'order-of-magnitude_len-gps')
    if not(os.path.exists(ideal_counts_path)):
        os.mkdir(ideal_counts_path)
    write_counts(len_gp_cpnum_alns, len(ubs), cpnums_all, os.path.join(ideal_counts_path, 'copynum-aln_counts'))
    write_counts(len_gp_cpnum_tps, len(ubs), cpnums_all, os.path.join(ideal_counts_path, 'copynum-tp_counts'))
    write_counts(len_gp_cpnum_positives, len(ubs), cpnums_all, os.path.join(ideal_counts_path, 'copynum-positive_counts'))
    len_gp_cpnum_lbs.to_csv(os.path.join(args.ideal_summary_folder, 'copynum-aln_lower-bounds' + (args.use_oom_len_gps * '_order-of-magnitude') + '.csv'), index_label = 'Len. gp.',
        header = ['Min. len.', 'Max. len'] + list(map(lambda n: 'copy # ' + str(n), cpnums_all)))
    write_stats(compute_ideal_stats(len_gp_cpnum_tps.loc[len(ubs), cpnums_all], len_gp_cpnum_alns.loc[len(ubs), cpnums_all], len_gp_cpnum_positives.loc[len(ubs), cpnums_all]),
        os.path.join(ideal_summary_stats_folder, 'stats_agg' + args.use_oom_len_gps * '_order-of-magnitude'), len_gp_cpnum_tps.loc[len(ubs), cpnums_all].sum() / all_seqs.shape[0])

