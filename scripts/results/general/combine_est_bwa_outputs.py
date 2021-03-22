import argparse
import math as m
import numpy as np
import os
import pandas as pd
import re
import statsmodels.api as sm
import sys

if os.getenv('WGS_COPYNUM_EST_HOME'):
    sys.path.insert(0, os.path.join(os.getenv('WGS_COPYNUM_EST_HOME'), 'scripts'))
else:
    raise RuntimeError('Please set environment variable WGS_COPYNUM_EST_HOME before running script')

import utils


argparser = argparse.ArgumentParser(description='Combine BWA alignment and copy number estimator outputs for classified sequences')
argparser.add_argument('--haploid', action='store_true', help='Dataset comes from haploid rather than diploid genome')
len_strata_help = 'Process sequences from/using length strata: unspecified (default, used in estimation), o (order of magnitude i.e. 0+, 100+, 1000+, 10000+ bps), k (k-mers only)'
argparser.add_argument('--use_length_strata', type=str, nargs='?', default='e', help=len_strata_help)
#argparser.add_argument('--max_cpnum', type=int, nargs='?', default=2, help='Largest copy number to evaluate performance for (all larger numbers collapsed into this one)')
lax_best_alnmt_help = 'Count multiplicity of best alignment subject to minimum sequence identity (seq_identity_threshold) but not best-alignment mean + stdev threshold'
argparser.add_argument('--lax_best_alnmt', action='store_true', help=lax_best_alnmt_help)
# use case for next option: highly heterozygous genomes, which have much lower contig length upper bounds
argparser.add_argument('--no_seq_identity_matching', action='store_true', help='Never use sequence identity for match counting (instead use edit distance stats in all cases)')
argparser.add_argument('--seq_identity_matching', action='store_true', help='Use sequence identity for match counting in all cases (instead of using edit distance stats in some)')
argparser.add_argument('--seq_identity_threshold', type=float, nargs='?', default=0.99, help='Minimum sequence identity to count alignment as match')
argparser.add_argument('--collapse_highest_est_cpnums', action='store_true', help="Collapse estimated copy numbers exceeding the estimator's maximum (use case Unicycler)")
argparser.add_argument('est_output', type=str, help='CSV file listing sequence data and classifications')
argparser.add_argument('bwa_parse_output', type=str, help='Parsed contig data from BWA reference alignment output SAM file')
argparser.add_argument('est_len_gp_stats', type=str, nargs='?', help='CSV file listing sequence length groups used in classification, with summary statistics')
args = argparser.parse_args()

def count_and_write(seqs, filename):
    est_components = [0]
    max_components_est = seqs.len_gp_max_cpnum_est.max()
    if max_components_est > 0.5:
        max_components_est = int(max_components_est)
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
            est_cpnum_condition = (seqs.copynum_est == j)
            if args.collapse_highest_est_cpnums and (j >= est_components[-1]):
                est_cpnum_condition = (seqs.copynum_est >= j)
            seqs_ij = seqs[(((seqs.len_gp_max_cpnum_est >= i) & (seqs.aln_match_count == i)) | ((seqs.len_gp_max_cpnum_est == i) & (seqs.aln_match_count > i))) & est_cpnum_condition]
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


if args.use_length_strata != 'k' and not(args.est_len_gp_stats):
    raise ValueError('CSV file listing sequence length groups used in classification required unless processing k-mers only')

seqs = pd.read_csv(args.est_output)
cols_from_to = { 'Length': 'length', 'Average k-mer depth': 'avg_depth', '1st Mode X': 'modex', 'GC %': 'GC', 'Estimation length group': 'len_gp', 'Likeliest copy #': 'copynum_est' }
seqs.rename(columns=cols_from_to, inplace=True)
seqs.set_index('ID', inplace=True)

print(not(args.haploid) and args.use_length_strata == 'k')
HALF = False
if not(args.haploid) and args.use_length_strata == 'k': # for GenomeScope evaluation and comparison...
    HALF = True
    seqs = seqs.loc[seqs.length == seqs.length.min()]
    if args.est_len_gp_stats:
        len_gp_stats = pd.read_csv(args.est_len_gp_stats)
        seqs.loc[:, 'len_gp_max_cpnum_est'] = len_gp_stats['Largest copy # estimated'].iloc[0]
    else:
        seqs.loc[:, 'len_gp_max_cpnum_est'] = seqs.copynum_est.max()
    print(seqs.shape)
elif args.est_len_gp_stats:
    len_gp_stats = pd.read_csv(args.est_len_gp_stats)
    HALF = ((len_gp_stats['Smallest copy # estimated'] == 0.5).sum() > 0)
    len_gp_est_component_counts = len_gp_stats['Largest copy # estimated'].tolist()
    seqs['len_gp_max_cpnum_est'] = seqs.len_gp.apply(lambda gp: len_gp_est_component_counts[int(gp)])
    del len_gp_est_component_counts

seq_alns = pd.read_csv(args.bwa_parse_output, delimiter='\t')
seq_alns.rename(columns={'sequence_identity': 'seq_identity'}, inplace=True)
# no need to sort by ID, since seq_alns is already sorted in the order needed
nonclipped_alns = seq_alns.loc[~(seq_alns.CIGAR.str.contains('S|H'))]

if args.use_length_strata != 'k':
    mins, maxs = len_gp_stats['Min. len.'].values, len_gp_stats['Max. len.'].values
    len_gps_count, len_gp_idxs = len(mins), range(len(mins))
    use_seq_iden_start_idx = len_gps_count

if args.seq_identity_matching:
    seq_aln_match_counts = nonclipped_alns.groupby('ID').seq_identity.apply(lambda i: (i >= args.seq_identity_threshold).sum())
    seq_aln_match_counts.name = 'aln_match_count'
    seqs = seqs.join(seq_aln_match_counts)
elif args.use_length_strata != 'k':
    seq_gps = list(map(lambda i: seqs.loc[(seqs.length >= mins[i]) & (seqs.length <= maxs[i])], len_gp_idxs))
    aln_gps = list(map(lambda i: nonclipped_alns.loc[(nonclipped_alns.length >= mins[i]) & (nonclipped_alns.length <= maxs[i])], len_gp_idxs))
    # Don't recall exactly why I decided to do the following instead of e.g. some per-sequence distance/length distribution summary, e.g. mean
    # Maybe I saw/thought those would be too similar across length strata
    edit_dist_sums = list(map(lambda seqs: seqs.total_edit_dist.sum(), aln_gps))
    length_sums = list(map(lambda seqs: seqs.length.sum(), seq_gps))
    dist_len_ratios = list(map(lambda sums: sums[0] / sums[1], zip(edit_dist_sums, length_sums)))
    median_lengths = list(map(lambda seqs: seqs.length.median(), seq_gps))
    if not args.no_seq_identity_matching:
        diffs = [0.0] * (len_gps_count - 2)
        if len(diffs) > 0:
            for i in len_gp_idxs[1:-1]:
                # Find "elbow" in ((sum of edit distances)/(sum of lengths)) vs. median length curve
                reg1 = sm.OLS(dist_len_ratios[:(i+1)], sm.add_constant(median_lengths[:(i+1)])).fit().params[1]
                reg2 = sm.OLS(dist_len_ratios[i:], sm.add_constant(median_lengths[i:])).fit().params[1]
                if (reg1 < 0) and (reg2 < 0):
                    diffs[i-1] = abs(reg1 - reg2)
            use_seq_iden_start_idx = np.argmax(diffs) + 2

if not args.seq_identity_matching:
    grouped_edit_dists = nonclipped_alns.groupby(['ID', 'MD', 'edit_distance'], sort=False).edit_distance
    # it doesn't matter whether I use min() or some other fn; it's just to enable groupby()
    best_edit_dists = grouped_edit_dists.min().groupby(level=[0]).nth(0)
    best_edit_dists.name = 'best_edit_dist'
    seqs = seqs.join(best_edit_dists)
    print(seqs.shape)
    aln_match_counts = grouped_edit_dists.count().groupby(level=[0]).nth(0)
    aln_match_counts.name = 'aln_match_count'
    seqs = seqs.join(aln_match_counts)
    print(seqs.shape)
    if args.use_length_strata == 'k':
        print(seqs.head())
        gp_best_edit_dist_summary = seqs['best_edit_dist'].describe()
        if not(args.lax_best_alnmt):
            seqs.loc[:, 'max_edit_dist'] = gp_best_edit_dist_summary['mean'] + gp_best_edit_dist_summary['std']
        seqs.loc[:, 'aln_match_count'] = seqs.apply(lambda seq: seq.aln_match_count if seq.best_edit_dist <= seq.max_edit_dist else 0, axis=1)
        print(seqs.describe())
    else:
        seqs.loc[:, 'max_edit_dist'] = np.nan
        if args.lax_best_alnmt:
            seqs.loc[:, 'max_edit_dist'] = (1 - args.seq_identity_threshold) * seqs.length
        for i in range(use_seq_iden_start_idx):
            len_condition = (seqs.length >= mins[i]) & (seqs.length <= maxs[i])
            gp_best_edit_dist_summary = seqs.loc[len_condition, 'best_edit_dist'].describe()
            if not(args.lax_best_alnmt):
                seqs.loc[len_condition, 'max_edit_dist'] = gp_best_edit_dist_summary['mean'] + gp_best_edit_dist_summary['std']
            seqs.loc[len_condition, 'aln_match_count'] = seqs.loc[len_condition].apply(lambda seq: seq.aln_match_count if seq.best_edit_dist <= seq.max_edit_dist else 0, axis=1)
        if use_seq_iden_start_idx:
            for i in range(use_seq_iden_start_idx, len_gps_count):
                aln_len_condition = (nonclipped_alns.length >= mins[i]) & (nonclipped_alns.length <= maxs[i])
                seq_aln_match_counts = nonclipped_alns.loc[aln_len_condition].groupby('ID').seq_identity.apply(lambda i: (i >= args.seq_identity_threshold).sum())
                len_condition = (seqs.length >= mins[i]) & (seqs.length <= maxs[i])
                seqs.loc[len_condition, 'aln_match_count'] = seq_aln_match_counts.reindex(seqs.loc[len_condition].index)
                seqs.loc[len_condition & seqs.aln_match_count.isnull(), 'aln_match_count'] = 0

if not(args.haploid):
    seqs['aln_match_count'] = seqs.aln_match_count.apply(lambda i: i if np.isnan(i) else m.ceil(i/2.0) if (i != 1 or not(HALF)) else 0.5) # alnmt counts still diploid

seqs.sort_values(by=['length', 'avg_depth'], inplace=True)
write_cols = ['length', 'avg_depth', 'GC', 'copynum_est', 'aln_match_count']
header = ['Length', 'Average depth', 'GC %', 'Likeliest copy #', 'Alignments (alns)']
seqs.loc[:, write_cols].to_csv('seq-est-and-aln.csv', header=header, index_label='ID')

if args.use_length_strata == 'o':
    seqs['len_gp'] = np.floor(np.log10(seqs.length)) - 1 # Assume all sequences have length > 10
    seqs.loc[seqs.len_gp > 3, 'len_gp'] = 3
    mins = [0, 100, 1000, 10000]
    maxs = [99, 999, 9999, np.inf]
    len_gp_est_component_counts = []
    for i in range(len(mins)):
        lt_cdtn = (len_gp_stats['Min. len.'] < mins[i]) & (len_gp_stats['Max. len.'] < mins[i])
        gt_cdtn = (len_gp_stats['Min. len.'] > maxs[i]) & (len_gp_stats['Max. len.'] > maxs[i])
        len_gp_est_component_counts.append(len_gp_stats.loc[~(lt_cdtn | gt_cdtn), 'Largest copy # estimated'].max())
    seqs['len_gp_max_cpnum_est'] = seqs.len_gp.apply(lambda gp: len_gp_est_component_counts[int(gp)])
    del len_gp_est_component_counts
elif args.use_length_strata != 'k':
    mins, maxs = len_gp_stats['Min. len.'].values, len_gp_stats['Max. len.'].values

if args.use_length_strata == 'k':
    seqlen = seqs.length.iloc[0]
    minmaxes = [(seqlen, seqlen)]
    count_files = ['aln-est_counts.csv']
elif args.use_length_strata == 'o':
    minmaxes = zip(mins, maxs)
    count_files = ['aln-est_counts_gte0lte99.csv', 'aln-est_counts_gte100lte999.csv', 'aln-est_counts_gte1000lte9999.csv', 'aln-est_counts_gte10000ltinf.csv']
else:
    minmaxes = zip(mins, maxs)
    count_files = ['aln-est_counts_'] * len_gp_stats.shape[0]
    for i in range(len(mins)):
        ub_str = 'lt' + (maxs[i] < np.inf) * 'e' + str(maxs[i])
        count_files[i] = count_files[i] + 'gte' + str(mins[i]) + ub_str + '.csv'

seqs_aln_defined = seqs.loc[seqs.aln_match_count.notna()]
len_gp_conditions = list(map(lambda bds: (seqs_aln_defined.length >= bds[0]) & (seqs_aln_defined.length <= bds[1]), minmaxes))
for i in range(len(len_gp_conditions)):
    count_and_write(seqs_aln_defined.loc[len_gp_conditions[i]], count_files[i])
count_and_write(seqs_aln_defined, 'aln-est_counts.csv')

