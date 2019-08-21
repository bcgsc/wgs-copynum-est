import argparse
import csv
import numpy as np
import pandas as pd
import re
import sys

description = "Compute summary statistics for copy number estimator. Outputs counts and scores for binary and non-binary classification (results collapsed into 1-many categorisation for the former)."
argparser = argparse.ArgumentParser(description=description)
argparser.add_argument("aln_est_counts", type=str, help="CSV file listing alignment counts and assignments for classified sequences")
argparser.add_argument("counts_output_prefix", type=str, help="Prefix for output CSV files summarising counts for alignment-assignment combinations.")
argparser.add_argument("stats_output_prefix", type=str,
    help="Prefix for output CSV files summarising classification performance scores for alignment-assignment combinations.")
args = argparser.parse_args()

# counts of combinations of reference alignment # and estimated most likely copy #,
# e.g. how many sequences have 1 reference alignment, and are estimated to occur 1, 2, ..., n+ times
pre_aln_est_combos = pd.read_csv(args.aln_est_counts)
pre_aln_est_combos.set_index('Aln\Est', inplace=True)
aln_est_combos = pre_aln_est_combos[pre_aln_est_combos.columns[pre_aln_est_combos.columns.map(lambda c: re.search('^\d\.?\d?', c)).notnull()]]
aln_est_combos.rename(columns={ k: v for (k, v) in list(map(lambda c: (c, float(re.search('^\d\.?\d?', c)[0])), aln_est_combos.columns.tolist())) }, inplace=True)

LABELS = aln_est_combos.index.size
HALF, IDX1 = False, 1
if aln_est_combos.columns[1] == 0.5:
    HALF, IDX1 = True, 0.5

ONE_IDX, MANY_IDX = 1, 2

one_to_one = aln_est_combos.loc[IDX1:1.5, IDX1:1.5].sum().sum()
one_to_many = aln_est_combos.loc[IDX1:1.5, 2:].sum().sum()
many_to_one = aln_est_combos.loc[2:, IDX1:1.5].sum().sum()
many_to_many = aln_est_combos.loc[2:, 2:].sum().sum()
alnmt_counts_1many = [0, 0, 0]
alnmt_counts_1many[ONE_IDX] = aln_est_combos.loc[IDX1:1.5, IDX1:].sum().sum()
alnmt_counts_1many[MANY_IDX] = aln_est_combos.loc[2:, IDX1:].sum().sum()

with open(args.counts_output_prefix + '_1many.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Aln.\Est.', 'One', 'Many', 'Total'])
    writer.writerow(['One', one_to_one, one_to_many, alnmt_counts_1many[ONE_IDX]])
    writer.writerow(['Many', many_to_one, many_to_many, alnmt_counts_1many[MANY_IDX]])

est_counts_1many = [0, 0, 0]
est_counts_1many[ONE_IDX] = aln_est_combos.loc[IDX1:, IDX1:1.5].sum().sum()
est_counts_1many[MANY_IDX] = aln_est_combos.loc[IDX1:, 2:].sum().sum()

# F1 calculation
# Let c = # of correct +ves, p = precision, r = recall (a.k.a. TPR or sensitivity)
# p0 = # actually +ve, p1 = # classified as +ve
# Then p = c / p1, r = c / p0, and
# F1 = 2 / (1/p + 1/r) = 2 / (p1/c + p0/c) = 2c / (p0 + p1)

with open(args.stats_output_prefix + '_1many.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['TPR', 'FNR', 'TNR', 'FPR', 'PPV', 'FDR', 'F1'])
    tpr = np.nan
    if alnmt_counts_1many[ONE_IDX] > 0:
        tpr = one_to_one * 1.0 / alnmt_counts_1many[ONE_IDX]
    fpr = np.nan
    if alnmt_counts_1many[MANY_IDX] > 0:
        fpr = many_to_one * 1.0 / alnmt_counts_1many[MANY_IDX]
    ppv = np.nan
    if est_counts_1many[ONE_IDX] > 0:
        ppv = one_to_one * 1.0 / est_counts_1many[ONE_IDX]
    f1 = np.nan
    if alnmt_counts_1many[ONE_IDX] + est_counts_1many[ONE_IDX] > 0:
        f1 = 2.0 * one_to_one / (alnmt_counts_1many[ONE_IDX] + est_counts_1many[ONE_IDX])
    writer.writerow([tpr, 1.0 - tpr, 1.0 - fpr, fpr, ppv, 1.0 - ppv, f1])

if HALF:
    HALF_IDX = 0
    half_to_half = aln_est_combos.loc[0.5, 0.5]
    one_to_one = aln_est_combos.loc[1, 1]
    alnmt_counts_half1many = [0, 0, 0]
    alnmt_counts_half1many[HALF_IDX] = aln_est_combos.loc[0.5, 0.5:].sum()
    alnmt_counts_half1many[ONE_IDX] = aln_est_combos.loc[1, 0.5:].sum()
    alnmt_counts_half1many[MANY_IDX] = aln_est_combos.loc[2:, 0.5:].sum().sum()

    with open(args.counts_output_prefix + '_half1many.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Aln.\Est.', 'Half', 'One', 'Many', 'Total'])
        writer.writerow(['Half', half_to_half, aln_est_combos.loc[0.5, 1], aln_est_combos.loc[0.5, 2:].sum(), alnmt_counts_half1many[HALF_IDX]])
        writer.writerow(['One', aln_est_combos.loc[1, 0.5], one_to_one, aln_est_combos.loc[1, 2:].sum(), alnmt_counts_half1many[ONE_IDX]])
        writer.writerow(['Many', aln_est_combos.loc[2:, 0.5].sum(), aln_est_combos.loc[2:, 1].sum(), many_to_many, alnmt_counts_half1many[MANY_IDX]])

    est_counts_1many = [0, 0, 0]
    est_counts_1many[HALF_IDX] = aln_est_combos.loc[0.5:, 0.5].sum()
    est_counts_1many[ONE_IDX] = aln_est_combos.loc[0.5:, 1].sum()
    est_counts_1many[MANY_IDX] = aln_est_combos.loc[0.5:, 2:].sum().sum()

    def do_half_stats(cpnum, aln_to_est, aln, est):
        tpr = np.nan
        if aln > 0:
            tpr = aln_to_est * 1.0 / aln
        ppv = np.nan
        if est > 0:
            ppv = aln_to_est * 1.0 / est
        f1 = np.nan
        if aln + est > 0:
            f1 = 2.0 * aln_to_est / (aln + est)
        writer.writerow([cpnum, tpr, 1.0 - tpr, ppv, 1.0 - ppv, f1])

    with open(args.stats_output_prefix + '_half1many.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Copy #', 'TPR', 'FNR', 'PPV', 'FDR', 'F1'])
        do_half_stats('Half', half_to_half, alnmt_counts_half1many[HALF_IDX], est_counts_1many[HALF_IDX])
        do_half_stats('One', one_to_one, alnmt_counts_half1many[ONE_IDX], est_counts_1many[ONE_IDX])
        do_half_stats('Many', many_to_many, alnmt_counts_half1many[MANY_IDX], est_counts_1many[MANY_IDX])

aln_est_combos.to_csv(args.counts_output_prefix + '_full.csv', header=map(lambda i: str(int(i)) if i != 0.5 else str(i), aln_est_combos.columns.values.tolist()), index_label='Aln\Est')

alnmt_counts_full = pd.Series([0] * LABELS, index = aln_est_combos.index)
for i in alnmt_counts_full.index:
    alnmt_counts_full[i] = aln_est_combos.loc[i].sum()
est_counts_full = pd.Series([0] * LABELS, index = aln_est_combos.index) # columns should be equal
for i in est_counts_full.index:
    est_counts_full[i] = aln_est_combos[i].sum()

with open(args.stats_output_prefix + '_full.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Copy #', 'TPR', 'FNR', 'PPV', 'FDR', 'F1'])
    rows = list(map(lambda i: [str(int(i)) if i != 0.5 else str(i)], aln_est_combos.index.tolist()))
    correct_counts = 0
    for i in aln_est_combos.index:
        if alnmt_counts_full[i] > 0:
            tpr = aln_est_combos.loc[i, i] * 1.0 / alnmt_counts_full[i]
        else:
            tpr = np.nan
        if est_counts_full[i] > 0:
            ppv = aln_est_combos.loc[i, i] * 1.0 / est_counts_full[i]
        else:
            ppv = np.nan
        f1 = np.nan
        if alnmt_counts_full[i] + est_counts_full[i] > 0:
            f1 = 2.0 * aln_est_combos.loc[i, i] / (alnmt_counts_full[i] + est_counts_full[i])
        writer.writerow([i, tpr, 1 - tpr, ppv, 1 - ppv, f1])
        correct_counts += aln_est_combos.loc[i, i]
    overall_sensitivity = correct_counts * 1.0 / aln_est_combos.sum().sum()
    writer.writerow(['Overall', overall_sensitivity, 1 - overall_sensitivity, '', '', ''])

