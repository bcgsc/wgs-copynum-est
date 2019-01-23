import csv
import numpy as np
import pandas as pd
import re
import sys

# counts of combinations of reference alignment # and estimated most likely copy #,
# e.g. how many sequences have 1 reference alignment, and are estimated to occur 1, 2, ..., n+ times
ALN_EST_COUNTS = sys.argv[1]
COUNTS_FILE = sys.argv[2]
NB_COUNTS_FILE = sys.argv[3] # non-binary: per-group figures
STATS_FILE = sys.argv[4]
NB_STATS_FILE = sys.argv[5]

pre_aln_est_combos = pd.read_csv(ALN_EST_COUNTS)
aln_est_combos = pre_aln_est_combos[1:][pre_aln_est_combos.columns[pre_aln_est_combos.columns.map(lambda c: re.search('^\d+', c)).notnull()][1:]]
aln_est_combos.rename(columns={k: v for (k, v) in list(map(lambda c: (c, int(re.search('^\d+', c)[0])), aln_est_combos.columns.tolist()))}, inplace=True)
LABELS = aln_est_combos.columns.size + 1

# TP, FP, FN, sensitivity (and variants?), F1
ONE_IDX = 0
MANY_IDX = 1

alnmt_counts = [0, 0]
alnmt_counts_nb = [0] * LABELS
alnmt_counts_nb[1] = aln_est_combos.loc[1].sum()
alnmt_counts[ONE_IDX] = alnmt_counts_nb[1]
for i in range(2, LABELS):
    alnmt_counts_nb[i] = aln_est_combos.loc[i].sum()
    alnmt_counts[MANY_IDX] += alnmt_counts_nb[i]

est_counts = [0, 0]
est_counts_nb = [0] * LABELS
est_counts_nb[1] = aln_est_combos[1].sum()
est_counts[ONE_IDX] = est_counts_nb[1]
for i in range(2, LABELS):
    est_counts_nb[i] = aln_est_combos[i].sum()
    est_counts[MANY_IDX] += est_counts_nb[i]

many_to_one = est_counts[ONE_IDX] - aln_est_combos.loc[1, 1]

with open(COUNTS_FILE, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Aln.\Est.', 'One', 'Many', 'Total'])
    writer.writerow(['One', aln_est_combos.loc[1, 1], alnmt_counts[ONE_IDX] - aln_est_combos.loc[1, 1], alnmt_counts[ONE_IDX]])
    writer.writerow(['Many', many_to_one, alnmt_counts[MANY_IDX] - many_to_one, alnmt_counts[MANY_IDX]])

with open(NB_COUNTS_FILE, 'w') as csvfile:
    writer = csv.writer(csvfile)
    cols = ['Aln.\Est.']
    cols.extend(map(lambda i: str(i), aln_est_combos.columns.values.tolist()))
    cols.append('Total')
    writer.writerow(cols)
    rows = list(map(lambda i: [str(i)], range(LABELS)))
    for i in range(1, LABELS):
        rows[i].extend(aln_est_combos.loc[i].tolist())
        rows[i].append(alnmt_counts_nb[i])
        writer.writerow(rows[i])

# F1 calculation
# Let c = # of correct +ves, p = precision, r = recall (a.k.a. TPR or sensitivity)
# p0 = # actually +ve, p1 = # classified as +ve
# Then p = c / p1, r = c / p0, and
# F1 = 2 / (1/p + 1/r) = 2 / (p1/c + p0/c) = 2c / (p0 + p1)
with open(STATS_FILE, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['TPR', 'FNR', 'TNR', 'FPR', 'PPV', 'FDR', 'F1'])
    tpr = aln_est_combos.loc[1, 1] * 1.0 / alnmt_counts[ONE_IDX]
    fpr = np.nan
    if alnmt_counts[MANY_IDX] > 0:
        fpr = many_to_one * 1.0 / alnmt_counts[MANY_IDX]
    ppv = np.nan
    if est_counts[ONE_IDX] > 0:
        ppv = aln_est_combos.loc[1, 1] * 1.0 / est_counts[ONE_IDX]
    writer.writerow([tpr, 1.0 - tpr, 1.0 - fpr, fpr, ppv, 1.0 - ppv, 2.0 * aln_est_combos.loc[1, 1] / (alnmt_counts[ONE_IDX] + est_counts[ONE_IDX])])

with open(NB_STATS_FILE, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Class \ Stat', 'TPR', 'FNR', 'PPV', 'FDR', 'F1'])
    rows = list(map(lambda i: [str(i)], range(LABELS)))
    correct_counts = 0
    for i in range(1, LABELS):
        if alnmt_counts_nb[i] > 0:
            tpr = aln_est_combos.loc[i, i] * 1.0 / alnmt_counts_nb[i]
        else:
            tpr = np.nan
        if est_counts_nb[i] > 0:
            ppv = aln_est_combos.loc[i, i] * 1.0 / est_counts_nb[i]
        else:
            ppv = np.nan
        f1 = np.nan
        if alnmt_counts_nb[i] + est_counts_nb[i] > 0:
            f1 = 2.0 * aln_est_combos.loc[i, i] / (alnmt_counts_nb[i] + est_counts_nb[i])
        rows[i].extend([tpr, 1 - tpr, ppv, 1 - ppv, f1])
        writer.writerow(rows[i])
        correct_counts += aln_est_combos.loc[i, i]
    overall_sensitivity = correct_counts * 1.0 / (alnmt_counts[ONE_IDX] + alnmt_counts[MANY_IDX])
    writer.writerow(['Non-binary overall', overall_sensitivity, 1 - overall_sensitivity, '', '', ''])

