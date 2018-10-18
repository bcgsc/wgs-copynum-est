import csv
import numpy as np
import sys

def get_counts_from_row(row, array_row, from_cols, to_cols):
    for i in range(len(to_cols)):
        array_row[to_cols[i]] = row[from_cols[i]]

def get_aln_total(aln_row):
    total = 0
    for i in range(aln_row.size):
        total += aln_row[i]
    return total

def get_col_total(table, start_row, col_idx):
    total = 0
    for i in range(start_row, np.shape(table)[0]):
        total += table[i][col_idx]
    return total

# counts of combinations of reference alignment # and estimated most likely copy #,
# e.g. how many sequences have 1 reference alignment, and are estimated to occur 1, 2, 3, 4, or 5+ times
ALN_EST_COUNTS = sys.argv[1]
# for each # of reference alignments, counts of sequences with 1 copy # as the likeliest, or 2nd, 3rd, or 4th+ (other) likeliest estimate
ALN_EST_RANKS = sys.argv[2]
COUNTS_FILE = sys.argv[3]
NB_COUNTS_FILE = sys.argv[4] # non-binary: per-group figures
STATS_FILE = sys.argv[5]
NB_STATS_FILE = sys.argv[6]

LABELS = 6
aln_est_combos = np.zeros((LABELS,LABELS), dtype=np.int)
aln_est_ranks = np.zeros((LABELS,2), dtype=np.int)

with open(ALN_EST_COUNTS, newline='') as csvfile:
    from_cols = [1, 4, 7, 10, 13]
    to_cols = range(1, LABELS)
    reader = csv.reader(csvfile)
    next(reader)
    row = next(reader)
    get_counts_from_row(row, aln_est_combos[0], from_cols, to_cols)
    row_idx = 1
    for row in reader:
        get_counts_from_row(row, aln_est_combos[row_idx], from_cols, to_cols)
        row_idx += 1

with open(ALN_EST_RANKS, newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    row_idx = 1
    for row in reader:
        get_counts_from_row(row, aln_est_ranks[row_idx], [1, 2], [0, 1])
        row_idx += 1

# TP, FP, FN, sensitivity (and variants?), F1
ONE_IDX = 0
MANY_IDX = 1

alnmt_counts = [0, 0]
alnmt_counts_nb = [0] * LABELS
alnmt_counts_nb[1] = get_aln_total(aln_est_combos[1])
alnmt_counts[ONE_IDX] = alnmt_counts_nb[1]
for i in range(2, LABELS):
    alnmt_counts_nb[i] = get_aln_total(aln_est_combos[i])
    alnmt_counts[MANY_IDX] += alnmt_counts_nb[i]

est_counts = [0, 0]
est_counts_nb = [0] * LABELS
est_counts_nb[1] = get_col_total(aln_est_combos, 1, 1)
est_counts[ONE_IDX] = est_counts_nb[1]
for i in range(2, LABELS):
    est_counts_nb[i] = get_col_total(aln_est_combos, 1, i)
    est_counts[MANY_IDX] += est_counts_nb[i]

many_to_one = est_counts[ONE_IDX] - aln_est_combos[1][1]

with open(COUNTS_FILE, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Aln.\Est.', 'One', 'Many', 'Total'])
    writer.writerow(['One', aln_est_combos[1][1], alnmt_counts[ONE_IDX] - aln_est_combos[1][1], alnmt_counts[ONE_IDX]])
    writer.writerow(['Many', many_to_one, alnmt_counts[MANY_IDX] - many_to_one, alnmt_counts[MANY_IDX]])

with open(NB_COUNTS_FILE, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Aln.\Est.', '1', '2', '3', '4', '5+', 'Total'])
    rows = [['0'], ['1'], ['2'], ['3'], ['4'], ['5+']]
    for i in range(1, LABELS):
        rows[i].extend(aln_est_combos[i][1:LABELS])
        rows[i].append(alnmt_counts_nb[i])
        writer.writerow(rows[i])

# F1 calculation
# Let c = # of correct +ves, p = precision, r = recall (a.k.a. TPR or sensitivity)
# p0 = # actually +ve, p1 = # classified as +ve
# Then p = c / p1, r = c / p0, and
# F1 = 2 / (1/p + 1/r) = 2 / (p1/c + p0/c) = 2c / (p0 + p1)
with open(STATS_FILE, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['TPR', 'FNR', 'TNR', 'FPR', 'F1'])
    tpr = aln_est_combos[1][1] * 1.0 / alnmt_counts[ONE_IDX]
    fpr = many_to_one * 1.0 / alnmt_counts[MANY_IDX]
    writer.writerow([tpr, 1.0 - tpr, 1.0 - fpr, fpr, 2.0 * aln_est_combos[1][1] / (alnmt_counts[ONE_IDX] + est_counts[ONE_IDX])])

with open(NB_STATS_FILE, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Class \ Stat', 'TPR', 'FNR', 'PPV', 'FDR', 'F1'])
    rows = [['0'], ['1'], ['2'], ['3'], ['4'], ['5+']]
    correct_counts = 0
    for i in range(1, LABELS):
        if alnmt_counts_nb[i] > 0:
            tpr = aln_est_combos[i][i] * 1.0 / alnmt_counts_nb[i]
        else:
            tpr = np.nan
        if est_counts_nb[i] > 0:
            ppv = aln_est_combos[i][i] * 1.0 / est_counts_nb[i]
        else:
            ppv = np.nan
        rows[i].extend([tpr, 1 - tpr, ppv, 1 - ppv, 2.0 * aln_est_combos[i][i] / (alnmt_counts_nb[i] + est_counts_nb[i])])
        writer.writerow(rows[i])
        correct_counts += aln_est_combos[i][i]
    overall_sensitivity = correct_counts * 1.0 / (alnmt_counts[ONE_IDX] + alnmt_counts[MANY_IDX])
    writer.writerow(['Non-binary overall', overall_sensitivity, 1 - overall_sensitivity, '', '', ''])
