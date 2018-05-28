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
OUTPUT_FILE = sys.argv[3]

LABELS = 6
alnmt_totals = []
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

tp = [0, 0]
fp = [0, 0]
fn = [0, 0]
sensitivity = [0, 0]

alnmt_counts = [0, 0]
alnmt_counts[ONE_IDX] = get_aln_total(aln_est_combos[1])
for i in range(2, LABELS):
    alnmt_counts[MANY_IDX] += get_aln_total(aln_est_combos[i])

est_counts = [0, 0]
est_counts[ONE_IDX] = get_col_total(aln_est_combos, 1, 1)
for i in range(2, LABELS):
    est_counts[MANY_IDX] += get_col_total(aln_est_combos, 1, i)

positives = [aln_est_combos[1][1], est_counts[MANY_IDX] - (alnmt_counts[ONE_IDX] - aln_est_combos[1][1])]

# F1 calculation
# Let c = # of correct +ves, p = precision (TP rate), r = recall (sensitivity)
# p0 = # actually +ve, p1 = # classified as +ve
# Then p = c / p1, r = c / p0, and
# F1 = 2 / (1/p + 1/r) = 2 / (p1/c + p0/c) = 2c / (p0 + p1)
f1 = [0, 0]
f1[ONE_IDX] = 2 * positives[ONE_IDX] / (alnmt_counts[ONE_IDX] + est_counts[ONE_IDX])
f1[MANY_IDX] = 2 * positives[MANY_IDX] / (alnmt_counts[MANY_IDX] + est_counts[MANY_IDX])

with open(OUTPUT_FILE, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Class\Stat', 'TPR', 'FPR', 'Est. total', 'Sensitivity', 'FNR', 'Aln. total', 'F1'])
    rows = [['One'], ['Many']]
    for idx in [ONE_IDX, MANY_IDX]:
        for denominator in [est_counts, alnmt_counts]:
            rows[idx].append(str(positives[idx] / denominator[idx]) + ' (' + str(positives[idx]) +')')
            ngtvs = denominator[idx] - positives[idx]
            rows[idx].append(str(ngtvs / denominator[idx]) + ' (' + str(ngtvs) + ')')
            rows[idx].append(denominator[idx])
        rows[idx].append(f1[idx])
    for r in rows:
        writer.writerow(r)

