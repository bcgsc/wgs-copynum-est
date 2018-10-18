import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys

SEQ_EST_ALN_FILE = sys.argv[1]
PLOTS_FILE_PREFIX = sys.argv[2]
NUM_CLASSES = 6

def fit_est_kdes(seqs, kdes, counts):
    vals = seqs[seqs.est1 == 1].mean_depth.values
    if len(vals) > 0:
        kdes[1] = sm.nonparametric.KDEUnivariate(vals)
        kdes[1].fit(bw='scott')
        counts[1] = len(vals)
    vals = seqs[seqs.est1 == 2].mean_depth.values
    if len(vals) > 0:
        kdes[2] = sm.nonparametric.KDEUnivariate(vals)
        kdes[2].fit(bw='scott')
        counts[2] = len(vals)
    vals = seqs[seqs.est1 == 3].mean_depth.values
    if len(vals) > 0:
        kdes[3] = sm.nonparametric.KDEUnivariate(vals)
        kdes[3].fit(bw='scott')
        counts[3] = len(vals)
    vals = seqs[seqs.est1 == 4].mean_depth.values
    if len(vals) > 0:
        kdes[4] = sm.nonparametric.KDEUnivariate(vals)
        kdes[4].fit(bw='scott')
        counts[4] = len(vals)
    vals = seqs[seqs.est1 >= 5].mean_depth.values
    if len(vals) > 0:
        kdes[5] = sm.nonparametric.KDEUnivariate(vals)
        kdes[5].fit(bw='scott')
        counts[5] = len(vals)

def fit_aln_kdes(seqs, kdes, counts):
    vals = seqs[seqs.alns == 0].mean_depth.values
    if len(vals) > 0:
        kdes[0] = sm.nonparametric.KDEUnivariate(vals)
        kdes[0].fit(bw='scott')
        counts[0] = len(vals)
    vals = seqs[(seqs.alns == 1) | (seqs.alns == 2)].mean_depth.values
    if len(vals) > 0:
        kdes[1] = sm.nonparametric.KDEUnivariate(vals)
        kdes[1].fit(bw='scott')
        counts[1] = len(vals)
    vals = seqs[(seqs.alns == 3) | (seqs.alns == 4)].mean_depth.values
    if len(vals) > 0:
        kdes[2] = sm.nonparametric.KDEUnivariate(vals)
        kdes[2].fit(bw='scott')
        counts[2] = len(vals)
    vals = seqs[(seqs.alns == 5) | (seqs.alns == 6)].mean_depth.values
    if len(vals) > 0:
        kdes[3] = sm.nonparametric.KDEUnivariate(vals)
        kdes[3].fit(bw='scott')
        counts[3] = len(vals)
    vals = seqs[(seqs.alns == 7) | (seqs.alns == 8)].mean_depth.values
    if len(vals) > 0:
        kdes[4] = sm.nonparametric.KDEUnivariate(vals)
        kdes[4].fit(bw='scott')
        counts[4] = len(vals)
    vals = seqs[seqs.alns >= 9].mean_depth.values
    if len(vals) > 0:
        kdes[5] = sm.nonparametric.KDEUnivariate(vals)
        kdes[5].fit(bw='scott')
        counts[5] = len(vals)

def plot_kdes(ax, kde_grid, est_kdes, aln_kdes, kde_all, est_counts, aln_counts, all_count):
    ax.plot(kde_grid, est_kdes[1].evaluate(kde_grid) * est_counts[1] / all_count, 'r--', lw=1, label = "Estimated copy # 1")
    ax.plot(kde_grid, est_kdes[2].evaluate(kde_grid) * est_counts[2] / all_count, 'g--', lw=1, label = "Estimated copy # 2")
    ax.plot(kde_grid, est_kdes[3].evaluate(kde_grid) * est_counts[3] / all_count, 'c--', lw=1, label = "Estimated copy # 3")
    ax.plot(kde_grid, est_kdes[4].evaluate(kde_grid) * est_counts[4] / all_count, 'm--', lw=1, label = "Estimated copy # 4")
    ax.plot(kde_grid, est_kdes[5].evaluate(kde_grid) * est_counts[5] / all_count, 'y--', lw=1, label = "Estimated copy # 5+")
    ax.plot(kde_grid, aln_kdes[0].evaluate(kde_grid) * aln_counts[0] / all_count, 'b', lw=1, label = "True copy # 0")
    ax.plot(kde_grid, aln_kdes[1].evaluate(kde_grid) * aln_counts[1] / all_count, 'r', lw=1, label = "True copy # 1")
    ax.plot(kde_grid, aln_kdes[2].evaluate(kde_grid) * aln_counts[2] / all_count, 'g', lw=1, label = "True copy # 2")
    ax.plot(kde_grid, aln_kdes[3].evaluate(kde_grid) * aln_counts[3] / all_count, 'c', lw=1, label = "True copy # 3")
    ax.plot(kde_grid, aln_kdes[4].evaluate(kde_grid) * aln_counts[4] / all_count, 'm', lw=1, label = "True copy # 4")
    ax.plot(kde_grid, aln_kdes[5].evaluate(kde_grid) * aln_counts[5] / all_count, 'y', lw=1, label = "True copy # 5+")
    ax.plot(kde_grid, kde_all.evaluate(kde_grid), 'k', lw=1, label = "All sequences")

est_kdes = [None] * NUM_CLASSES
est_counts = [None] * NUM_CLASSES
aln_kdes = [None] * NUM_CLASSES
aln_counts = [None] * NUM_CLASSES
kde_grid = np.linspace(0, 200, 200 * 100 + 1)

all_seqs = pd.read_csv(SEQ_EST_ALN_FILE)
all_seqs.rename(index=str, inplace=True,
            columns={ 'Length': 'length', 'Average depth': 'mean_depth', 'GC %': 'GC',
                      'Est. 1st label': 'est1', 'Est. 2nd label': 'est2', 'Est. 3rd label': 'est3',
                      'Ref. alns': 'alns', 'Ref. other alns': 'other_alns', 'Other alns CIGARs': 'other_aln_cigars', 'MAPQ (unique aln only)': 'unique_aln_mapq' })
all_seqs = all_seqs[all_seqs.mean_depth <= 800]

fit_est_kdes(all_seqs, est_kdes, est_counts)
fit_aln_kdes(all_seqs, aln_kdes, aln_counts)
vals = all_seqs.mean_depth.values
kde_all = sm.nonparametric.KDEUnivariate(vals)
kde_all.fit(bw='scott')
all_count = len(vals)
fig, ax = plt.subplots()
plot_kdes(ax, kde_grid, est_kdes, aln_kdes, kde_all, est_counts, aln_counts, all_count)
plt.legend()
ax.set_title('Densities for estimated and true copy numbers, sequences of all lengths')
fig.savefig(PLOTS_FILE_PREFIX + '.png')

lb = 0
for ub in [100, 1000, 10000, np.Inf]:
    seqs = all_seqs[(all_seqs.length >= lb) & (all_seqs.length < ub)]
    fit_est_kdes(seqs, est_kdes, est_counts)
    fit_aln_kdes(seqs, aln_kdes, aln_counts)
    vals = seqs.mean_depth.values
    kde_all = sm.nonparametric.KDEUnivariate(vals)
    kde_all.fit(bw='scott')
    all_count = len(vals)
    fig, ax = plt.subplots()
    plot_kdes(ax, kde_grid, est_kdes, aln_kdes, kde_all, est_counts, aln_counts, all_count)
    plt.legend()
    ax.set_title('Densities for estimated and true copy numbers, sequences with length in [' + str(lb) + ', ' + str(ub) + ')', fontsize=10)
    fig.savefig(PLOTS_FILE_PREFIX + '_len-gte' + str(lb) + 'lt' + str(ub) + '.png')
    lb = ub
