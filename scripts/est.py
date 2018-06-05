import array
import csv
import math as m
import numpy as np
import re
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import sys


def compute_gc_content(seq):
    gc_count = 0
    for b in seq:
        if b == 'G' or b == 'C':
            gc_count += 1
    return (gc_count * 100 / len(seq))

FASTA_FILE = sys.argv[1]
KMER_LEN = int(sys.argv[2])
OUTPUT_DIR = sys.argv[3]
seq_lens = array.array('L')
seq_kmers = array.array('L')
seq_avg_kmer_depths = array.array('d')
seq_gc_contents = array.array('d')

with open(FASTA_FILE) as fasta:
    line = fasta.readline()
    while line:
        if re.search('^>[0-9]', line):
            row = list(map(int, line[1:].split()))
            seq_lens.append(row[1])
            kmers = row[1] - KMER_LEN + 1
            seq_kmers.append(kmers)
            seq_avg_kmer_depths.append(row[2] / kmers)
        else:
            seq_gc_contents.append(compute_gc_content(line))
        line = fasta.readline()

if len(seq_kmers) != len(seq_avg_kmer_depths):
    raise SystemExit('Error: kmer array length != average depth array length')

numseqs = len(seq_kmers)
seqs = np.zeros(numseqs, dtype=[('ID', np.uint64), ('len', np.uint64), ('kmers', np.uint64), ('avg_depth', np.float64), ('modex', np.float64), ('est_gp', np.uint64), ('gc', np.float64), ('likeliest_labels', np.int, (3,))])
for i in range(numseqs):
    seqs[i]['ID'] = i
    seqs[i]['len'] = seq_lens[i]
    seqs[i]['kmers'] = seq_kmers[i]
    seqs[i]['avg_depth'] = seq_avg_kmer_depths[i]
    seqs[i]['est_gp'] = -1
    seqs[i]['gc'] = seq_gc_contents[i]
    for j in range(3):
        seqs[i]['likeliest_labels'][j] = sys.maxsize
seqs.sort(order=['kmers', 'avg_depth']) # also sorts by avg_depth, after kmer

# (Basically) the next smallest possible equally sized bins
BIN_MINSIZE = min(500, numseqs)
quantile = max(BIN_MINSIZE/numseqs, 0.02) * 100
if quantile > 2:
    quantile = 100 / m.floor(100 / quantile)

# Group sequences by length
percentiles_uniq = np.unique(np.percentile(seqs['kmers'], np.arange(0, 100, quantile), axis=0, interpolation='lower'))
sup = np.inf
len_gps = []
for i in range(len(percentiles_uniq) - 1, -1, -1):
    current_len_gp = seqs[np.where((seqs['kmers'] >= percentiles_uniq[i]) & (seqs['kmers'] < sup))[0]]
    if len(current_len_gp) >= BIN_MINSIZE:
        len_gps.append(current_len_gp)
        sup = percentiles_uniq[i]
len_gps.reverse()
# Ensure 1st group is large enough ... Not that it matters with the aggregation into orders of magnitude for actual use in estimation
if len(len_gps[0]) < BIN_MINSIZE and len(len_gps) > 1:
    len_gps[1] = np.concatenate((len_gps[0], len_gps[1]))
    len_gps.pop(0)

# estimate 1st mode from longest seqs
bw_est_grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.05, 2.0, 40)}, cv=20) # bandwidth selection; default cross-validation (o/w specify e.g. cv=20)
longest_gp = np.copy(len_gps[-1]['avg_depth'][:,None])
longest_gp.sort(axis=0)
max_depth = longest_gp[-1][0]
bw_est_grid.fit(longest_gp) # estimate best bandwidth for KDE
kde = KernelDensity(kernel='gaussian', bandwidth=bw_est_grid.best_params_['bandwidth']).fit(longest_gp)
density_pts = np.linspace(0, max_depth, (max_depth+1) * 10 + 1)[:, None]
log_dens = kde.score_samples(density_pts)
mode1_depth = np.argmax(log_dens) / 10
if np.var(longest_gp) / np.mean(longest_gp) >= 5: # dispersion index
    truncated_depths = longest_gp[np.where(longest_gp < min(first_mode * 10, np.percentile(longest_gp, 99.8)))[0]]
    bw_est_grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.05, 1.3, 26)}, cv=20) # bandwidth selection
    bw_est_grid.fit(truncated_depths)
    kde = KernelDensity(kernel='gaussian', bandwidth=bw_est_grid.best_params_['bandwidth']).fit(truncated_depths)
    max_depth = truncated_depths[-1][0]
    density_pts = np.linspace(0, max_depth, (max_depth+1) * 10 + 1)[:, None]
    log_dens = kde.score_samples(density_pts)
    mode1_depth = (np.argmax(log_dens) - 1) / 10

overall_80th_pctl = np.percentile(seqs['avg_depth'], 80, interpolation='higher')
overall_90th_pctl = np.percentile(seqs['avg_depth'], 90, interpolation='higher')
cutoffs = []
gmm_gp_maxlen = []
gmm_gp_maxlabel = []
gmm_wts = []
gmm_means = []
gmm_vars = []
seqs.sort(order='ID')
min_count = 25
for len_gp_idx in range(len(len_gps)):
    gp = np.copy(len_gps[len_gp_idx])
    gmm_gp_maxlen.append(np.amax(gp['len']))
    gp.sort(order='avg_depth')
    gp_80th_pctl = np.percentile(gp['avg_depth'], 80, interpolation='higher')
    gp_90th_pctl = np.percentile(gp['avg_depth'], 90, interpolation='higher')
    n_components = 1
    count = 0
    sup = 1.5 * mode1_depth
    for i in range(len(gp)):
        if gp[i]['avg_depth'] > sup:
            if count < min_count:
                break
            sup += mode1_depth
            if gp[i]['avg_depth'] > sup:
                break
            n_components += 1
            count = 0
        count += 1
    if count < min_count:
        n_components -= 1
    if overall_80th_pctl < gp_80th_pctl:
        use_pctl = gp_80th_pctl
    else:
        use_pctl = overall_80th_pctl
    pctls_in_modes = use_pctl / mode1_depth
    closest_mode_to_pctl = m.floor(use_pctl / mode1_depth)
    if pctls_in_modes - closest_mode_to_pctl > 0.5:
        closest_mode_to_pctl += 1
    n_components = min(n_components, closest_mode_to_pctl)
    gmm_gp_maxlabel.append(n_components - 1)
    max_for_kde = max((n_components + 2) * mode1_depth, gp_90th_pctl)
    obs_for_kde = gp[np.where(gp['avg_depth'] <= max_for_kde)[0]]['avg_depth'][:, None]
    bw_est_grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.05, 1.5, 30)}, cv=5)
    bw_est_grid.fit(obs_for_kde)
    kde = KernelDensity(kernel='gaussian', bandwidth=bw_est_grid.best_params_['bandwidth']).fit(obs_for_kde)
    density_pts = np.linspace(0, max_for_kde, max_for_kde * 20 + 1)[:, None]
    log_dens = kde.score_samples(density_pts)
    start = m.ceil(n_components * mode1_depth * 20) + 1
    depth_cutoff = (start + np.argmin(log_dens[start:m.ceil((n_components + 1) * mode1_depth * 20 + 2)]) - 1) / 20
    cutoffs.append(str(depth_cutoff))
    gp_for_gmm = gp[np.where(gp['avg_depth'] <= depth_cutoff)[0]]['avg_depth'][:, None]
    # GMM: spherical same as full for 1D estimation
    # Vanilla instead of Bayesian GMM used because the latter could need manual tuning (e.g. of weight_concentration_prior)
    gmm = GaussianMixture(n_components=n_components, covariance_type='spherical', max_iter=120, means_init=(np.arange(1, n_components+1) * mode1_depth)[:, None], verbose=2)
    gmm.fit(gp_for_gmm)
    obs_label_probs = gmm.predict_proba(gp_for_gmm)
    gmm_wts.append(gmm.weights_)
    gmm_means.append(gmm.means_)
    gmm_vars.append(gmm.covariances_)
    for i in range(len(gp_for_gmm)):
        seqs[gp[i]['ID']]['est_gp'] = len_gp_idx
        likeliest_labels_given = obs_label_probs[i].argsort()[-3:][::-1]
        for j in range(min(3, n_components)):
            seqs[gp[i]['ID']]['likeliest_labels'][j] = likeliest_labels_given[j]
        if n_components < 3:
            for j in range(2, n_components - 1, -1):
                seqs[gp[i]['ID']]['likeliest_labels'][j] = -1
    for i in range(len(gp_for_gmm), len(gp)):
        seqs[gp[i]['ID']]['est_gp'] = len_gp_idx

with open(OUTPUT_DIR + '/log.txt', 'w', newline='') as f:
    f.write('median unitig avg k-mer depth: ' + str(np.median(seqs['avg_depth'])) + '\n')
    f.write('mode: ' + str(mode1_depth) + '\n')
    f.write('80th percentile of average unitig kmer depth: ' + str(overall_80th_pctl) + '\n')
    f.write('max. depths for inclusion in estimation: ' + ' / '.join(cutoffs) + '\n')

with open(OUTPUT_DIR + '/params.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    header = ['Group max. len.', 'Group max. label', 'Component #', 'Weight', 'Mean', 'Covariance']
    writer.writerow(header)
    for gpidx in range(len(gmm_gp_maxlen)):
        for i in range(len(gmm_wts[gpidx])):
            writer.writerow([gmm_gp_maxlen[gpidx], gmm_gp_maxlabel[gpidx], i, gmm_wts[gpidx][i], gmm_means[gpidx][i][0], gmm_vars[gpidx][i]])

seqs.sort(order=['len', 'avg_depth'])
with open(OUTPUT_DIR + '/sequence-labels.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    header = ['ID', 'Length', 'Average depth', '1st Mode X', 'GC %', 'Group max. label', 'Likeliest label', '2nd likeliest', '3rd likeliest']
    writer.writerow(header)
    for i in range(numseqs):
        seqs[i]['modex'] = seqs[i]['avg_depth'] / mode1_depth
        row = [seqs[i]['ID'], seqs[i]['len'], seqs[i]['avg_depth'], seqs[i]['modex'], seqs[i]['gc'], gmm_gp_maxlabel[seqs[i]['est_gp']]]
        row.extend(seqs[i]['likeliest_labels'])
        writer.writerow(row)

