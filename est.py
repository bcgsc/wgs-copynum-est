import array
import csv
import math as m
import numpy as np
import re
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import sys


def get_obs_in_range(gp, feature, inf, sup):
    return gp[np.where((gp[feature] > inf) & (gp[feature] <= sup))[0]]

def get_mean_dev(obs, feature, center):
    n = len(obs)
    dev_sum = 0
    for i in range(n):
        dev_sum += m.fabs(obs[i][feature] - center)
    return (dev_sum / n)
    
def adjust_wrt_nbr(gp, modes, idx, nidx, half, obs, feature):
    nbr_obs = get_obs_in_range(gp, feature, modes[nidx] - half, modes[nidx] + half)
    nbr_mean_dev = get_mean_dev(nbr_obs, feature, modes[nidx])
    for i in range(len(obs)):
        if m.fabs(modes[nidx] - obs[i][feature]) <= 2.5 * nbr_mean_dev: # between a rock and a hard place: seems like the least meddlesome intervention
            #obs[i]['naive_label'] = nidx
            counts[idx] -= 1
            counts[nidx] += 1

def push_mean_and_label(means_init, mean, label_mappings, idx):
    means_init.append(mean)
    label_mappings.append(idx)


FASTA_FILE = sys.argv[1]
KMER_LEN = int(sys.argv[2])
seq_kmers = array.array('L')
seq_avg_kmer_depths = array.array('d')

with open(FASTA_FILE) as fasta:
    line = fasta.readline()
    while line:
        if re.search('^>[0-9]', line):
            row = list(map(int, line[1:].split()))
            kmers = row[1] - KMER_LEN + 1
            seq_kmers.append(kmers)
            seq_avg_kmer_depths.append(row[2] / kmers)
        line = fasta.readline()

if len(seq_kmers) != len(seq_avg_kmer_depths):
    raise SystemExit('Error: kmer array length != average depth array length')

numseqs = len(seq_kmers)
#seqs = np.zeros(numseqs, dtype=[('kmers', np.uint64), ('avg_depth', np.float64), ('naive_label', np.uint64), ('likeliest_labels', np.uint64, (3,))])
seqs = np.zeros(numseqs, dtype=[('ID', np.uint64), ('kmers', np.uint64), ('avg_depth', np.float64), ('likeliest_labels', np.int, (3,))])
for i in range(numseqs):
    seqs[i]['ID'] = i
    seqs[i]['kmers'] = seq_kmers[i]
    seqs[i]['avg_depth'] = seq_avg_kmer_depths[i]
seqs.sort(order=['kmers', 'avg_depth']) # also sorts by avg_depth, after kmer

# (Basically) the next smallest possible equally sized bins
BIN_MINSIZE = 500
quantile = max(BIN_MINSIZE/numseqs, 0.02) * 100
if quantile > 2:
    quantile = 100 / m.floor(100 / quantile)

# Group sequences by length
percentiles_uniq = np.unique(np.percentile(seqs, np.arange(0, 100, quantile), axis=0, interpolation='lower')['kmers'])
sup = np.inf
len_gps = []
for i in range(len(percentiles_uniq) - 1, -1, -1):
    current_len_gp = seqs[np.where((seqs['kmers'] >= percentiles_uniq[i]) & (seqs['kmers'] < sup))[0]]
    if len(current_len_gp) >= BIN_MINSIZE:
        len_gps.append(current_len_gp)
        sup = percentiles_uniq[i]
len_gps.reverse()
# Ensure 1st group is large enough
if len(len_gps[0]) < BIN_MINSIZE:
    len_gps[1] = np.concatenate((len_gps[0], len_gps[1]))
    len_gps.pop(0)

with open('seqs_debug_1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['ID', 'Length (in kmers)', 'Average depth'])
    for i in range(numseqs):
        writer.writerow([seqs[i]['ID'], seqs[i]['kmers'], seqs[i]['avg_depth']])
with open('len_gps_debug_1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['ID', 'Length (in kmers)', 'Average depth'])
    for gp in len_gps:
        print('group length: ' + str(len(gp)))
        for i in range(len(gp)):
            writer.writerow([gp[i]['ID'], gp[i]['kmers'], gp[i]['avg_depth']])

# estimate 1st mode from longest seqs
len_gps_count = len(len_gps)
mode_est_gps_count = m.ceil(len_gps_count/10)
b4_mode_est_gp = len_gps_count - mode_est_gps_count - 1

# estimate mode from each selected group (of the longest sequences)
bw_est_grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.05, 2.0, 40)}, cv=20) # bandwidth selection
mode_sum = 0
ughcount = 0
for gp in len_gps[-1:b4_mode_est_gp:-1]:
    print('iter '+ str(ughcount))
    ughcount += 1
    curr_gp = np.copy(gp['avg_depth'][:,None])
    curr_gp.sort(axis=0)
    max_depth = curr_gp[-1][0]
    print(max_depth)

    bw_est_grid.fit(curr_gp)
    #print('bandwidth: ' + str(bw_est_grid.best_params_['bandwidth']))
    kde = KernelDensity(kernel='gaussian', bandwidth=bw_est_grid.best_params_['bandwidth']).fit(curr_gp)
    density_pts = np.linspace(0, max_depth, (max_depth+1) * 10 + 1)[:, None]
    log_dens = kde.score_samples(density_pts)
    print('max density: ' + str(np.exp(np.amax(log_dens))))
    first_mode = np.argmax(log_dens) / 10
    print('max density value: ' + str(first_mode))

    cutoff = np.argmax(log_dens) * 2
    if cutoff < 0.8 * max_depth:
        truncated_depths = curr_gp[np.where(curr_gp < cutoff)[0]]
        bw_est_grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.05, 1.3, 26)}, cv=20) # bandwidth selection
        bw_est_grid.fit(truncated_depths)
        print('bandwidth: ' + str(bw_est_grid.best_params_['bandwidth']))
        kde = KernelDensity(kernel='gaussian', bandwidth=bw_est_grid.best_params_['bandwidth']).fit(truncated_depths)
        max_depth = truncated_depths[-1][0]
        print(max_depth)
        density_pts = np.linspace(0, max_depth, (max_depth+1) * 10 + 1)[:, None]
        log_dens = kde.fit(truncated_depths).score_samples(density_pts)
        print('max density: ' + str(np.exp(np.amax(log_dens))))
        first_mode = np.argmax(log_dens) / 10
        print('max density value: ' + str(first_mode))
    print('')
    mode_sum += first_mode

# mixture modeling
#mode_depth_1 = 19
mode_depth_1 = mode_sum / mode_est_gps_count
half_component_len = mode_depth_1 / 2
print('mode: ' + str(mode_depth_1))

seqs.sort(order='ID')
csvfile = open('len_gps_debug_2.csv', 'w', newline='')
writer = csv.writer(csvfile, delimiter=',')
writer.writerow(['ID', 'Length (in kmers)', 'Average depth'])
# Vanilla instead of Bayesian GMM used because the latter could need manual tuning (e.g. of weight_concentration_prior)
for gp in len_gps:
    gp.sort(order='avg_depth')
    print('group length: ' + str(len(gp)))
    for i in range(len(gp)):
        writer.writerow([gp[i]['ID'], gp[i]['kmers'], gp[i]['avg_depth']])
    max_components = gp[-1]['avg_depth'] / mode_depth_1
    if max_components - m.floor(max_components) > 0.5:
        max_components = m.ceil(max_components)
    else:
        max_components = m.floor(max_components)
    modes = np.zeros(max_components, dtype=np.float64)
    #obs = np.empty(max_components, dtype=object)
    counts = np.zeros(max_components, dtype=np.uint64)
    means_init_py = []
    label_mappings_py = []
    # what follows (in the rest of the iteration) may be influenced by superstition about floating-point arithmetic and performance that I haven't had time to debunk
    mode = mode_depth_1
    for i in range(max_components):
        modes[i] = mode
        mode += mode_depth_1
    j = 0
    sup = 1.5 * mode_depth_1
    for i in range(len(gp)):
        while gp[i]['avg_depth'] > sup:
            sup += mode_depth_1
            j += 1
        #gp[i]['naive_label'] = j
        counts[j] += 1
    # Compare low-count components to neighbours and adjust if appropriate
    r_obs = get_obs_in_range(gp, 'avg_depth', modes[0], modes[0] + half_component_len)
    len_obs = len(r_obs)
    if len_obs > 0 and len_obs <= 5 and len_obs * 5 < counts[1]:
        adjust_wrt_nbr(gp, modes, 0, 1, half_component_len, r_obs, 'avg_depth')
    if counts[0] > 0:
        push_mean_and_label(means_init_py, modes[0], label_mappings_py, 0)
    for i in range(1, max_components-1): 
        if counts[i] > 0 and counts[i] <= 10:
            l_obs = get_obs_in_range(gp, 'avg_depth', modes[i] - half_component_len, modes[i])
            len_obs = len(l_obs)
            if len_obs > 0 and len_obs <= 5 and len_obs * 5 < counts[i-1]:
                adjust_wrt_nbr(gp, modes, i, i-1, half_component_len, l_obs, 'avg_depth')
            r_obs = get_obs_in_range(gp, 'avg_depth', modes[i], modes[i] + half_component_len)
            len_obs = len(r_obs)
            if len_obs > 0 and len_obs <= 5 and len_obs * 5 < counts[i+1]:
                adjust_wrt_nbr(gp, modes, i, i+1, half_component_len, r_obs, 'avg_depth')
        if counts[i] > 0:
            push_mean_and_label(means_init_py, modes[i], label_mappings_py, i)
    l_obs = get_obs_in_range(gp, 'avg_depth', modes[-1] - half_component_len, modes[-1])
    len_obs = len(l_obs)
    if len_obs > 0 and len_obs <= 5 and len_obs * 5 < counts[-2]:
        adjust_wrt_nbr(gp, modes, -1, -2, half_component_len, l_obs, 'avg_depth')
    if counts[-1] > 0:
        push_mean_and_label(means_init_py, modes[-1], label_mappings_py, max_components - 1)
    # Could also vary # of components across runs by including or excluding the components adjusted above,
    # and choosing the one with the optimal (lowest) BIC, but that could involve too many permutations
    means_init = np.array(means_init_py)[:, None]
    n_components = means_init.size
    label_mappings = np.array(label_mappings_py)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', max_iter=120, means_init=means_init, verbose=2)
    gmm.fit(gp['avg_depth'][:, None])
    obs_label_probs = gmm.predict_proba(gp['avg_depth'][:,None])
    for i in range(len(gp)):
        likeliest_labels_given = obs_label_probs[i].argsort()[-3:][::-1]
        if n_components < 3: # to prevent label_mappings' barfing due to dimensional inadequacy
            likeliest_labels_given = np.lib.pad(likeliest_labels_given, (0, 3 - n_components), 'constant', constant_values=(0,0))
        labels = label_mappings[likeliest_labels_given]
        for j in range(3):
            seqs[gp[i]['ID']]['likeliest_labels'][j] = labels[j]
        if n_components < 3:
            for j in range(2, n_components - 1, -1):
                seqs[gp[i]['ID']]['likeliest_labels'][j] = -1
        if i < 3:
            print(gp[i])
            print(seqs[gp[i]['ID']])

with open('sequence_labels.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    header = ['ID', 'Length (in kmers)', 'Average depth', 'Likeliest label', '2nd likeliest', '3rd likeliest']
    writer.writerow(header)
    for i in range(numseqs):
        row = [seqs[i]['ID'], seqs[i]['kmers'], seqs[i]['avg_depth']]
        row.extend(seqs[i]['likeliest_labels'])
        writer.writerow(row)

    #with open('copy_num_probs.csv', 'w', newline='') as csvfile:
    #    writer = csv.writer(csvfile, delimiter=',')
    #    for i in range(len(obs_label_probs)):
    #        row = [gp[i]['avg_depth']]
    #        row.extend(map(str, obs_label_probs[i]))
    #        writer.writerow(row)
    ## Just in case the number of components actually used is smaller than that initially found
    ## But this causes error in zeros() with "TypeError: 'numpy.float64' object cannot be interpreted as an integer"???
    ## max_label = np.amax(gp['naive_label'][:])
    #naive_to_est_labels = np.zeros((max_components, max_components, 3), dtype=np.uint64)
    #for i in range(len(gp)):
    #    likeliest_labels_given = obs_label_probs[i].argsort()[-3:][::-1]
    #    if n_components < 3: # to prevent label_mappings' barfing due to dimensional inadequacy
    #        likeliest_labels_given = np.lib.pad(likeliest_labels_given, (0, 3 - n_components), 'constant', constant_values=(0,0))
    #    gp[i]['likeliest_labels'] = label_mappings[likeliest_labels_given]
    #    if n_components < 3:
    #        for j in range(2, n_components - 1, -1):
    #            gp[i]['likeliest_labels'][j] = -1
    #    for j in range(3):
    #        naive_to_est_labels[gp[i]['naive_label']][gp[i]['likeliest_labels'][j]][j] += 1
    #gp_kmers = gp['kmers'][:]
    #with open(str(np.amin(gp_kmers)) + 'to' + str(np.amax(gp_kmers)) + 'kmers_labels', 'w', newline='') as csvfile:
    #    writer = csv.writer(csvfile, delimiter=',')
    #    header = ['Labels']
    #    header.extend(list(range(max_components)))
    #    writer.writerow(header)
    #    for i in range(max_components):
    #        row = [i]
    #        for j in range(max_components):
    #            row.append('/'.join(map(str, naive_to_est_labels[i][j])))
    #        writer.writerow(row)

# DEBUG / SANITY CHECKS
#
#for i in range(10):
#    print(repr(seqs[i, 0]) + ' ' + repr(seqs[i, 1]))
#
#numseqs = 0
#for i in range(len(len_gps)):
#    print(len(len_gps[i]))
#    numseqs += len(len_gps[i])
#print(len(len_gps))
#print(numseqs)
#print(len(seq_kmers))

# SCRAPS
#len_gps = np.asarray(len_gps_tmp)
