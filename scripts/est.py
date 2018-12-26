# TODO: Delete unusued imports
import array
import csv
import datetime
from lmfit import Model, Parameter, Parameters
from lmfit.models import ConstantModel, GaussianModel
import math as m
import numpy as np
import pandas as pd
import re
from scipy import optimize, stats
from scipy.special import gamma as gamma_fn
import statsmodels.api as sm
#from sympy import Symbol, nsolve
import sys
import time


def compute_gc_content(seq):
    gc_count = 0
    for b in seq:
        if b == 'G' or b == 'C':
            gc_count += 1
    return (gc_count * 100 / len(seq))

def get_length_gps_for_est(seqs, len_percentiles_uniq, bin_minsize):
    length_gps = []
    ub = np.Inf
    lb_idx = len(len_percentiles_uniq) - 1
    while len(seqs[seqs.len < ub]) >= bin_minsize:
        while len(seqs[(seqs.len < ub) & (seqs.len >= len_percentiles_uniq[lb_idx])]) < bin_minsize:
            lb_idx -= 1
        i = 1
        base_gp = seqs[(seqs.len < ub) & (seqs.len >= len_percentiles_uniq[lb_idx])].mean_kmer_depth.values
        while stats.ks_2samp(base_gp, seqs[(seqs.len < ub) & (seqs.len >= len_percentiles_uniq[lb_idx - i])].mean_kmer_depth.values).pvalue > 0.05:
            i += 1
            if lb_idx < i:
                break
        if lb_idx < i:
            lb = -np.inf
        else:
            lb = len_percentiles_uniq[lb_idx - i + 1]
            if len(seqs[seqs.len < lb]) < len(seqs[(seqs.len < ub) & (seqs.len >= lb)]): # unlikely; hopefully never happens
                lb = -np.inf
            else:
                lb = len_percentiles_uniq[lb_idx - i + 1]
        length_gp = seqs[(seqs.len < ub) & (seqs.len >= lb)]
        length_gps.append(length_gp)
        bin_minsize = len(length_gp)
        ub = lb
    length_gps.reverse()
    return length_gps

def val_to_grid_idx(val, minval, density):
    return (density * (val - minval))

def grid_idx_to_val(idx, grid_density, minval):
    return ((idx / grid_density) + minval)

def get_approx_component_obs(data, mode, mode1, data_min): # assumes symmetric distribution
    half = mode1 / 2.0
    return data[(data >= max(data_min, mode - half)) & (data <= mode + half)]

def get_density_for_idx(idx, density):
    return (density[m.floor(idx)] + (idx - m.floor(idx)) * (density[m.ceil(idx)] - density[m.floor(idx)]))

def init_gaussian(i, component_weights):
    model = None
    if component_weights[i] > 0:
        prefix = 'gauss' + str(i) + '_'
        model = GaussianModel(prefix=prefix)
    return model

# TODO: Rm debug statements
def gamma(x, shape, loc, scale):
    #debug_file.write('shape: ' + str(shape) + ', loc: ' + str(loc) + ', scale: ' + str(scale))
    #debug_file.write(str(stats.gamma.pdf(x, shape, loc, scale)))
    #debug_file.write('')
    return stats.gamma.pdf(x, shape, loc, scale)


UNITIGS_FILE = sys.argv[1] # FASTA format
KMER_LEN = int(sys.argv[2])
OUTPUT_DIR = sys.argv[3]
NONNEG_CONSTANT = 1.e-12

seq_lens = array.array('L')
seq_mean_kmer_depths = array.array('d')
seq_gc_contents = array.array('d')

with open(UNITIGS_FILE) as unitigs:
    line = unitigs.readline()
    while line:
        if re.search('^>[0-9]', line):
            row = list(map(int, line[1:].split()))
            seq_lens.append(row[1])
            kmers = row[1] - KMER_LEN + 1
            seq_mean_kmer_depths.append(row[2] / kmers)
        else:
            seq_gc_contents.append(compute_gc_content(line))
        line = unitigs.readline()

# TODO: Remove modex?
numseqs = len(seq_mean_kmer_depths)
seqs = pd.DataFrame(columns=['ID', 'len', 'mean_kmer_depth', 'modex', 'gc', 'est_gp', 'likeliest_copynum'])
seqs['ID'] = range(numseqs)
seqs['len'] = seq_lens
seqs['mean_kmer_depth'] = seq_mean_kmer_depths
seqs['gc'] = seq_gc_contents
seqs['est_gp'] = -1 # Pandas FTW!
seqs['likeliest_copynum'] = -1
seqs.set_index('ID', inplace=True)
seqs.sort_values(by=['len', 'mean_kmer_depth'], inplace=True)

# adjust BIN_MIN as # of copy-number components to be estimated and variance increases (i.e. as length decreases)
BIN_MINSIZE = min(500, numseqs)
quantile = max(BIN_MINSIZE/numseqs, 0.0025) * 100
if quantile > 0.25:
    quantile = 100 / m.floor(100 / quantile) # (Basically) the next smallest possible equally sized bins

len_percentiles_uniq = np.unique(np.percentile(seqs.len.values, np.arange(quantile, 100, quantile), interpolation='higher'))
length_gps_for_est = get_length_gps_for_est(seqs, len_percentiles_uniq, BIN_MINSIZE)
length_gps_count = len(length_gps_for_est)

#len_gp_stats = []
#len_gp_stats.append(pd.DataFrame(columns=['ID', 'len', 'mean_kmer_depth', 'modex', 'gc', 'est_gp', 'likeliest_copynum']))
gp_min_lens = []
gp_max_lens = []
gp_max_depths = []
gp_max_depth_in_est = []
gp_max_depth_pctl_rank_in_est = []
gp_copynums = []
#copynum_depth_lbs = [[[None]] * length_gps_count] * 2
copynum_depth_lbs = []
copynum_depth_maxs = []
copynum_wts = []
copynum_means = []
copynum_stdevs = []

# Fit under assumption that first peak corresponds to mode of copy-number 1 or 2 (unique homozygous) sequences
mode_error = 0.05
mode1_copynums = [1, 2]
if length_gps_count > 3:
    mode1_copynums.extend([3, 4])

for mode1_copynum in mode1_copynums:
    log_file = open(OUTPUT_DIR + '/log' + str(mode1_copynum) + '.txt', 'w', newline='')
    # TODO: Rm debug file
    debug_file = open(OUTPUT_DIR + '/debug' + str(mode1_copynum) + '.txt', 'w', newline='')
    gp_min_lens.append([None] * length_gps_count)
    gp_max_lens.append([None] * length_gps_count)
    gp_max_depths.append([None] * length_gps_count)
    gp_max_depth_in_est.append([None] * length_gps_count)
    gp_max_depth_pctl_rank_in_est.append([None] * length_gps_count)
    gp_copynums.append([None] * length_gps_count)
    copynum_depth_lbs.append([])
    copynum_depth_maxs.append([])
    copynum_wts.append([])
    copynum_means.append([])
    copynum_stdevs.append([])
    mode, mode_min, mode_max = np.nan, NONNEG_CONSTANT, np.inf
    sigma_min = NONNEG_CONSTANT
    for len_gp_idx in range(length_gps_count):
        copynum_depth_lbs[mode1_copynum - 1].append([None])
        copynum_depth_maxs[mode1_copynum - 1].append([None])
        copynum_wts[mode1_copynum - 1].append([None])
        copynum_means[mode1_copynum - 1].append([None])
        copynum_stdevs[mode1_copynum - 1].append([None])

    for len_gp_idx in range(length_gps_count - 1, -1, -1):
        debug_file.write(str(len_gp_idx) + '\n')
        print(len_gp_idx)
        # Estimate any probable error distribution, and mode excluding error sequences
        depths = np.copy(length_gps_for_est[len_gp_idx].mean_kmer_depth.values)
        depths.sort()
        kde = sm.nonparametric.KDEUnivariate(depths)
        kde.fit(bw=np.percentile(depths, 50) * 0.04) # median/25: seems like a decent heuristic
        offset = 0.1
        depth_max_pctl_rank = 100 - (stats.variation(depths) * 1.5)
        depth_max_pctl = np.percentile(depths, depth_max_pctl_rank)
        grid_max = depth_max_pctl + offset # also seems like a decent heuristic
        kde_grid_density = 20
        grid_min = depths[0] - offset
        kde_grid = np.linspace(grid_min, grid_max, val_to_grid_idx(grid_max, grid_min, kde_grid_density) + 1)
        density = kde.evaluate(kde_grid)

        gp_min_lens[mode1_copynum - 1][len_gp_idx] = length_gps_for_est[len_gp_idx].len.min()
        gp_max_lens[mode1_copynum - 1][len_gp_idx] = length_gps_for_est[len_gp_idx].len.max()
        gp_max_depths[mode1_copynum - 1][len_gp_idx] = length_gps_for_est[len_gp_idx].mean_kmer_depth.max()
        gp_max_depth_in_est[mode1_copynum - 1][len_gp_idx] = depth_max_pctl
        gp_max_depth_pctl_rank_in_est[mode1_copynum - 1][len_gp_idx] = depth_max_pctl_rank

        min_density_depth_idx_1 = 0
        # hopefully universally effective heuristic to avoid inappropriately applying to very narrow depth distributions
        # location of min., if any, between error distribution and mode not likely to be lower than this; and if it is, hopefully little harm done being off by < 1
        if depths[0] + 1 < np.percentile(depths, 10):
            soft_lb_idx = int(val_to_grid_idx(depths[0] + 1, grid_min, kde_grid_density))
            min_density_depth_idx_1 = soft_lb_idx + np.argmin(density[soft_lb_idx:int(val_to_grid_idx(np.percentile(depths, 20), grid_min, kde_grid_density))])

        # confusing variable names follow: mode (soon) represents cp#1, while mode_idx represents the index at which maximum density occurs, whether cp#1 or 2
        if np.isnan(mode):
            mode_idx = min_density_depth_idx_1 + np.argmax(density[min_density_depth_idx_1:(len(density))])
            mode = grid_idx_to_val(mode_idx, kde_grid_density, grid_min)
            mode /= float(mode1_copynum) # for consistency: let it represent c#1
        else:
            mode_idx = val_to_grid_idx(mode * mode1_copynum, grid_min, kde_grid_density)

        # condition mostly to exclude cases without perceptible error distribution, i.e. most cases, except small genomes
        if min_density_depth_idx_1 > 0 and np.mean(density[:min_density_depth_idx_1]) > density[min_density_depth_idx_1]:
            depths = depths[min_density_depth_idx_1:]

        # TODO: Remove preceding blank line
        depths = depths[depths <= depth_max_pctl]

        # Estimate standard deviation of copy-number 1 sequences
        # Under assumption that 1st peak represents copy-number 2 sequences, more accurate to estimate from c#2 sequences because density of c#1 is likely to be low at c#2 mean, but not vice versa
        sigma = np.std(get_approx_component_obs(depths, mode * mode1_copynum, mode, depths[0])) / mode1_copynum
        sigma_sqrt_2pi = m.sqrt(2 * m.pi) * sigma
        component_weights = [np.nan] * mode1_copynum
        mode_density = get_density_for_idx(mode_idx, density)
        semiweighted_density = stats.norm.pdf(mode * mode1_copynum, mode * mode1_copynum, sigma * mode1_copynum)
        if mode1_copynum == 2:
            if mode >= depths[0]:
                unweighted = stats.norm.pdf(mode * mode1_copynum, mode, sigma)
                semiweighted_density += (get_density_for_idx(val_to_grid_idx(mode, grid_min, kde_grid_density), density) * unweighted / mode_density)
            if 3 * mode <= depths[-1]:
                unweighted = stats.norm.pdf(mode * mode1_copynum, 3 * mode, 3 * sigma)
                semiweighted_density += (get_density_for_idx(val_to_grid_idx(3 * mode, grid_min, kde_grid_density), density) * unweighted / mode_density)
            else:
                if 2 * mode <= depths[-1]:
                    unweighted = stats.norm.pdf(mode, 2 * mode, 2 * sigma)
                    semiweighted_density += (get_density_for_idx(val_to_grid_idx(2 * mode, grid_min, kde_grid_density), density) * unweighted / mode_density)

        # TODO: Delete preceding blank line
        mode_component_wt = mode_density / semiweighted_density
        component_weights.append(mode_component_wt)
        del mode_idx
        smallest_copynum = 1
        if mode1_copynum == 2:
            if mode < depths[0]:
                component_weights[1] = 0
            else:
                diff = get_density_for_idx(val_to_grid_idx(mode, grid_min, kde_grid_density), density) - (mode_component_wt * stats.norm.pdf(mode, 2*mode, 2*sigma))
                component_weights[1] = max(0, diff * sigma_sqrt_2pi)
            if component_weights[1] == 0:
                smallest_copynum = 2

        i = mode1_copynum
        while (i+1) * mode < depths[-1]:
            adjacent = 2 # Another heuristic: assume density of preceding and following components equal at current mode (mean)
            if (i+2) * mode >= depths[-1]:
                adjacent = 1
            adjacent_density = adjacent * component_weights[i] * stats.norm.pdf((i+1) * mode, i * mode, i * sigma)
            density_next_mode = max(0, get_density_for_idx(val_to_grid_idx((i+1) * mode, grid_min, kde_grid_density), density) - adjacent_density)
            if density_next_mode > 0.5:
                i += 1
                curr_denominator = i * sigma_sqrt_2pi
                component_weights.append(curr_denominator * density_next_mode)
            else:
                break

        params = Parameters()
        components = [None]
        dummy = ConstantModel(prefix='dummy_')
        dummy.set_param_hint('c', value=0, vary=False)
        params.update(dummy.make_params())
        copynum_components = dummy
        weight_expr = '0'
        smallest_prefix = None
        for j in range(1, i+1):
            model = init_gaussian(j, component_weights)
            components.append(model)
            if model is not None:
                copynum_components = copynum_components + model
                weight_expr += ' + ' + model.prefix + 'amplitude'
                params.update(model.make_params())
                if j == smallest_copynum:
                    params[model.prefix + 'center'].set(value = mode * smallest_copynum, min = mode_min * smallest_copynum, max = mode_max * smallest_copynum)
                    params[model.prefix + 'sigma'].set(value = sigma * smallest_copynum, min = sigma_min * smallest_copynum)
                    smallest_prefix = model.prefix
                else:
                    params[model.prefix + 'center'].set(vary = False, expr = str(j / smallest_copynum) + ' * ' + smallest_prefix + 'center')
                    params[model.prefix + 'sigma'].set(vary = False, expr = str(j / smallest_copynum) + ' * ' + smallest_prefix + 'sigma')
                params[model.prefix + 'amplitude'].set(value = component_weights[j], min = NONNEG_CONSTANT)

        j = i + 1
        if j * mode < depths[-1]:
            curr_mode = j * mode
            tail_stats = stats.describe(depths[depths > curr_mode])
            tail_mean_density = get_density_for_idx(val_to_grid_idx(tail_stats.mean, grid_min, kde_grid_density), density)
            est_components_left = (depths[-1] - (i + 0.5) * mode) / mode
            tail_fat_enough = (est_components_left >= 100) or (np.percentile(depths, 100 - est_components_left) > (i + 0.5) * mode)
            if tail_fat_enough and (tail_mean_density >= 3 * component_weights[i] * stats.norm.pdf(tail_stats.mean, i * mode, i * sigma)):
                gamma_model = Model(gamma, prefix='gamma_')
                components.append(gamma_model)
                # rough guesses (likely overestimates) for starting parameter values
                pre = 0.5 * depths[(depths > (j-1) * mode) & (depths <= curr_mode)].size
                post = depths[depths > curr_mode].size
                pre_fraction = pre * 1.0 / (pre + post)
                loc = curr_mode - (2 * mode)
                mean_start = pre_fraction * curr_mode + (1 - pre_fraction) * tail_stats.mean
                var_start = pre_fraction * j**2 * sigma_min**2 + (1 - pre_fraction) * tail_stats.variance #iffyy...??
                shape_start = 1 + ((curr_mode - loc) * (mean_start - loc) / var_start)
                params.update(gamma_model.make_params())
                params['gamma_mode_constraint'] = Parameter(value = mean_start - m.sqrt(var_start / shape_start), min = curr_mode + NONNEG_CONSTANT)
                params['gamma_shape'].set(value = shape_start, min = 1 + NONNEG_CONSTANT)
                params['gamma_var_constraint'] = Parameter(value = var_start, min = j**2 * sigma_min**2 + NONNEG_CONSTANT)
                params['gamma_scale'].set(expr = 'sqrt(gamma_var_constraint / gamma_shape)')
                params['gamma_mean_constraint'] = Parameter(expr = 'gamma_mode_constraint + gamma_scale')
                params['gamma_loc'].set(expr = 'gamma_mean_constraint - (gamma_shape * gamma_scale)')
                gamma_weight = 'gamma_wt_'
                gamma_weight_model = ConstantModel(prefix=gamma_weight)
                gamma_weight_model.set_param_hint('c', value = max(component_weights[i], (pre + post) * 1.0 / depths.size), min=NONNEG_CONSTANT)
                params.update(gamma_weight_model.make_params())
                copynum_components = copynum_components + gamma_weight_model * gamma_model
                weight_expr += ' + ' + gamma_weight + 'c'

        genome_scale_model = ConstantModel(prefix='genomescale_')
        params.update(genome_scale_model.make_params(c=depths.size))
        mixture_model = genome_scale_model * copynum_components

        gp_copynums[mode1_copynum - 1][len_gp_idx] = len(components) - components.count(None)
        if gp_copynums[mode1_copynum - 1][len_gp_idx] == 1:
            params[components[smallest_copynum].prefix + 'amplitude'].set(value = 1.0, vary=False)
        else: # TODO: Try to specify better
            params.add('weight', value=1.0, vary=False, expr=weight_expr)

        step = (depths[-1] - depths[0]) / m.floor(depths.size * 1.0 / 100) # heuristic n...
        lb_pts = np.arange(m.floor(depths[0]), m.ceil(depths[0]), step)
        if lb_pts.size > 0:
            diffs = depths[0] - lb_pts
            lb = lb_pts[diffs >= 0][diffs[diffs >= 0].argmin()]
        else:
            lb = depths[0]

        # TODO: Rm preceding blank line
        ub = lb + step * m.ceil((depths[-1] - lb) / step)

        lmfit_range = np.arange(lb, ub, step)
        result = mixture_model.fit(np.histogram(depths, lmfit_range)[0], params, x=lmfit_range[1:])

        # Set mode the first time, i.e. from estimation and classification of longest sequences
        if mode_max == np.inf:
            mode = result.params[components[smallest_copynum].prefix + 'center'].value / smallest_copynum
            mode_min = (1 - mode_error) * mode
            mode_max = (1 + mode_error) * mode

        # TODO: Delete blank line
        sigma_min = result.params[components[smallest_copynum].prefix + 'sigma'].value / smallest_copynum

        if smallest_copynum > 1:
            copynum_depth_lbs[mode1_copynum - 1][len_gp_idx].append(None)
            copynum_depth_maxs[mode1_copynum - 1][len_gp_idx].append(None)
            copynum_wts[mode1_copynum - 1][len_gp_idx].append(None)
            copynum_means[mode1_copynum - 1][len_gp_idx].append(None)
            copynum_stdevs[mode1_copynum - 1][len_gp_idx].append(None)

        gp_len_condition = (seqs.len >= gp_min_lens[mode1_copynum - 1][len_gp_idx]) & (seqs.len <= gp_max_lens[mode1_copynum - 1][len_gp_idx])
        seqs.loc[gp_len_condition, 'modex'] = seqs[gp_len_condition].mean_kmer_depth / mode
        seqs.loc[gp_len_condition, 'est_gp'] = len_gp_idx + 1
        lb, ub = 0, np.inf
        ubs = ['N/A'] * (smallest_copynum - 1)
        mean_next = None
        for idx in range(smallest_copynum, len(components) - 1):
            prefix_i = components[idx].prefix
            prefix_next = components[idx+1].prefix
            wt_i = result.params[prefix_i + 'amplitude'].value
            mean_i = result.params[prefix_i + 'center'].value
            sigma_i = result.params[prefix_i + 'sigma'].value
            #x = Symbol('x', real=True)
            #x = Symbol('x')
            if re.match('gauss', prefix_next):
                print('Next Gauss')
                wt_next = result.params[prefix_next + 'amplitude'].value
                mean_next = result.params[prefix_next + 'center'].value
                sigma_next = result.params[prefix_next + 'sigma'].value
                #ub = nsolve((wt_i * (i + 1) / (wt_next * i)) - (m.e ** (-0.5 * (((mean_next - x) / sigma_next)**2 - ((x - mean_i) / sigma_i)**2))), (idx + 0.5) * result.params['gauss1_center'].value)
                def densities_diff(x):
                    return (wt_i * (i + 1) / (wt_next * i)) - (m.e ** (-0.5 * (((mean_next - x) / sigma_next)**2 - ((x - mean_i) / sigma_i)**2)))
                roots = optimize.root(densities_diff, (idx + 0.5) * result.params[prefix_i + 'center'].value)
                debug_file.write('start: ' + str((idx + 0.5) * result.params[prefix_i + 'center'].value) + ', roots: ' + str(roots) + '\n')
                #print('start: ' + str((idx + 0.5) * result.params['gauss1_center'].value))
                ub = roots.x[0]
            else:
                print('Next Gamma')
                wt_next = result.params['gamma_wt_c'].value
                shape = result.params[prefix_next + 'shape'].value
                loc = result.params[prefix_next + 'loc'].value
                scale = result.params[prefix_next + 'scale'].value
                mean_next = shape * scale + loc
                sigma_next = m.sqrt(shape * scale**2)
                print(str(loc) + ', ' + str(shape) + ', ' + str(scale))
                print(str(mean_next) + ', ' + str(sigma_next))
                #print(str(gamma_fn(shape)))
                #print()
                #density_i = wt_i * (m.e ** (-0.5 * ((x - mean_i) / sigma_i)**2)) / (m.sqrt(2 * m.pi) * sigma_i)
                #density_next = ((x - loc)**(shape - 1)) * (m.e ** ((loc - x) / scale)) / (gamma_fn(shape) * scale**shape)
                #ub = nsolve(density_i - density_next, 0.5 * (mean_next + mean_i))
                def densities_diff(x):
                    density_i = wt_i * (m.e ** (-0.5 * ((x - mean_i) / sigma_i)**2)) / (m.sqrt(2 * m.pi) * sigma_i)
                    density_next = ((x - loc)**(shape - 1)) * (m.e ** ((loc - x) / scale)) / (gamma_fn(shape) * scale**shape)
                    return density_i - density_next
                roots = optimize.root(densities_diff, 0.5 * (mean_next - scale + mean_i)) # mean of modes (gamma mode = mean - scale)
                debug_file.write('start: ' + str(0.5 * (mean_next - scale + mean_i)) + 'roots: ' + str(roots) + '\n')
                print('start: ' + str(0.5 * (mean_next - scale + mean_i)))
                ub = roots.x[0]
            debug_file.write('ub: ' + str(ub) + '\n')
            #print('ub: ' + str(ub))
            seqs.loc[gp_len_condition & (seqs.mean_kmer_depth > lb) & (seqs.mean_kmer_depth <= ub), 'likeliest_copynum'] = idx
            copynum_depth_lbs[mode1_copynum - 1][len_gp_idx].append(lb)
            copynum_depth_maxs[mode1_copynum - 1][len_gp_idx].append(ub)
            copynum_wts[mode1_copynum - 1][len_gp_idx].append(wt_i)
            copynum_means[mode1_copynum - 1][len_gp_idx].append(mean_i)
            copynum_stdevs[mode1_copynum - 1][len_gp_idx].append(sigma_i)
            ubs.append(ub)
            lb = ub

        seqs.loc[gp_len_condition & (seqs.mean_kmer_depth > lb), 'likeliest_copynum'] = len(components) - 1

        # TODO: Delete blank line after conditional
        debug_file.write('')
        print('')
        if mean_next is None:
            wt_next = result.params[components[smallest_copynum].prefix + 'amplitude'].value
            mean_next = result.params[components[smallest_copynum].prefix + 'center'].value
            sigma_next = result.params[components[smallest_copynum].prefix + 'sigma'].value

        ub = np.inf
        copynum_depth_lbs[mode1_copynum - 1][len_gp_idx].append(lb)
        copynum_depth_maxs[mode1_copynum - 1][len_gp_idx].append(ub)
        copynum_wts[mode1_copynum - 1][len_gp_idx].append(wt_next)
        copynum_means[mode1_copynum - 1][len_gp_idx].append(mean_next)
        copynum_stdevs[mode1_copynum - 1][len_gp_idx].append(sigma_next)

        log_file.write(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H:%M:%S : Sequence group ') + str(len_gp_idx + 1) + ' out of ' + str(length_gps_count) + ' estimated\n')
        log_file.write('Group minimum and maximum lengths: ' + str(gp_min_lens[mode1_copynum - 1][len_gp_idx]) + ', ' + str(gp_max_lens[mode1_copynum - 1][len_gp_idx]) + '\n')
        log_file.write('Maximum mean k-mer depth of all sequences in group: ' + str(gp_max_depths[mode1_copynum - 1][len_gp_idx]) + '. ')
        log_file.write('Maximum used in estimation: ' + str(depth_max_pctl) + ' (' + str(depth_max_pctl_rank) + ' percentile).\n')
        log_file.write(result.fit_report())
        log_file.write('\n')

    log_file.close()
    debug_file.close()

    seq_label_filename = OUTPUT_DIR + '/sequence-labels-' + str(mode1_copynum) + '.csv'
    seqs.loc[:, 'len':].to_csv(seq_label_filename, header=['Length', 'Average k-mer depth', '1st Mode X', 'GC %', 'Estimation length group', 'Likeliest copy #'], index_label='ID')

# TODO: Store these data in Pandas arrays DUUUHHHHHHHH
for mode1_copynum in [1, 2]:
    with open(OUTPUT_DIR + '/length_gp_stats_' + str(mode1_copynum) + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        header = ['Group #', 'Min. len.', 'Max. len.', 'Max. depth', 'Max. depth in estimation', '# components']
        writer.writerow(header)
        for len_gp_idx in range(length_gps_count):
            row = [len_gp_idx + 1, gp_min_lens[mode1_copynum - 1][len_gp_idx], gp_max_lens[mode1_copynum - 1][len_gp_idx], gp_max_depths[mode1_copynum - 1][len_gp_idx]]
            row.extend([str(gp_max_depth_in_est[mode1_copynum - 1][len_gp_idx]) + ' (' + str(gp_max_depth_pctl_rank_in_est[mode1_copynum - 1][len_gp_idx]) + ' percentile)'])
            row.append(gp_copynums[mode1_copynum - 1][len_gp_idx])
            writer.writerow(row)

for mode1_copynum in [1, 2]:
    with open(OUTPUT_DIR + '/copynumber_params_' + str(mode1_copynum) + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        header = ['Group #', 'Group min. len.', 'Group max. len.', 'Component #', 'Component depth lower bound', 'Component max. depth', 'Weight', 'Mean', 'Std. deviation']
        writer.writerow(header)
        for len_gp_idx in range(length_gps_count):
            for i in range(1, len(copynum_wts[mode1_copynum - 1][len_gp_idx])):
                row = [len_gp_idx + 1, gp_min_lens[mode1_copynum - 1][len_gp_idx], gp_max_lens[mode1_copynum - 1][len_gp_idx], i]
                row.extend([copynum_depth_lbs[mode1_copynum - 1][len_gp_idx][i], copynum_depth_maxs[mode1_copynum - 1][len_gp_idx][i]])
                row.extend([copynum_wts[mode1_copynum - 1][len_gp_idx][i], copynum_means[mode1_copynum - 1][len_gp_idx][i], copynum_stdevs[mode1_copynum - 1][len_gp_idx][i]])
                writer.writerow(row)


#expected_depth_idx = 1 + (expected_kmer_depth - depths[0] + offset) * kde_grid_density
## check pos. of global max relative to min and expected depths
#if (mode_idx - soft_min_idx) < abs(expected_depth_idx - mode_idx):
#
## Exclude any probable sequences from an error distribution
#density_diffs_len = 2 * kde_grid_density
#density_diffs = density[1:(density_diffs_len + 1)] - density[:density_diffs_len]
#if len(density_diffs[density_diffs >= 0]) > (offset_for_inverse * density_diffs_len):
#    mode = grid_idx_to_val(np.argmax(density), kde_grid_density, offset_for_inverse)
#    component_obs = curr_gp[curr_gp.mean_kmer_depth < 1.5 * mode]
#    component_obs_count = len(curr_gp[curr_gp.mean_kmer_depth < 1.5 * mode])
#else:
#    density_argmax = np.argmax(density[density_diffs_len:])
#    mode = grid_idx_to_val(density_argmax, kde_grid_density, offset_for_inverse)
#    min_depth_for_classification = grid_idx_to_val(np.argmin(density[density_diffs_len:density_argmax]), kde_grid_density, offset_for_inverse)
#    component_obs = curr_gp[(curr_gp.mean_kmer_depth >= min_depth_for_classification) and (curr_gp.mean_kmer_depth < 1.5 * mode)]
#    component_obs_count = len(curr_gp[(curr_gp.mean_kmer_depth >= min_depth_for_classification) and (curr_gp.mean_kmer_depth < 1.5 * mode)])
#
#components_to_estimate = 1
#current_std = component_obs.mean_kmer_depth.describe()['std']
#current_gaussian = stats.norm(mode, current_std)
#next_mode = (components_to_estimate + 1) * mode
#next_gaussian = stats.norm(next_mode, current_std * (components_to_estimate + 1)**2)
## Likely overestimates where to stop, because this assumes proportional scaling whereas observed distributions are likely to shrink faster
#while current_gaussian.pdf(next_mode) < 0.25 * next_gaussian.pdf(next_mode):
#    components_to_estimate += 1
#
#components_with_obs_gte_lb = 1
#while component_obs_count >= (int(BIN_MINSIZE < 5000) * 50) + (int(BIN_MINSIZE >= 5000) * 0.01 * BIN_MINSIZE):
#    components_with_obs_gte_lb += 1
#
## "Homozygous mode"
#components_to_estimate = min(components_to_estimate, components_with_obs_gte_lb)
#
## DELETE depth lists?
#est_gp_min_lengths = []
#est_gp_ub_lengths = [np.Inf]
#est_gp_min_depths = []
#est_gp_max_depths = []
#est_gp_features = pd.DataFrame(columns=['min_length', 'ub_length', 'min_depth', 'max_depth'])
#
## Group sequences by length
#percentiles_uniq = np.unique(np.percentile(seqs['kmers'], np.arange(0, 100, quantile), axis=0, interpolation='lower'))
#sup = np.inf
#len_gps = []
#for i in range(len(percentiles_uniq) - 1, -1, -1):
#    current_len_gp = seqs[np.where((seqs['kmers'] >= percentiles_uniq[i]) & (seqs['kmers'] < sup))[0]]
#    if len(current_len_gp) >= BIN_MINSIZE:
#        len_gps.append(current_len_gp)
#        sup = percentiles_uniq[i]
#len_gps.reverse()
## Ensure 1st group is large enough ... Not that it matters with the aggregation into orders of magnitude for actual use in estimation
#if len(len_gps[0]) < BIN_MINSIZE and len(len_gps) > 1:
#    len_gps[1] = np.concatenate((len_gps[0], len_gps[1]))
#    len_gps.pop(0)
