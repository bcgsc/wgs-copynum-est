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

def add_to_copynum_stats(data, cols, stats_hash):
    for i in range(len(data)):
        stats_hash[cols[i]].append(data[i])

def variance_from_curve(len_group_mode, mode, longest_seqs_mode1_copynum, mode_error):
    ratio = len_group_mode / mode
    ret = (ratio >= (1 - mode_error) and ratio <= (1 + mode_error))
    if longest_seqs_mode1_copynum == 2:
        double_mode_error = 2 * mode_error
        return (ret or (ratio >= (2 - double_mode_error) and ratio <= (2 + double_mode_error)))
    return ret

# Fit linearised exponential decay: sigma = A * exp(K * length_median) => log(sigma) = K * length_median + log(A)
def guess_next_sigma(length_median, length_medians, sigmas):
    K, log_A = np.polyfit(length_medians, np.log(sigmas), 1)
    return m.exp(K * length_median + log_A)


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
length_gp_medians = list(map(lambda gp: gp.len.median(), length_gps_for_est))
length_gp_sigmas = [None, [None] * length_gps_count, [None] * length_gps_count]

LEN_GP_STATS_COLS = ['min_len', 'max_len', 'max_depth', 'max_depth_in_est', 'max_depth_pctl_rank_in_est', 'min_copynum', 'max_copynum_est']
len_gp_stats = [None]
COPYNUM_STATS_COLS = ['len_gp_id', 'len_gp_min_len', 'len_gp_max_len', 'copynum', 'depth_lb', 'depth_max', 'weight', 'depth_mean', 'depth_stdev']
copynum_stats = [None]

# Fit under assumption that first peak corresponds to mode of copy-number 1 or 2 (unique homozygous) sequences
mode_error = 0.05

for longest_seqs_mode1_copynum in [1, 2]:
    log_file = open(OUTPUT_DIR + '/log' + str(longest_seqs_mode1_copynum) + '.txt', 'w', newline='')
    # TODO: Rm debug file
    debug_file = open(OUTPUT_DIR + '/debug' + str(longest_seqs_mode1_copynum) + '.txt', 'w', newline='')
    len_gp_stats.append(pd.DataFrame(data=None, index=pd.RangeIndex(stop=length_gps_count), columns=LEN_GP_STATS_COLS))
    copynum_stats_hash = { col: [] for col in COPYNUM_STATS_COLS }

    mode, mode_min, mode_max = np.nan, NONNEG_CONSTANT, np.inf
    sigma_min = NONNEG_CONSTANT
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

        len_gp_stats[longest_seqs_mode1_copynum].loc[len_gp_idx, 'min_len'] = length_gps_for_est[len_gp_idx].len.min()
        len_gp_stats[longest_seqs_mode1_copynum].loc[len_gp_idx, 'max_len'] = length_gps_for_est[len_gp_idx].len.max()
        len_gp_stats[longest_seqs_mode1_copynum].loc[len_gp_idx, 'max_depth'] = length_gps_for_est[len_gp_idx].mean_kmer_depth.max()
        len_gp_stats[longest_seqs_mode1_copynum].loc[len_gp_idx, 'max_depth_in_est'] = depth_max_pctl
        len_gp_stats[longest_seqs_mode1_copynum].loc[len_gp_idx, 'max_depth_pctl_rank_in_est'] = depth_max_pctl_rank
        curr_len_gp_stats = len_gp_stats[longest_seqs_mode1_copynum].loc[len_gp_idx]

        min_density_depth_idx_1 = 0
        # hopefully universally effective heuristic to avoid inappropriately applying to very narrow depth distributions
        # location of min., if any, between error distribution and mode not likely to be lower than this; and if it is, hopefully little harm done being off by < 1
        if depths[0] + 1 < np.percentile(depths, 10):
            soft_lb_idx = m.ceil(val_to_grid_idx(depths[0] + 1, grid_min, kde_grid_density))
            min_density_depth_idx_1 = soft_lb_idx + np.argmin(density[soft_lb_idx:m.floor(val_to_grid_idx(np.percentile(depths, 20), grid_min, kde_grid_density))])

        # condition mostly to exclude cases without perceptible error distribution, i.e. most cases, except small genomes
        if min_density_depth_idx_1 > 0 and np.mean(density[:min_density_depth_idx_1]) > density[min_density_depth_idx_1]:
            depths = depths[min_density_depth_idx_1:]

        # TODO: Remove preceding blank line
        depths = depths[depths <= depth_max_pctl]

        mode_idx = min_density_depth_idx_1 + np.argmax(density[min_density_depth_idx_1:])
        len_group_mode = grid_idx_to_val(mode_idx, kde_grid_density, grid_min)
        if np.isnan(mode):
            mode = len_group_mode
            mode /= float(longest_seqs_mode1_copynum) # for consistency: let it represent c#1

        # Estimate standard deviation of copy-number 1 sequences
        if variance_from_curve(len_group_mode, mode, longest_seqs_mode1_copynum, mode_error):
            # If 1st peak represents c#2 sequences, likely more accurate to estimate from c#2 sequences because
            # density of c#1 is likely to be lower at c#2 mean than density of c#2 at c#1 mean
            mode1_copynum = round(len_group_mode / mode)
            sigma = np.std(get_approx_component_obs(depths, len_group_mode, mode, depths[0])) / mode1_copynum
        else:
            sigma = guess_next_sigma(length_gp_medians[len_gp_idx], length_gp_medians[(len_gp_idx + 1):], length_gp_sigmas[longest_seqs_mode1_copynum][(len_gp_idx + 1):])

        # TODO: rm blank line
        if sigma < sigma_min:
            sigma_min = NONNEG_CONSTANT

        # TODO: rm blank line
        sigma_sqrt_2pi = m.sqrt(2 * m.pi) * sigma
        sigma_sqrt_2pi_reciprocal = 1 / sigma_sqrt_2pi

        # Estimate copy-number component weights starting at means of cp#s 1 & 2, using heuristic of 2x cp#1 density at cp#2 mean
        component_weights = [np.nan] * (2 + int(depths[-1] > mode * 2))
        if len(component_weights) < 3:
            component_weights[1] = get_density_for_idx(val_to_grid_idx(mode, grid_min, kde_grid_density), density) * sigma_sqrt_2pi
        elif mode <= depths[0]:
            component_weights[1] = 0
            # likely slight overestimate if length group has copy-number 3 sequences
            component_weights[2] = get_density_for_idx(val_to_grid_idx(2 * mode, grid_min, kde_grid_density), density) * 2 * sigma_sqrt_2pi
        else:
            cpnum_density_2at1 = 0.5 * sigma_sqrt_2pi_reciprocal * stats.norm.pdf(mode, 2*mode, 2*sigma)
            cpnum_density_1at2 = sigma_sqrt_2pi_reciprocal * stats.norm.pdf(2*mode, mode, sigma)
            cpnum_densities_j_at_i = np.array([[sigma_sqrt_2pi_reciprocal, cpnum_density_2at1], [2 * cpnum_density_1at2, 0.5 * sigma_sqrt_2pi_reciprocal]])
            density_obs_at_mean1 = get_density_for_idx(val_to_grid_idx(mode, grid_min, kde_grid_density), density)
            density_obs_at_mean2 = get_density_for_idx(val_to_grid_idx(mode*2, grid_min, kde_grid_density), density)
            component_weights[1], component_weights[2] = np.linalg.solve(cpnum_densities_j_at_i, np.array([density_obs_at_mean1, density_obs_at_mean2]))

        smallest_copynum = 1
        if component_weights[1] == 0:
            if component_weights[2] == 0:
                error_msg = 'Lowest copy number for sequences of length ' + curr_len_gp_stats.min_len + ' to ' + curr_len_gp_stats.max_len + ' in dataset higher than 1: '
                error_msg += 'none are single-copy (either homo- or heterozygous)!'
                raise RuntimeError(error_msg)
            smallest_copynum = 2

        # Estimate weights for remaining, higher copy numbers
        hetero_to_homozg_maxdens = 1
        if len(component_weights) > 2:
            if component_weights[1] == 0:
                hetero_to_homozg_maxdens += 1
            else:
                start = m.ceil(val_to_grid_idx(max(depths[0], (1 - mode_error) * mode), grid_min, kde_grid_density))
                end = m.ceil(val_to_grid_idx(min(depths[-1], (1 + mode_error) * 2 * mode), grid_min, kde_grid_density))
                maxdens = grid_idx_to_val(start + np.argmax(density[start:end]), kde_grid_density, grid_min)
                ratio = maxdens / mode
                if (ratio >= 2 * (1 - mode_error)) and (ratio <= 2 * (1 + mode_error)):
                    hetero_to_homozg_maxdens += 1
                elif (ratio > 1 + mode_error) and (ratio < 2 * (1 - mode_error)):
                    hetero_to_homozg_maxdens = ratio

        i = 2
        while (i + 1) * mode < depths[-1]:
            adjacent = 1
            if (i + 2) * mode < depths[-1]:
                adjacent = 2 # Another heuristic: assume density of preceding and following components equal at current mode (mean)
            adjacent_density = adjacent * component_weights[i] * stats.norm.pdf((i + 1) * mode, i * mode, i * sigma)
            density_next_mode = max(0, get_density_for_idx(val_to_grid_idx((i + 1) * mode, grid_min, kde_grid_density), density) - adjacent_density)
            if density_next_mode == 0:
                break
            component_weights.append((i + 1) * sigma_sqrt_2pi * density_next_mode)
            next_eval_pt = (i + 1 + (int(adjacent == 2) * (hetero_to_homozg_maxdens - 1))) * mode
            if adjacent == 1:
                i += int(density_next_mode > 0.5)
                break
            else:
                next_adjacent_density = (1 + int((i+3) * mode < depths[-1])) * component_weights[i+1] * stats.norm.pdf((i+2) * mode, (i+1) * mode, (i+1) * sigma)
                density_following_mode = max(0, get_density_for_idx(val_to_grid_idx((i + 2) * mode, grid_min, kde_grid_density), density) - next_adjacent_density)
                weight_following = (i + 2) * sigma_sqrt_2pi * density_following_mode
                cpnum_dens = component_weights[i-1] * sigma_sqrt_2pi_reciprocal * stats.norm.pdf(next_eval_pt, (i - 1) * mode, (i - 1) * sigma) / (i - 1)
                cpnum_dens += component_weights[i] * sigma_sqrt_2pi_reciprocal * stats.norm.pdf(next_eval_pt, i * mode, i * sigma) / i
                next_obs_dens = get_density_for_idx(val_to_grid_idx(next_eval_pt, grid_min, kde_grid_density), density)
                if next_obs_dens - (2 * cpnum_dens) >= (5.0 / 12) * next_obs_dens:
                    component_weights.append(weight_following)
                    i += 2
                else:
                    break

        # TODO: rm blank line
        max_gaussian_copynums = max(1, int(depths[-1] > mode * 2) * i)

        params = Parameters()
        components = [None]
        dummy = ConstantModel(prefix='dummy_')
        dummy.set_param_hint('c', value=0, vary=False)
        params.update(dummy.make_params())
        copynum_components = dummy
        smallest_prefix = None
        for j in range(1, max_gaussian_copynums + 1):
            model = init_gaussian(j, component_weights)
            components.append(model)
            if model is not None:
                copynum_components = copynum_components + model
                params.update(model.make_params())
                if j == smallest_copynum:
                    params[model.prefix + 'center'].set(value = mode * smallest_copynum, min = mode_min * smallest_copynum, max = mode_max * smallest_copynum)
                    params[model.prefix + 'sigma'].set(value = sigma * smallest_copynum, min = sigma_min * smallest_copynum)
                    smallest_prefix = model.prefix
                else:
                    params[model.prefix + 'center'].set(vary = False, expr = str(j / smallest_copynum) + ' * ' + smallest_prefix + 'center')
                    params[model.prefix + 'sigma'].set(vary = False, expr = str(j / smallest_copynum) + ' * ' + smallest_prefix + 'sigma')
                params[model.prefix + 'amplitude'].set(value = component_weights[j], min = NONNEG_CONSTANT, max = 1 - NONNEG_CONSTANT)

        j = max_gaussian_copynums + 1
        if j * mode < depths[-1]:
            curr_mode = j * mode
            tail_stats = stats.describe(depths[depths > curr_mode])
            tail_mean_density = get_density_for_idx(val_to_grid_idx(tail_stats.mean, grid_min, kde_grid_density), density)
            est_components_left = max(0, (depths[-1] - (max_gaussian_copynums + 0.5) * mode) / mode)
            tail_fat_enough = (est_components_left >= 100) or ((est_components_left >= 4) and np.percentile(depths, 100 - est_components_left) > (max_gaussian_copynums + 0.5) * mode)
            if tail_fat_enough and (tail_mean_density >= 3 * component_weights[i] * stats.norm.pdf(tail_stats.mean, max_gaussian_copynums * mode, max_gaussian_copynums * sigma)):
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
                gamma_weight_model.set_param_hint('c', value = max(component_weights[max_gaussian_copynums], (pre + post) * 1.0 / depths.size), min=NONNEG_CONSTANT, max = 1 - NONNEG_CONSTANT)
                params.update(gamma_weight_model.make_params())
                copynum_components = copynum_components + gamma_weight_model * gamma_model

        genome_scale_model = ConstantModel(prefix='genomescale_')
        params.update(genome_scale_model.make_params(c=depths.size))
        mixture_model = genome_scale_model * copynum_components

        len_gp_stats[longest_seqs_mode1_copynum].loc[len_gp_idx, 'min_copynum'] = smallest_copynum
        len_gp_stats[longest_seqs_mode1_copynum].loc[len_gp_idx, 'max_copynum_est'] = len(components) - 1
        if len(components) - 1 == smallest_copynum:
            params[components[smallest_copynum].prefix + 'amplitude'].set(value = 1.0, vary=False)
        else:
            wt_expr = '1 - ' + ' - '.join(map(lambda c: c.prefix + 'amplitude', components[-2:(smallest_copynum - 1):-1]))
            if len(components) - max_gaussian_copynums == 1:
                params[components[-1].prefix + 'amplitude'].set(expr = wt_expr, min = 0, max = 1)
            else:
                params['gamma_wt_c'].set(expr = wt_expr, min = 0, max = 1)

        # Finally estimate copy numbers using lmfit
        step = (depths[-1] - depths[0]) / m.floor(depths.size * 1.0 / 100) # heuristic n...
        lb_pts = np.arange(m.floor(depths[0]), m.ceil(depths[0]), step)
        if lb_pts.size > 0:
            diffs = depths[0] - lb_pts
            lb = lb_pts[diffs >= 0][diffs[diffs >= 0].argmin()]
        else:
            lb = depths[0]

        # TODO: rm blank line
        ub = lb + step * (1 + m.ceil((depths[-1] - lb) / step))

        lmfit_range = np.arange(lb, ub, step)
        result = mixture_model.fit(np.histogram(depths, lmfit_range)[0], params, x=lmfit_range[1:])

        # Set mode the first time, i.e. from estimation and classification of longest sequences
        if mode_max == np.inf:
            mode = result.params[components[smallest_copynum].prefix + 'center'].value / smallest_copynum
            mode_min = (1 - mode_error) * mode
            mode_max = (1 + mode_error) * mode
        sigma_min = (result.params[components[smallest_copynum].prefix + 'sigma'].value - result.params[components[smallest_copynum].prefix + 'sigma'].stderr) / smallest_copynum
        length_gp_sigmas[longest_seqs_mode1_copynum][len_gp_idx] = result.params[components[smallest_copynum].prefix + 'sigma'].value

        # Compute likeliest copy number assignments
        gp_len_condition = (seqs.len >= curr_len_gp_stats.min_len) & (seqs.len <= curr_len_gp_stats.max_len)
        seqs.loc[gp_len_condition, 'modex'] = seqs[gp_len_condition].mean_kmer_depth / mode
        seqs.loc[gp_len_condition, 'est_gp'] = len_gp_idx + 1
        lb, ub = 0, np.inf
        wt_iplus1, mean_iplus1, sigma_iplus1 = 0, 0, 0
        if components[1] is not None:
            wt_iplus1 = result.params[components[1].prefix + 'amplitude']
            mean_iplus1, sigma_iplus1 = result.params[components[1].prefix + 'center'], result.params[components[1].prefix + 'sigma']
        if len(components) > 2:
            wt_iplus2 = result.params[components[2].prefix + 'amplitude']
            mean_iplus2, sigma_iplus2 = result.params[components[2].prefix + 'center'], result.params[components[2].prefix + 'sigma']
        for idx in range(2, len(components) - 1, 2):
            wt_prev = 0
            if components[idx-1] is not None:
                prefix_prev = components[idx-1].prefix
                wt_prev = result.params[prefix_prev + 'amplitude'].value
                mean_prev, sigma_prev = result.params[prefix_prev + 'center'].value, result.params[prefix_prev + 'sigma'].value
            prefix_i, prefix_iplus1 = components[idx].prefix, components[idx+1].prefix
            wt_i = result.params[prefix_i + 'amplitude'].value
            mean_i, sigma_i = result.params[prefix_i + 'center'].value, result.params[prefix_i + 'sigma'].value
            if re.match('gauss', prefix_iplus1):
                wt_iplus1 = result.params[prefix_iplus1 + 'amplitude'].value
                mean_iplus1, sigma_iplus1 = result.params[prefix_iplus1 + 'center'].value, result.params[prefix_iplus1 + 'sigma'].value
            else:
                wt_iplus1 = result.params['gamma_wt_c'].value
                mean_iplus1, sigma_iplus1 = result.params['gamma_mean_constraint'].value, m.sqrt(result.params['gamma_var_constraint'].value)
            if len(components) - idx > 2:
                prefix_iplus2 = components[idx+2].prefix
                wt_iplus2 = result.params[prefix_iplus2 + 'amplitude'].value
                mean_iplus2, sigma_iplus2 = result.params[prefix_iplus2 + 'center'].value, result.params[prefix_iplus2 + 'sigma'].value
            def densities_diff(x):
                density_curr = wt_prev * stats.norm.pdf(x, mean_prev, sigma_prev) + wt_i * stats.norm.pdf(x, mean_i, sigma_i)
                if len(components) - idx > 2:
                    return wt_iplus1 * stats.norm.pdf(x, mean_iplus1, sigma_iplus1) + wt_iplus2 * stats.norm.pdf(x, mean_iplus2, sigma_iplus2) - density_curr
                return wt_iplus1 * stats.norm.pdf(x, mean_iplus1, sigma_iplus1) - density_curr
            roots = optimize.root(densities_diff, (idx + 1) * result.params[components[smallest_copynum].prefix + 'center'].value)
            #debug_file.write('start: ' + str((idx + 0.5) * result.params[prefix_i + 'center'].value) + ', roots: ' + str(roots) + '\n')
            #print('start: ' + str((idx + 0.5) * result.params['gauss1_center'].value))
            ub = roots.x[0]
            #debug_file.write('ub: ' + str(ub) + '\n')
            #print('ub: ' + str(ub))
            haploid_copynum = idx / 2
            seqs.loc[gp_len_condition & (seqs.mean_kmer_depth > lb) & (seqs.mean_kmer_depth <= ub), 'likeliest_copynum'] = haploid_copynum
            copynum_stats_data = [len_gp_idx, curr_len_gp_stats.min_len, curr_len_gp_stats.max_len, haploid_copynum]
            wt_haploid_copynum = wt_prev + wt_i
            wt_prev_normed, wt_i_normed = wt_prev/wt_haploid_copynum, wt_i/wt_haploid_copynum
            mean_haploid_copynum = (wt_prev_normed * mean_prev) + (wt_i_normed * mean_i)
            sigma_haploid_copynum = m.sqrt((wt_prev_normed * sigma_prev**2) + (wt_i_normed * sigma_i**2) + (wt_prev_normed * wt_i_normed * (mean_prev - mean_i)**2))
            copynum_stats_data.extend([lb, ub, wt_haploid_copynum, mean_haploid_copynum, sigma_haploid_copynum])
            add_to_copynum_stats(copynum_stats_data, COPYNUM_STATS_COLS, copynum_stats_hash)
            lb = ub
        ub = np.inf
        haploid_copynum = m.floor(len(components) / 2)
        seqs.loc[gp_len_condition & (seqs.mean_kmer_depth > lb), 'likeliest_copynum'] = haploid_copynum
        copynum_stats_data = [len_gp_idx, curr_len_gp_stats.min_len, curr_len_gp_stats.max_len, haploid_copynum]
        if len(components) % 2 == 0:
            wt_last, mean_last, sigma_last = wt_iplus1, mean_iplus1, sigma_iplus1
        else:
            wt_last = wt_iplus1 + wt_iplus2
            wt_iplus1_normed, wt_iplus2_normed = wt_iplus1/wt_last, wt_iplus2/wt_last
            mean_last = (wt_iplus1_normed * mean_iplus1) + (wt_iplus2_normed * mean_iplus2)
            sigma_last = m.sqrt((wt_iplus1_normed * sigma_iplus1**2) + (wt_iplus2_normed * sigma_iplus2**2) + (wt_iplus1_normed * wt_iplus2_normed * (mean_iplus2 - mean_iplus1)**2))
        copynum_stats_data.extend([lb, ub, wt_last, mean_last, sigma_last])
        add_to_copynum_stats(copynum_stats_data, COPYNUM_STATS_COLS, copynum_stats_hash)

    debug_file.write('')
    print('')

    log_file.write(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H:%M:%S : Sequence group ') + str(len_gp_idx) + ' estimated\n')
    log_file.write('Group minimum and maximum lengths: ' + str(curr_len_gp_stats.min_len) + ', ' + str(curr_len_gp_stats.max_len) + '\n')
    log_file.write('Maximum mean k-mer depth of all sequences in group: ' + str(curr_len_gp_stats.max_depth) + '. ')
    log_file.write('Maximum used in estimation: ' + str(depth_max_pctl) + ' (' + str(depth_max_pctl_rank) + ' percentile).\n')
    log_file.write(result.fit_report())
    log_file.write('\n')

    log_file.close()
    debug_file.close()
    copynum_stats.append(pd.DataFrame.from_dict(copynum_stats_hash))
    seq_label_filename = OUTPUT_DIR + '/sequence-labels-' + str(longest_seqs_mode1_copynum) + '.csv'
    seqs.loc[:, 'len':].to_csv(seq_label_filename, header=['Length', 'Average k-mer depth', '1st Mode X', 'GC %', 'Estimation length group', 'Likeliest copy #'], index_label='ID')

LEN_GP_STATS_OUTPUT_COLS = tuple(['min_len', 'max_len', 'max_depth', 'max_depth_in_est', 'min_copynum', 'max_copynum_est'])
LEN_GP_STATS_OUTPUT_HEADER = ['Min. len.', 'Max. len.', 'Max. depth', 'Max. depth in estimation', 'Smallest diploid copy # present', 'Largest diploid copy # estimated']
for longest_seqs_mode1_copynum in [1, 2]:
    filename = OUTPUT_DIR + '/length_gp_stats_' + str(longest_seqs_mode1_copynum) + '.csv'
    len_gp_stats[longest_seqs_mode1_copynum].to_csv(filename, columns=LEN_GP_STATS_OUTPUT_COLS, header=LEN_GP_STATS_OUTPUT_HEADER, index_label='ID')

COPYNUM_STATS_OUTPUT_HEADER = ['Group #', 'Group min. len.', 'Group max. len.', 'Component #', 'Component depth lower bound', 'Component max. depth', 'Weight', 'Mean', 'Std. deviation']
for longest_seqs_mode1_copynum in [1, 2]:
    filename = OUTPUT_DIR + '/copynumber_params_' + str(longest_seqs_mode1_copynum) + '.csv'
    copynum_stats[longest_seqs_mode1_copynum].to_csv(filename, header=COPYNUM_STATS_OUTPUT_HEADER, index=False)


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
