import array
import argparse
import csv
import datetime
from lmfit import Model, Parameter, Parameters
from lmfit.models import ConstantModel, GaussianModel
import math as m
import numpy as np
import pandas as pd
import re
from scipy import stats
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
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

def get_kdes(depths, grid_min, grid_max, kde_grid_density):
    kde = sm.nonparametric.KDEUnivariate(depths)
    kde.fit(bw=np.percentile(depths, 50) * 0.04) # median/25: seems like a decent heuristic
    return kde.evaluate(np.linspace(grid_min, grid_max, val_to_grid_idx(grid_max, kde_grid_density, grid_min) + 1))

def set_curr_len_gp_stats(curr_len_gp_stats, curr_len_gp_seqs, depth_max_pctl, depth_max_pctl_rank):
    curr_len_gp_stats['count'] = curr_len_gp_seqs.shape[0]
    curr_len_gp_stats['min_len'] = curr_len_gp_seqs.len.min()
    curr_len_gp_stats['max_len'] = curr_len_gp_seqs.len.max()
    curr_len_gp_stats['max_depth'] = curr_len_gp_seqs.mean_kmer_depth.max()
    curr_len_gp_stats['max_depth_in_est'] = depth_max_pctl
    curr_len_gp_stats['max_depth_pctl_rank_in_est'] = depth_max_pctl_rank
    return curr_len_gp_stats

def val_to_grid_idx(val, grid_density, minval):
    return (grid_density * (val - minval))

def get_min_density_depth_candidate(depths, density, grid_min, kde_grid_density):
    # hopefully universally effective heuristic to avoid inappropriately applying to very narrow depth distributions
    # location of min., if any, between error distribution and mode not likely to be lower than this; and if it is, hopefully little harm done being off by < 1
    if depths[0] + 1 < np.percentile(depths, 10):
        soft_lb_idx = m.ceil(val_to_grid_idx(depths[0] + 1, kde_grid_density, grid_min))
        return (soft_lb_idx + np.argmin(density[soft_lb_idx:m.floor(val_to_grid_idx(np.percentile(depths, 20), kde_grid_density, grid_min))]))
    return 0

def truncate_depths(depths, min_density_depth_candidate, density, depth_max_pctl):
    # condition mostly to exclude cases without perceptible error distribution, i.e. most cases, except small genomes
    if min_density_depth_candidate > 0 and np.mean(density[:min_density_depth_candidate]) > density[min_density_depth_candidate]:
        depths = depths[min_density_depth_candidate:]
    return depths[depths <= depth_max_pctl]

def grid_idx_to_val(idx, grid_density, minval):
    return ((idx / grid_density) + minval)

def get_len_group_mode(min_density_depth_candidate, density, grid_min, kde_grid_density):
    mode_idx = min_density_depth_candidate + np.argmax(density[min_density_depth_candidate:])
    return grid_idx_to_val(mode_idx, kde_grid_density, grid_min)

def get_approx_component_obs(data, mode, half, data_min): # assumes symmetric distribution
    return data[(data >= max(data_min, mode - half)) & (data <= mode + half)]

def setup_mode_densities_and_cdfs(mode, len_group_mode, depths, density, grid_min, kde_grid_density, density_ECDF):
    if (0.5 * mode) <= depths[0]:
        density_at_modes, cdf_at_modes = [0], [0]
    else:
        density_at_modes, cdf_at_modes = [get_density_for_idx(val_to_grid_idx(0.5 * mode, kde_grid_density, grid_min), density)], [density_ECDF([0.5 * mode])[0]]
    if mode < depths[-1]:
        density_at_modes.append(get_density_for_idx(val_to_grid_idx(mode, kde_grid_density, grid_min), density))
        cdf_at_modes.append(density_ECDF([mode])[0])
    # min(x1, x2): see notes with illustration
    i, modes_in_depths = 2, round(depths[-1] * 1.0 / mode)
    for i in np.arange(2.0, min(round((len_group_mode * 1.0 / mode) + 2.5), modes_in_depths + 1 - int(density_ECDF([(modes_in_depths - 0.5) * mode])[0] > 0.99))):
        if i * mode <= depths[-1]:
            density_at_modes.append(get_density_for_idx(val_to_grid_idx(i * mode, kde_grid_density, grid_min), density))
            cdf_at_modes.append(density_ECDF([i * mode])[0])
    return (density_at_modes, cdf_at_modes)

def get_gamma_min_density_ratio(copynums_in_90thpctl_mode_diff):
    return (0.65 + min(copynums_in_90thpctl_mode_diff / 40, 1) * 0.2)

def get_gamma_min_cdf(copynums_in_90thpctl_mode_diff):
    len_group_mode_pctl_rank = stats.percentileofscore(depths, len_group_mode)
    return ((len_group_mode_pctl_rank + ((90.0 - len_group_mode_pctl_rank) / m.pow(copynums_in_90thpctl_mode_diff, 1/3))) / 100.0)

def get_density_for_idx(idx, density):
    return (density[m.floor(idx)] + (idx - m.floor(idx)) * (density[m.ceil(idx)] - density[m.floor(idx)]))

def get_component_weights(density_at_modes, use_gamma, cdf_at_modes = None):
    if use_gamma:
        gamma_prob = 2.0 * (1 - cdf_at_modes[-1]) / 3
        component_weights = np.zeros(len(density_at_modes))
        component_weights[:-1] = np.array(density_at_modes[:-1]) * (1 - gamma_prob) / sum(density_at_modes[:-1])
        component_weights[-1] = gamma_prob
    else:
        component_weights = np.array(density_at_modes) / sum(density_at_modes)
    return component_weights

def init_dummy_model():
    dummy = ConstantModel(prefix='dummy_')
    dummy.set_param_hint('c', value=0, vary=False)
    return dummy

def init_gaussian(i, component_weights):
    model = None
    if component_weights[m.floor(i)] > 0:
        numstr = str(i)
        if i < 1:
            numstr = 'half'
        prefix = 'gauss' + numstr + '_'
        model = GaussianModel(prefix=prefix)
    return model

def init_params(model):
    params = Parameters()
    params.update(model.make_params())
    return params

def init_gaussians(components, component_weights, copynum_components, params, param_guesses, smallest_copynum, max_gaussian_copynums):
    mode, mode_min, mode_max = param_guesses['mode'], param_guesses['mode_min'], param_guesses['mode_max']
    sigma, sigma_min = param_guesses['sigma'], param_guesses['sigma_min']
    for j in [0.5] + list(range(1, m.floor(max_gaussian_copynums) + 1)):
        model = init_gaussian(j, component_weights)
        components.append(model)
        if model:
            copynum_components = copynum_components + model
            params.update(model.make_params())
            if j == smallest_copynum:
                params[model.prefix + 'center'].set(value = mode * smallest_copynum, min = mode_min * smallest_copynum, max = mode_max * smallest_copynum)
                params[model.prefix + 'sigma'].set(value = sigma * smallest_copynum, min = sigma_min * smallest_copynum, max = depths[-1] - depths[0])
                smallest_prefix = model.prefix
            else:
                params[model.prefix + 'center'].set(vary = False, expr = str(j / smallest_copynum) + ' * ' + smallest_prefix + 'center')
                params[model.prefix + 'sigma'].set(vary = False, expr = str(j / smallest_copynum) + ' * ' + smallest_prefix + 'sigma')
            params[model.prefix + 'amplitude'].set(value = component_weights[m.floor(j)], min = NONNEG_CONSTANT, max = 1 - NONNEG_CONSTANT)
    return (components, copynum_components, params)

def init_components_and_params(component_weights, param_guesses, smallest_copynum, max_gaussian_copynums):
    copynum_components = init_dummy_model()
    return init_gaussians([], component_weights, copynum_components, init_params(copynum_components), param_guesses, smallest_copynum, max_gaussian_copynums)

def init_gamma(depths, components, copynum_components, params, mode, sigma, mode_error):
    gamma_model = Model(gamma, prefix='gamma_')
    components.append(gamma_model)
    params.update(gamma_model.make_params())
    j = len(components) - 1
    curr_mode = j * mode
    tail_stats = stats.describe(depths[depths > curr_mode])
    params['gamma_mode'] = Parameter(value = curr_mode + mode, min = j * (1 - mode_error - NONNEG_CONSTANT) * mode, max = tail_stats.mean) # avoid starting too near boundary
    params['gamma_shape'].set(value = tail_stats.mean * 1.0 / (tail_stats.mean - curr_mode), min = 1 + (mode * 1.0) / (depths[-1] - curr_mode))
    params['gamma_scale'].set(value = tail_stats.mean - curr_mode, min = NONNEG_CONSTANT, max = depths[-1] - curr_mode)
    params['gamma_mean'] = Parameter(expr = 'gamma_mode + gamma_scale')
    params['gamma_loc'].set(expr = 'gamma_mean - gamma_shape * gamma_scale', min = -1.0 * curr_mode, max = (j - 1) * (1 - mode_error) * mode) # 9 = 2 * 4.5
    params['gamma_variance'] = Parameter(expr = 'gamma_shape * (gamma_scale ** 2)', min = sigma ** 2, max = stats.describe(depths).variance)
    pre_obs = depths[(depths > (j-1) * mode) & (depths <= curr_mode)]
    pre, post = 0.5 * pre_obs.size, depths[depths > curr_mode].size
    gamma_weight_model = ConstantModel(prefix='gamma_wt_')
    params.update(gamma_weight_model.make_params())
    copynum_components = copynum_components + gamma_weight_model * gamma_model
    return (components, copynum_components, params)

def init_genome_scale_model(params, model_const):
    genome_scale_model = ConstantModel(prefix='genomescale_')
    params.update(genome_scale_model.make_params(c=model_const))
    return genome_scale_model

def finalise_params(params, components, smallest_copynum, use_gamma):
    if len(components) - 1 == m.floor(smallest_copynum):
        params[components[m.floor(smallest_copynum)].prefix + 'amplitude'].set(value = 1.0, vary=False)
    else:
        wt_expr = '1 - ' + ' - '.join(map(lambda c: c.prefix + 'amplitude', components[m.floor(smallest_copynum):-1]))
        wt_param = components[-1].prefix + 'amplitude'
        if use_gamma:
            wt_param = 'gamma_wt_c'
        params[wt_param].set(expr = wt_expr, min = NONNEG_CONSTANT, max = 1 - NONNEG_CONSTANT)
    return params

def fit(depths, mixture_model, params):
    step = (depths[-1] - depths[0]) / m.floor(depths.size * 1.0 / 100) # heuristic n...
    lb_pts = np.arange(m.floor(depths[0]), m.ceil(depths[0]), step)
    if lb_pts.size > 0:
        diffs = depths[0] - lb_pts
        lb = lb_pts[diffs >= 0][diffs[diffs >= 0].argmin()]
    else:
        lb = depths[0]
    ub = lb + step * (1 + m.ceil((depths[-1] - lb) / step))
    lmfit_range = np.arange(lb, ub, step)
    return mixture_model.fit(np.histogram(depths, lmfit_range)[0], params, x=lmfit_range[1:])

def finalise_and_fit(components, copynum_components, params, smallest_copynum, use_gamma, depths_size):
    mixture_model = init_genome_scale_model(params, depths_size) * copynum_components
    params = finalise_params(params, components, smallest_copynum, use_gamma)
    return fit(depths, mixture_model, params)

def setup_and_fit(depths, density, grid_min, kde_grid_density, param_guesses, len_group_mode, mode_error):
    mode, mode_min, mode_max = param_guesses['mode'], param_guesses['mode_min'], param_guesses['mode_max']
    sigma, sigma_min = param_guesses['sigma'], param_guesses['sigma_min']
    density_ECDF = ECDF(depths)
    density_at_modes, cdf_at_modes = setup_mode_densities_and_cdfs(mode, len_group_mode, depths, density, grid_min, kde_grid_density, density_ECDF)
    i, use_gamma, vary_copynum = len(density_at_modes) - 1, False, False
    if (i > 1) and ((i + 1) * mode <= depths[-1]):
        vary_copynum = True
        copynums_in_90thpctl_mode_diff = (np.percentile(depths, 90) - len_group_mode) * 1.0 / mode
        gamma_min_density_ratio = get_gamma_min_density_ratio(copynums_in_90thpctl_mode_diff)
        gamma_min_cdf = get_gamma_min_cdf(copynums_in_90thpctl_mode_diff)
        density_ratio = density_at_modes[-1] / density_at_modes[-2]
        while (cdf_at_modes[-1] < 0.95) and (density_ratio > 0.1):
            i += 1
            if (cdf_at_modes[-1] > gamma_min_cdf) and (density_ratio > gamma_min_density_ratio):
                use_gamma, i = True, i - 1
                break
            density_at_modes.append(get_density_for_idx(val_to_grid_idx(i * mode, kde_grid_density, grid_min), density))
            cdf_at_modes.append(density_ECDF([i * mode])[0])
            density_ratio = density_at_modes[-1] / density_at_modes[-2]
    lb = i - 1
    if vary_copynum:
        lb = i - 2
        if ((i + 1) * mode <= depths[-1]) and (density_ECDF([(i + 1) * mode])[0] < 0.99):
            i += 1
            density_at_modes.append(get_density_for_idx(val_to_grid_idx(i * mode, kde_grid_density, grid_min), density))
            cdf_at_modes.append(density_ECDF([i * mode])[0])
    smallest_copynum = 0.5
    if density_at_modes[0] == 0:
        if density_at_modes[1] == 0:
            raise RuntimeError
        smallest_copynum = 1
    aic = np.inf
    for j in range(i, lb, -1):
        component_weights = get_component_weights(density_at_modes[:(j+1)], use_gamma, cdf_at_modes[:(j+1)])
        max_gaussian_copynums = max(0.5, int(depths[-1] > mode) * len(component_weights) - 1 - int(use_gamma))
        components, copynum_components, params = init_components_and_params(component_weights, param_guesses, smallest_copynum, max_gaussian_copynums)
        if use_gamma:
            components, copynum_components, params = init_gamma(depths, components, copynum_components, params, mode, sigma, mode_error)
        # Finally estimate copy numbers using lmfit
        result_temp = finalise_and_fit(components, copynum_components, params, smallest_copynum, use_gamma, depths.size)
        if (result_temp.aic / aic) < 1 + min(0.025, 0.005 * (i - 3)): # slight penalty for higher # of components, increasing with i ("base" #)
            result, aic = result_temp, result_temp.aic
    return result

def get_smallest_copynum(prefixes):
    if 'gausshalf_' in prefixes:
        return 0.5
    return 1

def gamma(x, shape, loc, scale):
    return stats.gamma.pdf(x, shape, loc, scale)

def add_to_copynum_stats(data, cols, stats_hash):
    for i in range(len(data)):
        stats_hash[cols[i]].append(data[i])

# Assume variance should only be estimated 1. Directly from observations around peak (mode), and 2. Only if it is <= m, m being the mode depth of the longest sequences.
# (2) is because variance is more likely to be overestimated from observations around a peak with a location > m, which are likely to include a relatively high proportion of
# sequences from copy# components other than those from the component represented by the peak
def variance_from_curve(len_group_mode, mode, longest_seqs_mode1_copynum, mode_error):
    ratio = len_group_mode / mode
    half_mode_error = 0.5 * mode_error
    ret = (ratio >= (0.5 - half_mode_error) and ratio <= (0.5 + half_mode_error))
    if longest_seqs_mode1_copynum == 1:
        return (ret or (ratio >= (1 - mode_error) and ratio <= (1 + mode_error)))
    return ret

# Fit linearised exponential decay: sigma = A * exp(K * length_median) => log(sigma) = K * length_median + log(A)
def guess_next_sigma(length_median, length_medians, sigmas):
    K, log_A = np.polyfit(length_medians, np.log(sigmas), 1)
    return m.exp(K * length_median + log_A)

def get_component_params(prefix, params):
    if re.match('gauss', prefix):
        return (params[prefix + 'amplitude'].value, params[prefix + 'center'].value, params[prefix + 'sigma'].value, None, None, None)
    return (params['gamma_wt_c'].value, params['gamma_mean'].value, m.sqrt(params['gamma_variance'].value), params['gamma_loc'].value, params['gamma_shape'].value, params['gamma_scale'].value)

def compute_gaussian_density_at(x, prefix, params):
    wt, mean, sigma = get_component_params(prefix, params)[:3]
    return wt * stats.norm.pdf(x, mean, sigma)

def compute_density_at(x, prefix, params):
    if re.match('gauss', prefix):
        return compute_gaussian_density_at(x, prefix, params)
    return params['gamma_wt_c'] * stats.gamma.pdf(x, params['gamma_shape'], params['gamma_loc'], params['gamma_scale'])

def compute_likeliest_copynum_at(x, prefixes, smallest_copynum, use_gamma, include_half, params, empirical_dens = None):
    densities = [0] * (len(prefixes) + int(smallest_copynum == 1))
    if not(include_half) and len(densities) == 1:
        densities.append(0)
    if 'gauss1_' in prefixes: # should always be True
        densities[1] = compute_density_at(x, 'gauss1_', params)
    if smallest_copynum == 0.5:
        densities[int(not(include_half))] += compute_density_at(x, 'gausshalf_', params)
    for i in range(2, len(densities) - int(use_gamma)):
        densities[i] = compute_density_at(x, 'gauss' + str(i) + '_', params)
    if use_gamma:
        densities[i+1] = compute_density_at(x, 'gamma_', params)
    maxdens_idx = np.argmax(densities)
    if empirical_dens:
        if empirical_dens - sum(densities) > densities[maxdens_idx]:
            return 0
    return (maxdens_idx or 0.5)


argparser = argparse.ArgumentParser(description='Estimate genomic copy number for haploid or diploid whole-genome shotgun assembly sequences')
argparser.add_argument('--half', action="store_true", help='Include copy number 0.5, i.e. heterozygous single-copy, in sequence classification')
argparser.add_argument('unitigs_file', type=str, help='FASTA file listing sequences to be classified')
argparser.add_argument('kmer_len', type=int, help='Value of k used in assembly that output sequences to be classified')
argparser.add_argument('output_dir', type=str, help='Directory to which output files should be written')
args = argparser.parse_args()

NONNEG_CONSTANT = 1.e-12

seq_lens = array.array('L')
seq_mean_kmer_depths = array.array('d')
seq_gc_contents = array.array('d')

with open(args.unitigs_file) as unitigs:
    line = unitigs.readline()
    while line:
        if re.search('^>[0-9]', line):
            row = list(map(int, line[1:].split()))
            seq_lens.append(row[1])
            kmers = row[1] - args.kmer_len + 1
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
seqs['likeliest_copynum'] = -1.0
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

LEN_GP_STATS_COLS = ['count', 'min_len', 'max_len', 'max_depth', 'max_depth_in_est', 'max_depth_pctl_rank_in_est', 'min_copynum', 'max_copynum_est']
len_gp_stats = []
COPYNUM_STATS_COLS = ['len_gp_id', 'len_gp_min_len', 'len_gp_max_len', 'copynum', 'depth_lb', 'depth_max', 'weight', 'depth_mean', 'depth_stdev', 'depth_loc', 'depth_shape', 'depth_scale']
copynum_stats = []

# Fit under assumption that first peak of density curve for longest sequences corresponds to mode of copy-number 0.5 or 1 (unique homozygous) sequences
mode_error = 0.1
aic = np.inf
better_fit_model = 1

log_file = open(args.output_dir + '/log.txt', 'w', newline='')

for longest_seqs_mode1_copynum in [0.5, 1.0]:
    log_header = 'ESTIMATION ROUND ' + str(longest_seqs_mode1_copynum * 2) + ': ASSUME 1ST PEAK OF DENSITY CURVE FOR LONGEST SEQUENCES CORRESPONDS TO MODE OF COPY-NUMBER '
    log_header += str(longest_seqs_mode1_copynum) + ' SEQUENCES\n'
    log_file.write(log_header)

    len_gp_stats.append(pd.DataFrame(data=None, index=pd.RangeIndex(stop=length_gps_count), columns=LEN_GP_STATS_COLS))
    length_gp_sigmas = [None] * length_gps_count
    copynum_stats_hash = { col: [] for col in COPYNUM_STATS_COLS }
    aic_current = 0
    mode, mode_min, mode_max, sigma_min = np.nan, 0, np.inf, 0 # 0 mins because NONNEG_CONSTANT resulted in no CIs (numerical instability?)

    for len_gp_idx in range(length_gps_count - 1, -1, -1):
        print(len_gp_idx)
        # Estimate any probable error distribution, and mode excluding error sequences
        depths = np.copy(length_gps_for_est[len_gp_idx].mean_kmer_depth.values)
        depths.sort()
        offset = 0.1
        depth_max_pctl_rank = 100 - (stats.variation(depths) * 1.5)
        depth_max_pctl = np.percentile(depths, depth_max_pctl_rank)
        kde_grid_density, grid_min, grid_max = 20, depths[0] - offset, depth_max_pctl + offset # last also seems like a decent heuristic
        density = get_kdes(depths, grid_min, grid_max, kde_grid_density)
        curr_len_gp_stats = set_curr_len_gp_stats(len_gp_stats[-1].loc[len_gp_idx], length_gps_for_est[len_gp_idx], depth_max_pctl, depth_max_pctl_rank)

        min_density_depth_idx_1 = get_min_density_depth_candidate(depths, density, grid_min, kde_grid_density)
        depths = truncate_depths(depths, min_density_depth_idx_1, density, depth_max_pctl)

        len_group_mode = get_len_group_mode(min_density_depth_idx_1, density, grid_min, kde_grid_density)
        if np.isnan(mode):
            mode = len_group_mode
            mode /= longest_seqs_mode1_copynum # for consistency: let it represent c#1
        # Estimate standard deviation of copy-number 1 sequences
        if variance_from_curve(len_group_mode, mode, longest_seqs_mode1_copynum, mode_error):
            # Perhaps it's a mistake not to enforce that mode1_copynum be <= 2?
            sigma = np.std(get_approx_component_obs(depths, len_group_mode, mode * 0.25, depths[0])) / (round(len_group_mode / mode) or 0.5)
        else:
            sigma = guess_next_sigma(length_gp_medians[len_gp_idx], length_gp_medians[(len_gp_idx + 1):], length_gp_sigmas[(len_gp_idx + 1):])
        if sigma < sigma_min:
            sigma_min = 0

        param_guesses = { 'mode': mode, 'mode_min': mode_min, 'mode_max': mode_max, 'sigma': sigma, 'sigma_min': sigma_min }
        try:
            result = setup_and_fit(depths, density, grid_min, kde_grid_density, param_guesses, len_group_mode, mode_error)
        except RuntimeError:
            error_msg = 'Lowest copy number for sequences of length ' + curr_len_gp_stats.min_len + ' to ' + curr_len_gp_stats.max_len + ' in dataset higher than 1: '
            error_msg += 'none are single-copy (either homo- or heterozygous)!'
            raise RuntimeError(error_msg)
        copynum_component_prefixes = set(map(lambda name: re.search(r'(.+?_)', name).group(), result.params.valuesdict().keys())) - { 'dummy_', 'genomescale_' }
        use_gamma = ('gamma_' in copynum_component_prefixes)
        smallest_copynum = get_smallest_copynum(copynum_component_prefixes)
        len_gp_stats[-1].loc[len_gp_idx, 'min_copynum'] = smallest_copynum
        len_gp_stats[-1].loc[len_gp_idx, 'max_copynum_est'] = len(copynum_component_prefixes) - int(smallest_copynum == 0.5)
        aic_current += result.aic

        # Set mode the first time, i.e. from estimation and classification of longest sequences
        smallest_copynum_prefix = 'gauss' + str(smallest_copynum) + '_'
        if smallest_copynum == 0.5:
            smallest_copynum_prefix = 'gausshalf_'
        if mode_max == np.inf:
            mode = result.params[smallest_copynum_prefix + 'center'].value / smallest_copynum
            mode = result.params[smallest_copynum_prefix + 'center'].value / smallest_copynum
            mode_min = (1 - mode_error) * mode
            mode_max = (1 + mode_error) * mode
        sigma_err = result.params[smallest_copynum_prefix + 'sigma'].stderr or 0
        sigma_min = (result.params[smallest_copynum_prefix + 'sigma'].value - sigma_err) / smallest_copynum
        if sigma_err == 0:
            sigma_min = (1 - mode_error) * sigma_min
        length_gp_sigmas[len_gp_idx] = result.params[smallest_copynum_prefix + 'sigma'].value / smallest_copynum

        # Compute copy# component boundaries (enough to classify observations; may not be technically accurate [i.e. at population level])
        def eval_copynum_at(depth, nonzero, likeliest_copynums, likeliest_copynum_ubs):
            empirical_dens = None
            if not(nonzero):
                empirical_dens = get_density_for_idx(val_to_grid_idx(depth, kde_grid_density, grid_min), density)
            copynum = compute_likeliest_copynum_at(depth, copynum_component_prefixes, smallest_copynum, use_gamma, args.half, result.params, empirical_dens)
            if copynum != likeliest_copynums[-1]:
                if copynum > 0:
                    nonzero = True
                likeliest_copynum_ubs.append(depth)
                likeliest_copynums.append(copynum)
            return nonzero

        likeliest_copynums, likeliest_copynum_ubs = [], []
        depth, step = grid_min + offset, 0.02
        empirical_dens = get_density_for_idx(val_to_grid_idx(depth, kde_grid_density, grid_min), density)
        copynum = compute_likeliest_copynum_at(depth, copynum_component_prefixes, smallest_copynum, use_gamma, args.half, result.params, empirical_dens)
        nonzero = (copynum > 0)
        likeliest_copynums.append(copynum)
        for depth in np.arange(depth + step, depths[0], step):
            nonzero = eval_copynum_at(depth, nonzero, likeliest_copynums, likeliest_copynum_ubs)
        for i in range(0, len(depths)):
            nonzero = eval_copynum_at(depths[i], nonzero, likeliest_copynums, likeliest_copynum_ubs)

        # Initial assignments might be slightly out of order: have to infer orderly final assignments
        copynum_assnmts, copynums_unique = [likeliest_copynums[0]], { likeliest_copynums[0] }
        copynum_lbs = [np.inf] * (len(copynum_component_prefixes) + int(smallest_copynum == 1))
        copynum_ubs = [np.inf] * (len(copynum_component_prefixes) + int(smallest_copynum == 1))
        if copynum_assnmts[0] > 0:
            copynum_lbs[m.floor(copynum_assnmts[0])] = 0
        # Assume that 1. Larger copy#s don't occur in order before smaller copy#s, e.g. 1, 3, 4, 2 does not occur
        # (whereas 1, 4, 3, 2 could [copy#4 variance is larger than that of 3])
        # 2. Out-of-order copy#s are always followed eventually by in-order copy#s, e.g.: 1, 3, 2 will be followed eventually by a copy# >= 3
        for i in range(1, len(likeliest_copynums)):
            if likeliest_copynums[i] not in copynums_unique:
                if likeliest_copynums[i] < copynum_assnmts[-1]:
                    copynum_lbs[m.floor(copynum_assnmts[-1])] = np.inf
                    copynums_unique.remove(copynum_assnmts[-1])
                    copynum_assnmts.pop()
                if len(copynum_assnmts) == 0:
                    copynum_lbs[m.floor(likeliest_copynums[i])] = 0
                else:
                    copynum_lbs[m.floor(likeliest_copynums[i])] = likeliest_copynum_ubs[i-1]
                    if copynum_assnmts[-1] > 0:
                        copynum_ubs[m.floor(copynum_assnmts[-1])] = likeliest_copynum_ubs[i-1]
                copynum_assnmts.append(likeliest_copynums[i])
                copynums_unique.add(likeliest_copynums[i])

        # Assign to sequences in the corresponding ranges
        gp_len_condition = (seqs.len >= curr_len_gp_stats.min_len) & (seqs.len <= curr_len_gp_stats.max_len)
        seqs.loc[gp_len_condition, 'modex'] = seqs.loc[gp_len_condition].mean_kmer_depth / mode
        seqs.loc[gp_len_condition, 'est_gp'] = len_gp_idx
        # Note: 1. copynum_lbs[i] == copynum_ubs[i-1]. 2. 0 does not have an entry in copynum_lbs and copynum_ubs.
        if len(copynum_assnmts) > 1:
            seqs.loc[gp_len_condition & (seqs.mean_kmer_depth < copynum_lbs[m.floor(copynum_assnmts[1])]), 'likeliest_copynum'] = copynum_assnmts[0]
        else:
            seqs.loc[gp_len_condition, 'likeliest_copynum'] = copynum_assnmts[0]
        for i in range(1, len(copynum_assnmts)):
            idx = m.floor(copynum_assnmts[i])
            seqs.loc[gp_len_condition & (seqs.mean_kmer_depth >= copynum_lbs[idx]) & (seqs.mean_kmer_depth < copynum_ubs[idx]), 'likeliest_copynum'] = copynum_assnmts[i]

        def get_copynum_stats_data(idx, wt, mean, sigma, loc = None, shape = None, scale = None): # copy number idx
            return [len_gp_idx, curr_len_gp_stats.min_len, curr_len_gp_stats.max_len, idx, copynum_lbs[m.floor(idx)], copynum_ubs[m.floor(idx)], wt, mean, sigma, loc, shape, scale]

        wt, mean, sigma = 0, 0, 0
        wt_i, mean_i, sigma_i = 0, 0, 0
        if smallest_copynum < 1:
            wt, mean, sigma = get_component_params('gausshalf_', result.params)[:3]
        has_copynum1 = (smallest_copynum == 1) or next((True for p in copynum_component_prefixes if re.search(r'gauss1', p)), False)
        if has_copynum1:
            wt_i, mean_i, sigma_i = get_component_params('gauss1_', result.params)[:3]
        if args.half:
            add_to_copynum_stats(get_copynum_stats_data(0.5, wt, mean, sigma), COPYNUM_STATS_COLS, copynum_stats_hash)
            if has_copynum1:
                add_to_copynum_stats(get_copynum_stats_data(1, wt_i, mean_i, sigma_i), COPYNUM_STATS_COLS, copynum_stats_hash)
        else:
            wt1 = wt + wt_i
            wt_normed, wt_i_normed = wt/wt1, wt_i/wt1
            mean1 = (wt_normed * mean) + (wt_i_normed * mean_i)
            sigma1 = m.sqrt((wt_normed * sigma**2) + (wt_i_normed * sigma_i**2) + (wt_normed * wt_i_normed * (mean - mean_i)**2))
            copynum_stats_data = [len_gp_idx, curr_len_gp_stats.min_len, curr_len_gp_stats.max_len, 1, copynum_lbs[0], copynum_ubs[int(has_copynum1)], wt1, mean1, sigma1, None, None, None]
            add_to_copynum_stats(copynum_stats_data, COPYNUM_STATS_COLS, copynum_stats_hash)
        for i in range(2, len(copynum_component_prefixes) - int(smallest_copynum < 1) - int(use_gamma) + 1):
            wt, mean, sigma, loc, shape, scale = get_component_params('gauss' + str(i) + '_', result.params)
            add_to_copynum_stats(get_copynum_stats_data(i, wt, mean, sigma, loc, shape, scale), COPYNUM_STATS_COLS, copynum_stats_hash)
        if use_gamma:
            wt, mean, sigma, loc, shape, scale = get_component_params('gamma_', result.params)
            add_to_copynum_stats(get_copynum_stats_data(i + 1, wt, mean, sigma, loc, shape, scale), COPYNUM_STATS_COLS, copynum_stats_hash)

        log_file.write(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H:%M:%S : Sequence group ') + str(len_gp_idx) + ' estimated\n')
        log_file.write('Group minimum and maximum lengths: ' + str(curr_len_gp_stats.min_len) + ', ' + str(curr_len_gp_stats.max_len) + '\n')
        log_file.write('Maximum mean k-mer depth of all sequences in group: ' + str(curr_len_gp_stats.max_depth) + '. ')
        log_file.write('Maximum used in estimation: ' + str(depth_max_pctl) + ' (' + str(depth_max_pctl_rank) + ' percentile).\n\n')
        log_file.write('Fit report:\n')
        log_file.write(result.fit_report())
        log_file.write('\n\n')

    copynum_stats.append(pd.DataFrame.from_dict(copynum_stats_hash))
    if aic_current < aic:
        aic = aic_current
        better_fit_model = longest_seqs_mode1_copynum
    log_file.write('Sum of per-length-group model AICs: ' + str(aic))
    log_file.write('\n\n')
    print('')
    seq_label_filename = args.output_dir + '/sequence-labels.csv'
    if better_fit_model == longest_seqs_mode1_copynum:
        seqs.loc[:, 'len':].to_csv(seq_label_filename, header=['Length', 'Average k-mer depth', '1st Mode X', 'GC %', 'Estimation length group', 'Likeliest copy #'], index_label='ID')

log_footer = 'BETTER-FIT MODEL (LOWER SUM OF PER-LENGTH-GROUP MODEL AIC SCORES): 1ST PEAK OF DENSITY CURVE FOR LONGEST SEQUENCES CORRESPONDS TO MODE OF COPY-NUMBER '
log_footer += str(better_fit_model) + ' SEQUENCES\n'
log_file.write(log_footer)
log_file.close()

# Write length group and copy-number component stats
LEN_GP_STATS_OUTPUT_COLS = tuple(['count', 'min_len', 'max_len', 'max_depth', 'max_depth_in_est', 'min_copynum', 'max_copynum_est'])
LEN_GP_STATS_OUTPUT_HEADER = ['Number of sequences', 'Min. len.', 'Max. len.', 'Max. depth', 'Max. depth in estimation', 'Smallest copy # present', 'Largest copy # estimated']
len_gp_stats[m.floor(better_fit_model)].to_csv(args.output_dir + '/length_gp_stats.csv', columns=LEN_GP_STATS_OUTPUT_COLS, header=LEN_GP_STATS_OUTPUT_HEADER, index_label='ID')

COPYNUM_STATS_OUTPUT_HEADER = ['Group #', 'Group min. len.', 'Group max. len.', 'Component #', 'Component depth lower bound', 'Component max. depth',
                               'Weight', 'Mean', 'Std. deviation', 'Location', 'Shape', 'Scale']
copynum_stats[m.floor(better_fit_model)].to_csv(args.output_dir + '/copynumber_params.csv', header=COPYNUM_STATS_OUTPUT_HEADER, index=False)

