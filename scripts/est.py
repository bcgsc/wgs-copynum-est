import argparse
import csv
import datetime
from lmfit import Model, Parameter, Parameters
from lmfit.models import ConstantModel, ExponentialModel, GaussianModel
import math as m
import numpy as np
import os
import pandas as pd
import re
from scipy import stats
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
import sys
import time

if os.getenv('WGS_COPYNUM_EST_HOME'):
  sys.path.insert(0, os.path.join(os.getenv('WGS_COPYNUM_EST_HOME'), 'scripts'))
else:
  raise RuntimeError('Please set environment variable WGS_COPYNUM_EST_HOME before running script')

import utils.utils as utils


def val_to_grid_idx(val, grid_density, minval):
    return (grid_density * (val - minval))

def get_kdes(depths, grid_min, grid_max, kde_grid_density):
    kde = sm.nonparametric.KDEUnivariate(depths)
    kde.fit(bw=np.percentile(depths, 50) * 0.04) # median/25: seems like a decent heuristic
    return kde.evaluate(np.linspace(grid_min, grid_max, val_to_grid_idx(grid_max, kde_grid_density, grid_min) + 1))

def set_curr_len_gp_stats(curr_len_gp_stats, curr_len_gp_seqs, depth_max_pctl, depth_max_pctl_rank):
    curr_len_gp_stats['count'] = curr_len_gp_seqs.shape[0]
    curr_len_gp_stats['min_len'] = curr_len_gp_seqs.length.min()
    curr_len_gp_stats['max_len'] = curr_len_gp_seqs.length.max()
    curr_len_gp_stats['max_depth'] = curr_len_gp_seqs.mean_kmer_depth.max()
    curr_len_gp_stats['max_depth_in_est'] = depth_max_pctl
    curr_len_gp_stats['max_depth_pctl_rank_in_est'] = depth_max_pctl_rank
    return curr_len_gp_stats

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

def grid_idx_to_val(idx, grid_density, minval):
    return ((idx / grid_density) + minval)

def get_len_group_mode(min_density_depth_candidate, density, grid_min, kde_grid_density):
    mode_idx = min_density_depth_candidate + np.argmax(density[min_density_depth_candidate:])
    return grid_idx_to_val(mode_idx, kde_grid_density, grid_min)

def get_approx_component_obs(data, mode, half, data_min): # assumes symmetric distribution
    return data[(data >= max(data_min, mode - half)) & (data <= mode + half)]

def get_density_for_idx(idx, density):
    return (density[m.floor(idx)] + (idx - m.floor(idx)) * (density[m.ceil(idx)] - density[m.floor(idx)]))

# Assumes that density increases in some ranges between min. depth and 0.5 * mode through mode: o/w, density at both mode and half-mode should be * 0.5
def setup_mode_densities_and_cdfs(mode, len_group_mode, depths, density, grid_min, kde_grid_density, min_density_depth_idx1, depth_ECDF, haploid):
    density_at_modes, cdf_at_modes = { 0.0: 0 }, { 0.0: 0 }
    density_1 = int(mode < depths[-1]) and get_density_for_idx(val_to_grid_idx(mode, kde_grid_density, grid_min), density)
    if haploid:
        peak_copynum = 1
    else:
        density_at_modes[0.5], cdf_at_modes[0.5] = 0, 0
        density_pt5 = int(depths[0] <= 0.5 * mode) and get_density_for_idx(val_to_grid_idx(0.5 * mode, kde_grid_density, grid_min), density)
        if (depths[0] <= 0.5 * mode) and (mode < depths[-1]):
            peak_copynum = int(density_pt5 < density_1) # 0 if density_pt5 == density_1
            if density_pt5 > density_1:
                peak_copynum = 0.5
            modept5_idx_flr = m.floor(val_to_grid_idx(0.5 * mode, kde_grid_density, grid_min))
            mode_pt5to1_mindens = np.min(density[modept5_idx_flr:(modept5_idx_flr+1)])
    if depths[0] <= 0.75 * 0.5 * mode:
        x = min_density_depth_idx1
        if x >= 0.5 * mode:
            x = 0
        density_at_modes[0.0], cdf_at_modes[0.0] = get_density_for_idx(x, density), depth_ECDF([grid_idx_to_val(x, kde_grid_density, grid_min)])[0]
    # The better defined or more separated (smaller mode_pt5to1_mindens / peak_density) the densities around mode and 0.5mode,
    # the less the apparent density of the lower-weight component is discounted
    if not(haploid):
        if depths[0] <= 0.5 * mode:
            factor = ((mode < depths[-1]) and (peak_copynum == 1) and min(1, 0.5 + 1 - min(1, mode_pt5to1_mindens / density_pt5))) or 1
            density_at_modes[0.5], cdf_at_modes[0.5] = density_pt5 * factor, depth_ECDF([0.5 * mode])[0]
    if mode < depths[-1]:
        factor = ((depths[0] <= 0.5 * mode) and (peak_copynum == 0.5) and min(1, 0.5 + 1 - min(1, mode_pt5to1_mindens / density_1))) or 1
        density_at_modes[1.0], cdf_at_modes[1.0] = density_1 * factor, depth_ECDF([mode])[0]
    if 1 not in density_at_modes.keys():
        if haploid or density_at_modes[0.5] == 0:
            raise RuntimeError
    # min(x1, x2): see notes with illustration
    i, modes_in_depths = 2.0, round(depths[-1] * 1.0 / mode)
    for i in np.arange(2.0, min(round((len_group_mode * 1.0 / mode) + 2.5), modes_in_depths + 1.0 - int(depth_ECDF([(modes_in_depths - 0.5) * mode])[0] > 0.99))):
        if i * mode <= depths[-1]:
            density_at_modes[i] = get_density_for_idx(val_to_grid_idx(i * mode, kde_grid_density, grid_min), density)
            cdf_at_modes[i] = depth_ECDF([i * mode])[0]
    return (density_at_modes, cdf_at_modes)

def get_gamma_min_density_ratio(copynums_in_90thpctl_mode_diff):
    return (0.65 + min(copynums_in_90thpctl_mode_diff / 40, 1) * 0.2)

def get_gamma_min_cdf(depths, len_group_mode, copynums_in_90thpctl_mode_diff):
    len_group_mode_pctl_rank = stats.percentileofscore(depths, len_group_mode)
    return ((len_group_mode_pctl_rank + ((90.0 - len_group_mode_pctl_rank) / m.pow(copynums_in_90thpctl_mode_diff, 1/3))) / 100.0)

def get_component_params(prefix, params):
    if re.match('gauss', prefix):
        return (params[prefix + 'amplitude'].value, params[prefix + 'center'].value, params[prefix + 'sigma'].value, None, None, None)
    elif re.match('cgauss', prefix):
        return (params[prefix + 'amplitude1'].value + params[prefix + 'ampdiff'].value, params[prefix + 'center'].value, params[prefix + 'sigma'].value, None, None, None)
    return (params['gamma_wt_c'].value, params['gamma_mean'].value, m.sqrt(params['gamma_variance'].value), params['gamma_loc'].value, params['gamma_shape'].value, params['gamma_scale'].value)

def compute_gaussian_density_at(x, prefix, params):
    wt, mean, sigma = get_component_params(prefix, params)[:3]
    return wt * stats.norm.pdf(x, mean, sigma)

def compute_density_at(x, prefix, params):
    if re.match('c?gauss', prefix):
        ans = compute_gaussian_density_at(x, prefix, params)
        return ans
    elif re.match('exp', prefix):
        return params['exp_wt_c'] * stats.expon.pdf(x, 0, params['exp_decay'])
    return params['gamma_wt_c'] * stats.gamma.pdf(x, params['gamma_shape'], params['gamma_loc'], params['gamma_scale'])

def get_component_prefix(copynum, prefixes):
    if copynum == 0:
        return 'exp_'
    if copynum == 0.5:
        return 'gausshalf_'
    prefix = 'gauss' + str(copynum) + '_'
    if prefix in prefixes:
        return prefix
    elif ('c' + prefix) in prefixes:
        return ('c' + prefix)
    elif 'gamma_' in prefixes:
        return 'gamma_'
    return None

def get_component_weights(density_at_modes, min_density_depth_idx1, haploid, use_gamma = False, cdf_at_modes = None, next_mode_cdf = None):
    if use_gamma:
        gamma_prob = 0.5 * (next_mode_cdf - cdf_at_modes.iloc[-1]) + (1 - next_mode_cdf)
        component_weights = pd.Series(index = density_at_modes.index)
        component_weights.iloc[:-1] = density_at_modes.iloc[:-1] * (1 - gamma_prob) / density_at_modes.iloc[:-1].sum()
        component_weights[-1] = gamma_prob
    else:
        component_weights = density_at_modes / density_at_modes.sum()
    if haploid or component_weights[0.5] > 0:
        min_cpnum = int(haploid) or 0.5
        copynum_pt5_sigma_est = component_weights[min_cpnum] / (m.sqrt(2 * m.pi) * density_at_modes[min_cpnum])
        min_density1 = get_density_for_idx(val_to_grid_idx(depths[min_density_depth_idx1], kde_grid_density, grid_min), density)
        if min_density1 < 1.25 * component_weights[min_cpnum] * stats.norm.pdf(depths[min_density_depth_idx1], min_cpnum * mode, copynum_pt5_sigma_est):
            density_at_modes[0] = 0
            component_weights = density_at_modes / density_at_modes.sum()
    return component_weights

def init_dummy_model():
    dummy = ConstantModel(prefix='dummy_')
    dummy.set_param_hint('c', value=0, vary=False)
    return dummy

def gaussian(x, amplitude1, ampdiff, center, sigma):
    return ((amplitude1 + ampdiff) * stats.norm.pdf(x, center, sigma))

# 'cgauss': custom Gaussian, to enforce decreasing weights after max-density depth
def init_gaussian(i, component_weights, mode_copynum_ub):
    if component_weights[i] > 0:
        numstr = str(i)
        if i < 1:
            numstr = 'half'
        if i > mode_copynum_ub:
            return Model(gaussian, prefix='cgauss' + numstr + '_')
        else:
            return GaussianModel(prefix='gauss' + numstr + '_')
    return None

def init_params(model):
    params = Parameters()
    params.update(model.make_params())
    return params

def get_mode_copynum_ub(len_group_mode, mode, haploid, est_half):
    mode_copynum = len_group_mode * 1.0 / mode
    mode_copynum_ub = round(mode_copynum) or (haploid * 1)
    if not(haploid) and (mode_copynum < 0.75):
        mode_copynum_ub = 0.5 + int(not(est_half)) * 0.5
    return mode_copynum_ub

def init_gaussians(components, component_weights, copynum_components, params, param_guesses, len_gp_mode_copynum_ub, smallest_copynum, max_gaussian_copynums, haploid):
    mode, mode_min, mode_max = param_guesses['mode'], param_guesses['mode_min'], param_guesses['mode_max']
    sigma, sigma_min = param_guesses['sigma'], param_guesses['sigma_min']
    for j in ([0.5] * int(not(haploid))) + list(range(1, m.floor(max_gaussian_copynums) + 1)):
        model = init_gaussian(j, component_weights, len_gp_mode_copynum_ub)
        components[j] = model
        if model:
            copynum_components = copynum_components + model
            params.update(model.make_params())
            if j == smallest_copynum:
                params[model.prefix + 'center'].set(value = mode * smallest_copynum, min = mode_min * smallest_copynum, max = mode_max * smallest_copynum)
                sqrt_cpnum = m.sqrt(smallest_copynum)
                params[model.prefix + 'sigma'].set(value = sigma * sqrt_cpnum, min = sigma_min * sqrt_cpnum, max = (depths[-1] - depths[0]) or 1)
                smallest_prefix = model.prefix
            else:
                params[model.prefix + 'center'].set(vary = False, expr = str(j / smallest_copynum) + ' * ' + smallest_prefix + 'center')
                params[model.prefix + 'sigma'].set(vary = False, expr = str(m.sqrt(j / smallest_copynum)) + ' * ' + smallest_prefix + 'sigma')
            if re.search(r'^cgauss', model.prefix):
                diffval = component_weights[j] - component_weights[(j-1) or 0.5]
                idx = (max(components.keys()) - 1) or 0.5 # should always be >= 1 if haploid
                if re.search(r'^cgauss', components[idx].prefix):
                    params[model.prefix + 'amplitude1'].set(vary = False, expr = components[idx].prefix + 'amplitude1 + ' + components[idx].prefix + 'ampdiff')
                    if diffval >= 0:
                        params[model.prefix + 'ampdiff'].set(value = -0.05 * component_weights[j], max = 0)
                    else:
                        params[model.prefix + 'ampdiff'].set(value = diffval, max = 0)
                else:
                    params[model.prefix + 'amplitude1'].set(vary = False, expr = components[idx].prefix + 'amplitude')
                    params[model.prefix + 'ampdiff'].set(value = diffval, max = max(diffval * 1.1, 0))
            else:
                params[model.prefix + 'amplitude'].set(value = component_weights[j], min = NONNEG_CONSTANT, max = 1 - NONNEG_CONSTANT)
    return (components, copynum_components, params)

def init_components_and_params(component_weights, param_guesses, mode, len_gp_mode_copynum_ub, smallest_copynum, max_gaussian_copynums, haploid):
    components = {}
    if component_weights[0] == 0:
        copynum_components = init_dummy_model()
        params = init_params(copynum_components)
        components[0] = None
    else: # Model error distribution with exponential PDF: (1/T) * exp(-x/T), CDF: 1 - exp(-x/T), cdf(mode) > 0.99 <=> -mode/T < ln(0.01) <=> T < -mode / ln(0.01)
        exp_model = ExponentialModel(prefix='exp_')
        components[0] = exp_model
        params = init_params(exp_model)
        params['exp_decay'].set(value = -0.5 * mode / m.log(0.01), min = NONNEG_CONSTANT, max = -1 * mode / m.log(0.01)) # start at half the limit
        params['exp_amplitude'].set(vary = False, expr = '1.0 / exp_decay')
        exp_wt_model = ConstantModel(prefix='exp_wt_')
        params.update(exp_wt_model.make_params())
        params['exp_wt_c'].set(value = component_weights[0], min = NONNEG_CONSTANT, max = 1 - NONNEG_CONSTANT)
        copynum_components = exp_wt_model * exp_model
    return init_gaussians(components, component_weights, copynum_components, params, param_guesses, len_gp_mode_copynum_ub, smallest_copynum, max_gaussian_copynums, haploid)

def gamma(x, shape, loc, scale):
    return stats.gamma.pdf(x, shape, loc, scale)

def init_gamma(depths, components, copynum_components, params, mode, sigma, mode_error):
    gamma_model = Model(gamma, prefix='gamma_')
    j = max(components.keys()) + 1
    params.update(gamma_model.make_params())
    components[j] = gamma_model
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

def finalise_params(params, components, haploid, smallest_copynum, len_gp_mode_copynum_ub, use_gamma):
    if not(components[0]) and (max(components.keys()) == smallest_copynum):
        params[components[smallest_copynum].prefix + 'amplitude'].set(value = 1.0, vary=False)
    else:
        margin = 2 - int(haploid)
        wt_expr = '1' + (components[0] is not None) * ' - exp_wt_c' + ((smallest_copynum == 0.5) and (max(components.keys()) > 0.5)) * ' - gausshalf_amplitude'
        for i in range(1, min(m.floor(len_gp_mode_copynum_ub) + 1, len(components) - margin)):
            wt_expr += ' - ' + components[i].prefix + 'amplitude'
        for i in range(m.floor(len_gp_mode_copynum_ub) + 1, len(components) - margin):
            wt_expr += ' - ' + components[i].prefix + 'amplitude1 - ' + components[i].prefix + 'ampdiff'
        i = (len(components) - margin) or 0.5
        wt_param = components[i].prefix + 'amplitude'
        wt_min, wt_max = NONNEG_CONSTANT, 1 - NONNEG_CONSTANT
        if use_gamma:
            wt_param = 'gamma_wt_c'
        elif len_gp_mode_copynum_ub < i:
            wt_expr += ' - ' + components[i].prefix + 'amplitude1'
            wt_param = components[i].prefix + 'ampdiff'
            wt_min, wt_max = NONNEG_CONSTANT - 1, 0
        params[wt_param].set(expr = wt_expr, min = wt_min, max = wt_max)
    return params

def fit(depths, mixture_model, params):
    free_params = sum(list(map(lambda p: p.vary, params.values())))
    numsteps = max(m.floor(depths.size * 1.0 / 100), 4 * free_params)
    if depths.size * 1.0 / numsteps < 10:
      # should very rarely if ever be the latter; also assumes the latter still leaves a reasonable expected numobs per bin/step
      numsteps = max(m.floor(depths.size / 10.0), 2 * free_params)
    step = (depths[-1] - depths[0]) * 1.0 / numsteps # heuristic n...
    if step > 0:
        lb_pts = np.arange(m.floor(depths[0]), m.ceil(depths[0]), step)
        if lb_pts.size > 0:
            diffs = depths[0] - lb_pts
            lb = lb_pts[diffs >= 0][diffs[diffs >= 0].argmin()]
        else:
            lb = depths[0]
        ub = lb + step * (1 + m.ceil((depths[-1] - lb) / step))
        lmfit_range = np.arange(lb, ub, step)
        return mixture_model.fit(np.histogram(depths, lmfit_range)[0], params, x=lmfit_range[1:])
    else:
        return mixture_model.fit([1], params, x=[depths[0]])

def finalise_and_fit(components, copynum_components, params, haploid, smallest_copynum, len_gp_mode_copynum_ub, use_gamma, depths):
    mixture_model = init_genome_scale_model(params, depths.size) * copynum_components
    params = finalise_params(params, components, haploid, smallest_copynum, len_gp_mode_copynum_ub, use_gamma)
    return fit(depths, mixture_model, params)

def setup_and_fit(depths, density, grid_min, kde_grid_density, min_density_depth_idx1, param_guesses, len_group_mode, mode_error, haploid, est_half):
    mode, sigma = param_guesses['mode'], param_guesses['sigma']
    depth_ECDF = ECDF(depths)
    density_at_modes, cdf_at_modes = setup_mode_densities_and_cdfs(mode, len_group_mode, depths, density, grid_min,
        kde_grid_density, min_density_depth_idx1, depth_ECDF, haploid)
    i, use_gamma = (len(density_at_modes) - 2) or 0.5, False
    if haploid:
        i = len(density_at_modes) - 1
    vary_copynum = int(i > 1)
    if (i > 1) and ((i + 1) * mode <= depths[-1]):
        copynums_in_90thpctl_mode_diff = (np.percentile(depths, 90) - len_group_mode) * 1.0 / mode
        gamma_min_density_ratio = get_gamma_min_density_ratio(copynums_in_90thpctl_mode_diff)
        gamma_min_cdf = get_gamma_min_cdf(depths, len_group_mode, copynums_in_90thpctl_mode_diff)
        density_ratio = density_at_modes[i] / density_at_modes[i-1]
        while (cdf_at_modes[i] < 0.95) and (density_ratio > 0.1):
            i += 1
            if (cdf_at_modes[i-1] > gamma_min_cdf) and (density_ratio > gamma_min_density_ratio):
                use_gamma, i = True, i - 1
                break
            density_at_modes[i] = get_density_for_idx(val_to_grid_idx(i * mode, kde_grid_density, grid_min), density)
            cdf_at_modes[i] = depth_ECDF([i * mode])[0]
            density_ratio = density_at_modes[i] / density_at_modes[i-1]
    lb = max(i - 1, 0) # i = 0 possible in principle for haploid genome
    if vary_copynum:
        lb = i - 2
        if use_gamma and (lb < 1):
            lb = 1
        if ((i + 1) * mode <= depths[-1]) and (depth_ECDF([(i + 1) * mode])[0] < 0.99):
            i += 1
            density_at_modes[i] = get_density_for_idx(val_to_grid_idx(i * mode, kde_grid_density, grid_min), density)
            cdf_at_modes[i] = depth_ECDF([i * mode])[0]
    density_at_modes, cdf_at_modes = pd.Series(density_at_modes), pd.Series(cdf_at_modes)
    len_gp_mode_copynum_ub = get_mode_copynum_ub(len_group_mode, mode, haploid, est_half)
    aic = np.inf
    for j in np.arange(i, lb, -1):
        # (i + 1) * mode <= depths[-1] practically guaranteed when gamma used
        next_mode_cdf = (use_gamma and (((j == i) and depth_ECDF([(j + 1) * mode])[0]) or cdf_at_modes[j+1])) or None
        component_weights = get_component_weights(density_at_modes[:j], min_density_depth_idx1, haploid, use_gamma, cdf_at_modes[:j], next_mode_cdf)
        smallest_copynum = 1
        if component_weights.iloc[1:][component_weights.iloc[1:] > 0].index[0] == 0.5:
            smallest_copynum = 0.5
        max_gaussian_copynums = int(depths[-1] > mode) * len(component_weights) - 1 - int(not(haploid)) - int(use_gamma)
        if not(haploid) and (max_gaussian_copynums < 1):
            max_gaussian_copynums = 0.5
        components, copynum_components, params = init_components_and_params(component_weights, param_guesses, mode,
            len_gp_mode_copynum_ub, smallest_copynum, max_gaussian_copynums, haploid)
        if use_gamma:
            components, copynum_components, params = init_gamma(depths, components, copynum_components, params, mode, sigma, mode_error)
        # Finally estimate copy numbers using lmfit
        result_temp = finalise_and_fit(components, copynum_components, params, haploid, smallest_copynum, len_gp_mode_copynum_ub, use_gamma, depths)
        if (result_temp.aic / aic) < 1 + min(0.025, 0.005 * (i - 3)): # slight penalty for higher # of components, increasing with i ("base" #)
            result, aic = result_temp, result_temp.aic
    return result

def get_smallest_copynum(prefixes):
    if ('gausshalf_' in prefixes) or ('cgausshalf_' in prefixes):
        return 0.5
    return 1

def get_mode_copynum_ub_from_prefixes(prefixes):
    numgauss_prefixes = { p for p in prefixes if re.search(r'^gauss(half|\d+)_', p) }
    sorted_gauss_copynums = sorted(map(lambda p: int(re.search(r'gauss(\d+)_', p).group(1)), numgauss_prefixes - {'gausshalf_'}))
    if sorted_gauss_copynums:
        return sorted_gauss_copynums[-1]
    return 0.5

def set_smallest_copynum_prefix(smallest_copynum, prefixes):
    if smallest_copynum == 0.5:
        return 'gausshalf_'
    return get_component_prefix(smallest_copynum, prefixes)

# Set mode the first time, i.e. from estimation and classification of longest sequences
def set_mode(mode_val, mode_error, smallest_copynum):
    mode = mode_val / smallest_copynum
    return (mode, (1 - mode_error) * mode, (1 + mode_error) * mode)

def set_sigma_min(sigma_result, smallest_copynum, mode_error):
    sigma_err = sigma_result.stderr or 0
    sigma_min = (sigma_result.value - sigma_err) / m.sqrt(smallest_copynum)
    if sigma_err == 0:
        return ((1 - mode_error) * sigma_min)
    return sigma_min

def assign_sequence_copynums(seqs, gp_len_condition, len_gp_idx, mode, copynum_assnmts, copynum_lbs, copynum_ubs):
    seqs.loc[gp_len_condition, 'modex'] = seqs.loc[gp_len_condition].mean_kmer_depth / mode
    seqs.loc[gp_len_condition, 'est_gp'] = len_gp_idx
    # Note: copynum_lbs[i] == copynum_ubs[i-1]
    if len(copynum_assnmts) > 1:
        seqs.loc[gp_len_condition & (seqs.mean_kmer_depth < copynum_lbs[copynum_assnmts[1]]), 'likeliest_copynum'] = copynum_assnmts[0]
    else:
        seqs.loc[gp_len_condition, 'likeliest_copynum'] = copynum_assnmts[0]
    for i in range(1, len(copynum_assnmts)):
        depth_condition = (seqs.mean_kmer_depth >= copynum_lbs[copynum_assnmts[i]]) & (seqs.mean_kmer_depth < copynum_ubs[copynum_assnmts[i]])
        seqs.loc[gp_len_condition & depth_condition, 'likeliest_copynum'] = copynum_assnmts[i]

def add_to_copynum_stats(data, cols, stats_hash):
    for i in range(len(data)):
        stats_hash[cols[i]].append(data[i])

def create_copynum_stats(smallest_copynum, copynums, include_half, params, component_prefixes, copynum_stats_hash):
    if 'exp_' in component_prefixes:
        add_to_copynum_stats(get_copynum_stats_data(0, params['exp_wt_c'].value, params['exp_decay'].value, params['exp_decay'].value),
            COPYNUM_STATS_COLS, copynum_stats_hash)
    wt, mean, sigma = 0, 0, 0
    wt_i, mean_i, sigma_i = 0, 0, 0
    mode_copynum_ub = get_mode_copynum_ub_from_prefixes(component_prefixes)
    if smallest_copynum < 1:
        wt, mean, sigma = get_component_params('gausshalf_', params)[:3]
    has_copynum1 = (smallest_copynum == 1) or next((True for p in component_prefixes if re.match('c?gauss1', p)), False)
    if has_copynum1:
        wt_i, mean_i, sigma_i = get_component_params(((mode_copynum_ub == 0.5) * 'c') + 'gauss1_', params)[:3]
    if include_half and (smallest_copynum < 1):
        add_to_copynum_stats(get_copynum_stats_data(0.5, wt, mean, sigma), COPYNUM_STATS_COLS, copynum_stats_hash)
        if has_copynum1:
            add_to_copynum_stats(get_copynum_stats_data(1, wt_i, mean_i, sigma_i), COPYNUM_STATS_COLS, copynum_stats_hash)
    else:
        wt1 = wt + wt_i
        wt_normed, wt_i_normed = wt/wt1, wt_i/wt1
        mean1 = (wt_normed * mean) + (wt_i_normed * mean_i)
        sigma1 = m.sqrt((wt_normed * sigma**2) + (wt_i_normed * sigma_i**2) + (wt_normed * wt_i_normed * (mean - mean_i)**2))
        add_to_copynum_stats(get_copynum_stats_data(1, wt1, mean1, sigma1), COPYNUM_STATS_COLS, copynum_stats_hash)
    for i in range(2, int(copynums[-1]) + 1):
        prefix = get_component_prefix(i, copynum_component_prefixes)
        wt, mean, sigma, loc, shape, scale = get_component_params(prefix, params)
        add_to_copynum_stats(get_copynum_stats_data(i, wt, mean, sigma, loc, shape, scale), COPYNUM_STATS_COLS, copynum_stats_hash)

def write_to_log(log_file, len_gp_idx, minlen, maxlen, maxdepth, depth_max_pctl, depth_max_pctl_rank, fit_report=None):
    log_file.write(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H:%M:%S : Sequence group ') + str(len_gp_idx) + ' estimated\n')
    log_file.write('Group minimum and maximum lengths: ' + str(minlen) + ', ' + str(maxlen) + '\n')
    log_file.write('Maximum mean k-mer depth of all sequences in group: ' + str(maxdepth) + '. ')
    if fit_report:
        log_file.write('Maximum used in estimation: ' + str(depth_max_pctl) + ' (' + str(depth_max_pctl_rank) + ' percentile).\n\n')
        log_file.write('Fit report:\n')
        log_file.write(fit_report)
    log_file.write('\n\n')


argparser = argparse.ArgumentParser(description='Estimate genomic copy number for haploid or diploid whole-genome shotgun assembly sequences')
argparser.add_argument('--haploid', action="store_true", help='Dataset comes from haploid rather than diploid genome')
argparser.add_argument('--half', action="store_true", help='Include copy number 0.5, i.e. heterozygous single-copy, in sequence classification')
argparser.add_argument('--per_unitig_mean_depth_given', action="store_true", help='Mean k-mer depth (instead of sum of depths) given for each sequence in unitigs_file')
argparser.add_argument('unitigs_file', type=str, help='FASTA file listing sequences to be classified')
argparser.add_argument('kmer_len', type=int, help='Value of k used in assembly that output sequences to be classified')
argparser.add_argument('output_dir', type=str, help='Directory to which output files should be written')
args = argparser.parse_args()

if args.haploid and args.half:
    raise ValueError('Classification of copy number 0.5 only valid with diploid genome dataset')

NONNEG_CONSTANT = 1.e-12

seqs = utils.seqs_from_abyss_contigs(args.unitigs_file, compute_mean_depth=not(args.per_unitig_mean_depth_given), k=args.kmer_len)
seqs.sort_values(by=['length', 'mean_kmer_depth'], inplace=True)

length_gps_for_est = utils.get_contig_length_gps(seqs, seqs.length)
length_gps_count = len(length_gps_for_est)
length_gp_medians = list(map(lambda gp: gp.length.median(), length_gps_for_est))

haploid_or_trivial = (args.haploid or (seqs.shape[0] == 1))
try_peak_as_half = not(haploid_or_trivial) and (length_gps_count == 1) # long sequences seem very unlikely to have a heterozygous peak
no_peak_as_half = not(try_peak_as_half)

LEN_GP_STATS_COLS = ['count', 'min_len', 'max_len', 'max_depth', 'max_depth_in_est', 'max_depth_pctl_rank_in_est', 'min_copynum', 'max_copynum_est']
len_gp_stats = [None] * no_peak_as_half
COPYNUM_STATS_COLS = ['len_gp_id', 'len_gp_min_len', 'len_gp_max_len', 'copynum', 'depth_lb', 'depth_max', 'weight', 'depth_mean', 'depth_stdev', 'depth_loc', 'depth_shape', 'depth_scale']
copynum_stats = [None] * no_peak_as_half

# Fit under assumption that first peak of density curve for longest sequences corresponds to mode of copy-number 0.5 or 1 (unique homozygous) sequences
mode_error = 0.1
aic = np.inf
better_fit_model = 1

log_file = open(args.output_dir + '/log.txt', 'w', newline='')

for longest_seqs_mode1_copynum in ([0.5] * int(try_peak_as_half) + [1.0]):
    if try_peak_as_half:
        log_header = 'ESTIMATION ROUND ' + str(longest_seqs_mode1_copynum * 2) + ': ASSUME 1ST PEAK OF DENSITY CURVE FOR LONGEST SEQUENCES CORRESPONDS TO MODE OF COPY-NUMBER '
        log_header += str(longest_seqs_mode1_copynum) + ' SEQUENCES\n'
        log_file.write(log_header)

    len_gp_stats.append(pd.DataFrame(data=None, index=pd.RangeIndex(stop=length_gps_count), columns=LEN_GP_STATS_COLS))
    length_gp_sigmas = [None] * length_gps_count
    copynum_stats_hash = { col: [] for col in COPYNUM_STATS_COLS }
    aic_current = 0
    mode, mode_min, mode_max, sigma_min = np.nan, 0, np.inf, 0

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

        min_density_depth_idx1 = np.argmin(density[:m.floor(val_to_grid_idx(np.percentile(depths, 25), kde_grid_density, grid_min))])
        depths = depths[depths <= depth_max_pctl]

        len_group_mode = get_len_group_mode(min_density_depth_idx1, density, grid_min, kde_grid_density)
        if np.isnan(mode):
            mode = len_group_mode
            mode /= longest_seqs_mode1_copynum # for consistency: let it represent c#1
        # Estimate standard deviation of copy-number 1 sequences
        if variance_from_curve(len_group_mode, mode, longest_seqs_mode1_copynum, mode_error):
            # Perhaps it's a mistake not to enforce that mode1_copynum be <= 2?
            # Post-hoc comment: mode * 0.25 likely used for len_group_mode == mode as well because it might help avoid over-estimating variance
            sigma = np.std(get_approx_component_obs(depths, len_group_mode, mode * 0.25, depths[0])) / (round(len_group_mode / mode) or m.sqrt(0.5))
        else:
            sigma = guess_next_sigma(length_gp_medians[len_gp_idx], length_gp_medians[(len_gp_idx + 1):], length_gp_sigmas[(len_gp_idx + 1):])
        if sigma < sigma_min:
            sigma_min = 0

        param_guesses = { 'mode': mode, 'mode_min': mode_min, 'mode_max': mode_max, 'sigma': sigma, 'sigma_min': sigma_min }
        if depths.size == 1: # only possible in (and with) a unique length stratum, because any stratum with such a small # obs will be absorbed into the next largest stratum
            len_gp_stats[-1].loc[len_gp_idx, 'min_copynum'] = 1
            len_gp_stats[-1].loc[len_gp_idx, 'max_copynum_est'] = 1
            # Would have to be copy number 1, not 0.5
            add_to_copynum_stats([0, curr_len_gp_stats.min_len, curr_len_gp_stats.max_len, 1, np.nan, np.nan, 1, depths[0], np.nan, None, None, None],
                COPYNUM_STATS_COLS, copynum_stats_hash)
            write_to_log(log_file, len_gp_idx, curr_len_gp_stats.min_len, curr_len_gp_stats.max_len, curr_len_gp_stats.max_depth,
                depth_max_pctl, depth_max_pctl_rank)
            seqs.loc[seqs.index[0], 'modex'] = 1
            seqs.loc[seqs.index[0], 'est_gp'] = 0
            seqs.loc[seqs.index[0], 'likeliest_copynum'] = 1
            break
        try:
            result = setup_and_fit(depths, density, grid_min, kde_grid_density, min_density_depth_idx1, param_guesses, len_group_mode, mode_error,
                args.haploid, args.half)
        except RuntimeError:
            error_msg = 'Lowest copy number for sequences of length ' + curr_len_gp_stats.min_len + ' to ' + curr_len_gp_stats.max_len + ' in dataset higher than 1: '
            error_msg += 'none are single-copy ' + int(not(args.haploid)) * ' (either homo- or heterozygous)!'
            raise RuntimeError(error_msg)
        write_to_log(log_file, len_gp_idx, curr_len_gp_stats.min_len, curr_len_gp_stats.max_len, curr_len_gp_stats.max_depth,
                     depth_max_pctl, depth_max_pctl_rank, result.fit_report())
        aic_current += result.aic
        copynum_component_prefixes = set(map(lambda name: re.search(r'(([a-zA-Z]+\d??)_)', name).group(), result.params.valuesdict().keys())) - { 'dummy_', 'genomescale_' }
        use_gamma = ('gamma_' in copynum_component_prefixes)
        smallest_copynum = get_smallest_copynum(copynum_component_prefixes)
        # Safe to use because error thrown whenever smallest copynum > 1, so first clause should always evaluate to > 0 for haploid genomes
        max_copynum_est = (len(copynum_component_prefixes) - int('exp_' in copynum_component_prefixes) - int(smallest_copynum == 0.5)) or 0.5

        smallest_copynum_prefix = set_smallest_copynum_prefix(smallest_copynum, copynum_component_prefixes)
        if mode_max == np.inf:
            mode, mode_min, mode_max = set_mode(result.params[smallest_copynum_prefix + 'center'].value, mode_error, smallest_copynum)
        sigma_min = set_sigma_min(result.params[smallest_copynum_prefix + 'sigma'], smallest_copynum, mode_error)
        length_gp_sigmas[len_gp_idx] = result.params[smallest_copynum_prefix + 'sigma'].value / smallest_copynum

        zero_or_imputed = smallest_copynum - 0.5 - 0.5 * args.haploid # Assumes smallest_copynum <= 1
        copynums = [zero_or_imputed, smallest_copynum]
        if max_copynum_est > 0.5:
            copynums = copynums + list(range(m.floor(smallest_copynum + 1), max_copynum_est + 1))
        depths_grid = np.array(depths)
        copynum_densities = pd.DataFrame(0.0, index = copynums, columns = depths_grid) # Density 0 needed for copy numbers not fitted
        cpnum_prefixes = pd.Series(dtype='object', index=copynums)
        for cpnum in copynums:
            if ((cpnum == 0) and ('exp_' in copynum_component_prefixes)) or (cpnum >= smallest_copynum):
                copynum_densities.loc[cpnum] = depths_grid
                cpnum_prefixes[cpnum] = get_component_prefix(cpnum, copynum_component_prefixes)
                copynum_densities.loc[cpnum] = copynum_densities.loc[cpnum].apply(lambda x: compute_density_at(x, cpnum_prefixes[cpnum], result.params))

        copynum_assnmts, copynum_lbs, copynum_ubs = utils.get_cpnums_and_bounds(copynum_densities, copynums)
        cn2 = next((cpnum for cpnum in copynums if cpnum > 1), None)
        if cn2:
            for cpnum in range(cn2, copynums[-1]):
                bd = copynum_ubs[cpnum]
                wt1, mean1, sigma1 = get_component_params(cpnum_prefixes[cpnum], result.params)[:3]
                wt2, mean2, sigma2 = get_component_params(cpnum_prefixes[cpnum+1], result.params)[:3]
                misclassified1, misclassified2 = stats.norm.sf(copynum_ubs[cpnum], mean1, sigma1), stats.norm.cdf(copynum_ubs[cpnum], mean2, sigma2)
                #if ((misclassified1 > 0.25) or (misclassified2 > 0.25)) and (cpnum + 1 < copynums[-1]):
                if (misclassified1 > 0.2) or (misclassified2 > 0.2):
                    copynum_densities.loc[cpnum, :] = copynum_densities.loc[cpnum:].sum()
                    copynum_densities = copynum_densities.loc[:cpnum]
                    max_copynum_est = cpnum
                    copynums = [zero_or_imputed, smallest_copynum] + list(range(m.floor(smallest_copynum + 1), max_copynum_est + 1))
                    copynum_assnmts, copynum_lbs, copynum_ubs = utils.get_cpnums_and_bounds(copynum_densities, copynums)
                    break

        len_gp_stats[-1].loc[len_gp_idx, 'min_copynum'] = smallest_copynum
        len_gp_stats[-1].loc[len_gp_idx, 'max_copynum_est'] = max_copynum_est

        if (smallest_copynum == 1) or ('exp_' not in copynum_component_prefixes):
            imputed = zero_or_imputed
            copynum_densities.loc[imputed] = depths_grid
            copynum_densities.loc[imputed] = copynum_densities.loc[imputed].apply(lambda x: get_density_for_idx(val_to_grid_idx(x, kde_grid_density, grid_min), density))
            copynum_densities.loc[imputed] = copynum_densities.loc[imputed] - copynum_densities.iloc[1:].sum()
            next_cpnum_wt, next_cpnum_mean, next_cpnum_sigma = get_component_params(smallest_copynum_prefix, result.params)[:3]
            # Heuristic to take into account component weight in estimating reasonable boundary location
            utils.impute_lowest_cpnum_and_bds(copynum_densities, imputed, copynum_assnmts, copynum_lbs, copynum_ubs, next_cpnum_mean - 2*next_cpnum_sigma - next_cpnum_wt)

        if max_copynum_est < 2:
            imputed = int(max_copynum_est + 1)
            copynum_densities.loc[imputed] = depths_grid
            copynum_densities.loc[imputed] = copynum_densities.loc[imputed].apply(lambda x: get_density_for_idx(val_to_grid_idx(x, kde_grid_density, grid_min), density))
            copynum_densities.loc[imputed] = copynum_densities.loc[imputed] - copynum_densities.loc[smallest_copynum:max_copynum_est].sum()
            max_copynum_prefix = get_component_prefix(max_copynum_est, copynum_component_prefixes)
            prev_cpnum_wt, prev_cpnum_mean, prev_cpnum_sigma = get_component_params(max_copynum_prefix, result.params)[:3]
            utils.impute_highest_cpnum_and_bds(copynum_densities, imputed, copynum_assnmts, copynum_lbs, copynum_ubs,
                prev_cpnum_mean + 2 * prev_cpnum_sigma + prev_cpnum_wt)
            len_gp_stats[-1].loc[len_gp_idx, 'max_copynum_est'] = imputed

        # Assign to sequences in the corresponding ranges
        gp_len_condition = (seqs.length >= curr_len_gp_stats.min_len) & (seqs.length <= curr_len_gp_stats.max_len)
        assign_sequence_copynums(seqs, gp_len_condition, len_gp_idx, mode, copynum_assnmts, copynum_lbs, copynum_ubs)
        valcounts = seqs.loc[gp_len_condition].likeliest_copynum.value_counts()

        def get_copynum_stats_data(idx, wt, mean, sigma, loc = None, shape = None, scale = None): # copy number idx
            return [len_gp_idx, curr_len_gp_stats.min_len, curr_len_gp_stats.max_len, idx, copynum_lbs[idx], copynum_ubs[idx], wt, mean, sigma, loc, shape, scale]

        create_copynum_stats(smallest_copynum, copynums, args.half, result.params, copynum_component_prefixes, copynum_stats_hash)
    # End inner loop across sequence length groups

    copynum_stats.append(pd.DataFrame.from_dict(copynum_stats_hash))
    if aic_current < aic:
        aic = aic_current
        better_fit_model = longest_seqs_mode1_copynum
    if try_peak_as_half:
        log_file.write('Sum of per-length-group model AICs: ' + str(aic_current) + '\n\n')
    print('')
    seq_label_filename = args.output_dir + '/sequence-labels.csv'
    if better_fit_model == longest_seqs_mode1_copynum:
        seqs.loc[:, 'length':].to_csv(seq_label_filename, header=['Length', 'Average k-mer depth', '1st Mode X', 'GC %', 'Estimation length group', 'Likeliest copy #'], index_label='ID')

if try_peak_as_half:
    log_footer = 'BETTER-FIT MODEL (LOWER SUM OF PER-LENGTH-GROUP MODEL AIC SCORES): 1ST PEAK OF DENSITY CURVE FOR LONGEST SEQUENCES CORRESPONDS TO MODE OF COPY-NUMBER '
    log_footer += str(better_fit_model) + ' SEQUENCES\n'
    log_file.write(log_footer)
log_file.close()

# Write length group and copy-number component stats
LEN_GP_STATS_OUTPUT_COLS = tuple(['count', 'min_len', 'max_len', 'max_depth', 'max_depth_in_est', 'min_copynum', 'max_copynum_est'])
LEN_GP_STATS_OUTPUT_HEADER = ['Number of sequences', 'Min. len.', 'Max. len.', 'Max. depth', 'Max. depth in estimation', 'Smallest copy # estimated', 'Largest copy # estimated']
len_gp_stats[m.floor(better_fit_model)].to_csv(args.output_dir + '/length_gp_stats.csv', columns=LEN_GP_STATS_OUTPUT_COLS, header=LEN_GP_STATS_OUTPUT_HEADER, index_label='ID')

COPYNUM_STATS_OUTPUT_HEADER = ['Group #', 'Group min. len.', 'Group max. len.', 'Component #', 'Component depth lower bound', 'Component max. depth',
                               'Weight', 'Mean', 'Std. deviation', 'Location', 'Shape', 'Scale']
copynum_stats[m.floor(better_fit_model)].to_csv(args.output_dir + '/copynumber_params.csv', header=COPYNUM_STATS_OUTPUT_HEADER, index=False)

