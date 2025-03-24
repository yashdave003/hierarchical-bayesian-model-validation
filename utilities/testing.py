import numpy as np
import pandas as pd
import scipy
from scipy import integrate, interpolate, stats, special
from pathlib import Path
import pickle
import os
import pywt.data
from PIL import Image
import warnings
import scipy.special
from tqdm import tqdm
import git 
np.set_printoptions(legacy='1.25')

USE_MATLAB=True
if USE_MATLAB:
    import matlab.engine 
    eng = matlab.engine.connect_matlab()
else:
    eng = None
    
def get_project_root():
    return Path(git.Repo('.', search_parent_directories=True).working_tree_dir)

def compute_cdf_vals(r, beta, scale, xs, use_matlab = True, debug = False):
    prior_cdf = np.zeros_like(xs)
    if use_matlab:
        if debug:
            for j in tqdm(range(len(xs))):
                if scale == 1:
                    prior_cdf[j] = eng.compute_cdf_using_gengamma(float(r), float(beta), float(xs[j]), nargout=1)
                else:
                    prior_cdf[j] = eng.compute_cdf_using_gengamma_with_scale(float(r), float(beta), float(scale), float(xs[j]), nargout=1)
        else:
            for j, x in enumerate(xs):
                if scale == 1:
                    prior_cdf[j] = eng.compute_cdf_using_gengamma(float(r), float(beta), float(xs[j]), nargout=1)
                else:
                    prior_cdf[j] = eng.compute_cdf_using_gengamma_with_scale(float(r), float(beta), float(scale), float(xs[j]), nargout=1)
    else:
        def gauss_density(z, x):
            return np.exp(-0.5 * (x/z)**2) / (np.sqrt(2*np.pi) * z)

        def gen_gamma_cdf(x):
            return prior_cdf.gammainc(beta, (x/scale)**r)

        def integrand(z, x):
            return gauss_density(z, x) * (1 - gen_gamma_cdf((x/z)**2))
        
        for j, x in enumerate(xs):
            res = integrate.quad(integrand, 0, np.inf, args=(x,))[0]
            prior_cdf[j] = res
    return prior_cdf

def compute_prior_pdf(r, eta, method='gamma_cdf', n_samples = 1000, tail_bound = 0.001, tail_percent = 0.1, scale = 1, use_matlab=True, eng=None, debug=False, enforce_assert=True, return_assert=False):
    
    if method == 'gamma_cdf':
        xs, cdf = compute_prior_cdf(r=r, eta=eta, method='gamma_cdf', n_samples=n_samples, tail_bound=tail_bound, tail_percent=tail_percent, scale=scale, use_matlab=use_matlab, eng=eng, enforce_assert=enforce_assert, return_assert=return_assert, return_xs=True, debug=debug)
        return xs, cdf.derivative()
    elif method == 'normal_cdf':
        xs, cdf = compute_prior_cdf_using_normal_cdf(r=r, eta=eta, n_samples=n_samples, tail_bound=tail_bound, tail_percent=tail_percent, scale=scale, use_matlab=use_matlab, eng=eng, enforce_assert=enforce_assert, return_assert=return_assert, return_xs=True)
        return xs, cdf.derivative()
    
def compute_prior_cdf(r, eta, method='gamma_cdf', n_samples = 1000, tail_bound = 0.001, tail_percent = 0.1, scale = 1, use_matlab=True, eng=eng, debug=True, enforce_assert=True, return_assert=False, return_xs=False):

    if method == 'gamma_cdf':
        return compute_prior_cdf_using_gamma_cdf(r=r, eta=eta, n_samples=n_samples, tail_bound=tail_bound, tail_percent=tail_percent, scale=scale, use_matlab=use_matlab, eng=eng, enforce_assert=enforce_assert, return_assert=return_assert, return_xs=return_xs, debug=debug)
    elif method == 'normal_cdf':
        return compute_prior_cdf_using_normal_cdf(r=r, eta=eta, n_samples=n_samples, tail_bound=tail_bound, tail_percent=tail_percent, scale=scale, use_matlab=use_matlab, eng=eng, enforce_assert=enforce_assert, return_assert=return_assert, return_xs=return_xs, debug=debug)
    else:
        print("Not a valid method, valid options are: gamma_cdf, normal_cdf, numerical_old")
       
def compute_prior_cdf_using_gamma_cdf(r, eta, n_samples=1000, tail_bound=0.001, tail_percent=0.1, scale=1, use_matlab=True, eng=eng, enforce_assert=True, return_assert=False, return_xs=False, debug=False):
    beta = (eta + 1.5) / r
    var_prior = scale * special.gamma(beta + 1/r) / special.gamma(beta)
    
    cheby = np.sqrt(var_prior / tail_bound)
    if np.isnan(var_prior) or np.isinf(var_prior):
        cheby = 1e100
    x_max = min(99, cheby)
    n_tail = int(n_samples * tail_percent) if cheby >= 120 else 0

    if debug:
        print(f"Params: {r}, {eta}")
        print(f"Chebyshev bound: {cheby}")
        print(f"{'No tail' if n_tail == 0 else f'Tail samples: {n_tail}'}")

    xs_minus = np.concatenate((-np.logspace(np.log10(cheby), 2, n_tail),
                               np.linspace(-x_max, 0, n_samples//2-n_tail)))

    prior_cdf_minus = compute_cdf_vals(r, beta, scale=scale, xs = xs_minus, use_matlab = use_matlab, debug = debug)

    if debug:
        print("Maximum Diff in y-values:", max(abs(np.diff(prior_cdf_minus))))
    diff_cutoff = 0.02
    if(max(abs(np.diff(prior_cdf_minus))) > diff_cutoff and eta < 0):
        
        while max(abs(np.diff(prior_cdf_minus))) > diff_cutoff:
            filtered_y, idx = np.unique(prior_cdf_minus, return_index = True)
            filtered_x = xs_minus[idx]
            idx2 = filtered_y > 1e-15
            filtered_y = filtered_y[idx2]
            filtered_x = filtered_x[idx2]

            inv_linspline = interpolate.InterpolatedUnivariateSpline(x= filtered_y, y= filtered_x, k=1, ext='const')
            xs_add = inv_linspline(np.linspace(0, 0.5, n_samples//2))
            
            prior_cdf_add = compute_cdf_vals(r, beta, scale=scale, xs = xs_add, use_matlab = use_matlab, debug = debug)

            xs_minus, idx_merge = np.unique(np.append(xs_minus, xs_add), return_index = True)
            prior_cdf_minus = np.sort(np.append(prior_cdf_minus, prior_cdf_add)[idx_merge])
            if debug:
                print("Maximum Diff in y-values with new points:", max(abs(np.diff(prior_cdf_minus))))

    xs_plus = -np.flip(xs_minus[:-1])
    prior_cdf_plus = 1-np.flip(prior_cdf_minus[:-1])

    xs = np.concatenate((xs_minus, xs_plus))
    prior_cdf = np.concatenate((prior_cdf_minus, prior_cdf_plus))

    if debug:
        print(f"First CDF value: {prior_cdf[0]}")
        print(f"Last CDF value: {prior_cdf[-1]}")
        print(f"Tail bound: {tail_bound}")

    if return_assert or enforce_assert:
        eps = tail_bound
        if not (-eps < prior_cdf[0] < eps and 1 - eps < prior_cdf[-1] < 1 + eps):
            if return_assert:
                return (xs, None) if return_xs else None
            elif enforce_assert:
                raise AssertionError("CDF bounds not satisfied")
        
    xs = np.concatenate(([1.01 * xs[0]], xs, [1.01 * xs[-1]]))
    prior_cdf = np.concatenate(([0], prior_cdf, [1]))
    cdf_spline = interpolate.InterpolatedUnivariateSpline(x=xs, y=prior_cdf, k=3, ext='const')

    if return_assert or enforce_assert:
        x = np.sort(sample_prior(r, eta, 10000, scale=scale))
        res = stats.ks_1samp(x, cdf_spline)
        if debug:
            print(res)
        if not 0 <= res.statistic <= 0.2:
            if return_assert:
                return (xs, None) if return_xs else None
            elif enforce_assert:
                raise AssertionError("KS test failed")

    return (xs, cdf_spline) if return_xs else cdf_spline

def compute_prior_cdf_using_normal_cdf(r, eta, n_samples = 2000, tail_bound = 0.01, tail_percent = 0.1, scale = 1, use_matlab=False, eng= None, enforce_assert=True, return_assert = False, return_xs=False, debug=False):
    
    beta = (eta + 1.5)/r 
    var_prior = scale * scipy.special.gamma(beta + 1/r)/scipy.special.gamma(beta)
    cheby = np.sqrt(var_prior/(tail_bound))
    n_tail = int(n_samples*tail_percent)
    
    x_max = min(99, cheby) 
    if cheby < 120:
        n_tail = 0
        if debug:
            print(f"No tail")
    if debug:
        print(f"Chebyshev bound: {cheby}")

    xs = np.linspace(-x_max, x_max, n_samples-2*n_tail)
    xs = np.append(-np.logspace(np.log10(cheby), 2, n_tail), xs)
    xs = np.append(xs, np.logspace(2, np.log10(cheby), n_tail))

    prior_cdf = np.full(xs.shape, np.nan)

    for j in tqdm(range(len(xs))):

        x = xs[j]
        def gen_gamma_density(theta):
            return (np.abs(r)/scipy.special.gamma(beta)) * (1/scale) * (theta/scale)**(r*beta - 1) * np.exp(-(theta/scale)**r)

        def integrand(theta):
            return stats.norm.cdf(x/np.sqrt(theta)) * gen_gamma_density(theta)

        if use_matlab:
            prior_cdf[j] = eng.compute_cdf_using_phi(float(r), float(eta), float(x), nargout=1)
        else:
            prior_cdf[j] = integrate.quad(integrand, 0, np.inf)[0]
    
    normalizer = prior_cdf[-1]
    first = prior_cdf[1]

    if debug:
        print("First CDF value:", first)
        print("Last CDF value:", normalizer)

    eps = 0.01
    if return_assert:
        if not -eps < first < eps:
            return None
        if not 1 - eps < normalizer < 1 + eps:
            return None    

    if enforce_assert:
        assert -eps < first < eps    
        assert 1 - eps < normalizer < 1 + eps
    
    prior_cdf = prior_cdf/normalizer   

    k = int(0.01*len(xs))
    zero_padding = np.zeros(k)
    ones_padding = np.ones(k)

    pad_max = max(10e5, np.round(max(np.abs(xs)) ** 2))
    if debug:
        print(f"0, 1 padding bounds: {pad_max}")

    prior_cdf_padded = np.concatenate([zero_padding, prior_cdf, ones_padding])
    xs_padded = np.concatenate([
        np.linspace(-pad_max, xs[0] - 1e-5, k),
        xs,
        np.linspace(xs[-1] + 1e-5, pad_max, k)
    ])

    cdf_spline = interpolate.CubicSpline(x=xs_padded, y=prior_cdf_padded)

    if enforce_assert:
        x = np.sort(sample_prior(r, eta, 100000))
        res = stats.ks_1samp(x, cdf_spline)
        if debug:
            print(res)
        assert 0 <= res.statistic <= .1
        if res.pvalue < 0.01:
            assert np.abs(res.statistic_location) > cheby

    if return_assert:
        x = np.sort(sample_prior(r, eta, 100000))
        res = stats.ks_1samp(x, cdf_spline)
        if debug:
            print(res)
        if not 0 <= res.statistic <= .1:
            return None
        
    
    if return_xs:
        return xs, cdf_spline
    else:
        return cdf_spline

def sample_prior(r, eta, size=1, scale=1):
    '''
    Samples from prior distribution of signal x
    r : shape parameter, must be nonzero
    eta : shape parameter, controls roundedness of peak, must be picked such that beta=(1.5+eta)/r > 0
    size : integer specifying number of samples required

    Note: Theta ~ GenGamma is modeled as the variance of the Normal, scale takes in the standard deviation. 
    This matches up with the original paper on "Sparse Reconstructions ..." by Calvetti et. al. 2020
    '''
    beta = (eta + 1.5)/r
    assert beta > 0
    vars = stats.gengamma.rvs(a = beta, c = r, scale=scale, size = size)
    x = np.random.normal(scale = np.sqrt(vars), size=size)
    return x

def round_to_sigfigs(x, num_sigfigs=8):
    if x == np.zeros_like(x):
        return 0
    return np.round(x, -int(np.floor(np.log10(abs(x)))-(num_sigfigs-1)))

def kstest_custom(x, cdf, return_loc = False):
    n = len(x)
    x = np.sort(x)
    cdfvals = cdf(x)
    dplus, dminus = (np.arange(1.0, n + 1) / n - cdfvals), (cdfvals - np.arange(0.0, n)/n)
    plus_amax, minus_amax = dplus.argmax(), dminus.argmax()
    loc_max, loc_min = x[plus_amax], x[minus_amax]
    d = max(dplus[plus_amax], dminus[minus_amax])
    if return_loc:
        if d == plus_amax:
            return d, stats.kstwo.sf(d, n), loc_max
        else:
            return d, stats.kstwo.sf(d, n), loc_min
    return d, stats.kstwo.sf(d, n)

def combine_pickles(dir):
    pickles = os.listdir(dir)
    cdfs = dict()
    for pkl in pickles:
        pkl_path = os.path.join(dir, pkl)
        with open(pkl_path, 'rb') as handle:
            new_cdf = pickle.load(handle)
        if type(new_cdf) == dict:
            cdfs = cdfs | new_cdf
    return cdfs

def compute_ksstat(sample, cdf, sorted_sample = True):
    '''
    Computes the KS-Test Statistic, assumes that the sample is already sorted
    '''
    if not sorted_sample:
        sample = np.sort(sample)

    if isinstance(cdf, tuple):
        r = cdf[0]
        eta = cdf[1]
        cdf = compute_prior_cdf(r, eta)
    
    n = len(sample)
    cdfvals = cdf(sample)
    dplus, dminus = (np.arange(1.0, n + 1) / n - cdfvals), (cdfvals - np.arange(0.0, n)/n)
    return np.max(np.append(dplus, dminus))

def compute_ksstat_tail(sample, cdf, sorted_sample = True, tail_cutoff = 2):
    '''
    Computes the KS-Test Statistic, assumes that the sample is already sorted
    '''
    if not sorted_sample:
        sample = np.sort(sample)

    tail_idxs = np.argwhere(np.abs(sample) > tail_cutoff)
    tails = sample[tail_idxs]

    if isinstance(cdf, tuple):
        r = cdf[0]
        eta = cdf[1]
        cdf = compute_prior_cdf(r, eta)
    
    n = len(sample)
    cdfvals = cdf(sample)
    dplus, dminus = (np.arange(1.0, n + 1) / n - cdfvals), (cdfvals - np.arange(0.0, n)/n)
    dplus_t, dminus_t = dplus[tail_idxs], dminus[tail_idxs]
    return np.max(np.append(dplus_t, dminus_t))

def compute_ksratio(sample, cdf, sorted_sample = True, tail_cutoff = 0):
    '''
    Computes the ratio of empirical and computed cdfs, assumes that the sample is already sorted
    '''
    if not sorted_sample:
        sample = np.sort(sample)
    tail_idxs = np.argwhere((np.abs(sample) > tail_cutoff))
    tails = sample[tail_idxs]

    if isinstance(cdf, tuple):
        r = cdf[0]
        eta = cdf[1]
        cdf = compute_prior_cdf(r, eta)
    
    n = len(sample)
    tail_vals = cdf(tails)
    d = (np.arange(1.0, n + 1) / n)
    tail_ratios = np.nan_to_num(d[tail_idxs] / tail_vals)
    # empirical / computed
    return (round_to_sigfigs(np.min(tail_ratios)), round_to_sigfigs(np.max(tail_ratios)))


def gridsearch(sample, all_cdfs, top_k = 1, debug = False, scales = None):
    '''
    Takes in a sample and list of CDFs, 
    Returns the KS-Test Statistic computed with respect to each CDF, the top-k minimizing parameters and the corresponding distances
    '''
    cdf_keys = sorted(all_cdfs)
    cdf_splines = [all_cdfs[key] for key in cdf_keys]
    num_cdfs = len(cdf_keys)
    ksstats = np.zeros(num_cdfs)
    empirical_var = np.var(sample)
    
    if debug:
        loop = tqdm(range(num_cdfs))
    else:
        loop = range(num_cdfs)
    for i in loop:
        if scales is not None:
            r, eta = cdf_keys[i]
            scale = scales[i]
            ksstats[i] = compute_ksstat(sample / np.sqrt(scale), cdf_splines[i])
        else:
            ksstats[i] = compute_ksstat(sample, cdf_splines[i])
    
    min_k = 2*np.ones(top_k).astype(int)
    if debug:
        print(f"Finding Minimum after computing {num_cdfs} CDFs")
    if top_k > 1:
        ksstats_copy = ksstats.copy()
        for i in np.arange(top_k):
            min_k[i] = np.argmin(ksstats_copy)
            ksstats_copy[min_k[i]] = 2
        return ksstats, [cdf_keys[j] for j in min_k], ksstats[min_k]
    else:
        return ksstats, cdf_keys[np.argmin(ksstats)], np.min(ksstats) 


def add_cdfs(r_range, eta_range, n_samples, use_matlab=False, folder_name='', debug = False, eng=None, enforce_assert=True, return_assert = False):
    '''
    folder_name: Name of directory that contains pickles of dictionaries of cdfs
    r_range: range of r values, assumes use of np.arange
    eta_range: range of eta values, assumes use of np.arange
    check_redundant: if True, checks if key already exists in dictionary
    n_samples: number of samples used when computing prior_cdf
    '''
    
    if not os.path.isdir("CDFs"):
        raise Exception("This Directory Does Not Contain CDFs")
    
    if folder_name == '':
        folder_name = f'r{min(r_range)}-{max(r_range)}_eta{min(eta_range)}-{max(eta_range)}_{n_samples}'

    FOLDER_PATH = os.path.join("CDFs", folder_name)
    cdfs_completed = set()
    if os.path.isdir(FOLDER_PATH):
        print(FOLDER_PATH)    
        for pkl in os.listdir(FOLDER_PATH):
            with open(os.path.join(FOLDER_PATH, pkl), 'rb') as handle:
                next_cdf = pickle.load(handle)
            cdfs_completed.update(next_cdf.keys())
    else:
        Path(os.path.join(os.getcwd(), FOLDER_PATH)).mkdir()
    if debug:
        print("CDFs completed:", len(cdfs_completed))
    n = len(r_range)*len(eta_range)

    if len(cdfs_completed) == n:
        if debug:
            print("Already computed")
        return
    
    cnt = 0
    grouped_r_cdf = dict()
    flag = False
    cut_max_eta = float('inf')
    for r in r_range:
        r_cdf = dict()
        r = round_to_sigfigs(r, 6)
        for eta in eta_range:
            eta = round_to_sigfigs(eta, 6)
            if ((r, eta) in cdfs_completed):
                continue
            cnt += 1
            if debug:
                print(f'{(r, eta)}, {cnt} of {n}')
            computed_cdf = compute_prior_cdf(r = r, eta = eta, method = 'gengamma', n_samples = n_samples, tail_percent = 0.1, tail_bound = 0.01, use_matlab=use_matlab, eng=eng, enforce_assert=enforce_assert, return_assert=return_assert)
            if computed_cdf is None:
                with open("faultyCDFs.csv", 'a') as handle:
                    handle.write(f"{r},{eta},{n_samples}\n")
                with open("faultyCDF_log.txt", 'a') as handle:
                    handle.write(f"Failed assert for r={r}, eta={eta}, n_samples={n_samples}")
                    handle.write(f"Skipping {eta} (exclusive) to {max(eta_range)} for r={r}")
                print(f"Failed assert for r={r}, eta={eta}, n_samples={n_samples}\n")
                print(f"Skipping {eta} (exclusive) to {max(eta_range)} for r={r}\n")
                r_cdf[(r, eta)] = computed_cdf
                cut_max_eta = eta
                break
            r_cdf[(r, eta)] = computed_cdf

        # Store pickle every outer loop iteration as its own file
        # CDFs/<optional_folder_name><number of samples>/<r>_<min(eta)>-<max(eta)>.pickle
        min_eta, max_eta = round_to_sigfigs(eta_range[0], 6), min(round_to_sigfigs(eta_range[-1], 6), cut_max_eta)
        
        if len(eta_range) > 1:
            pkl_path = os.path.join(FOLDER_PATH,f'{r}_{min_eta}-{max_eta}.pickle')
            dump_dict_pkl(r_cdf, pkl_path, overwrite=False)
        else:
            grouped_r_cdf = grouped_r_cdf | r_cdf
            flag = True
    if flag:
        pkl_path = os.path.join(FOLDER_PATH, f'{round_to_sigfigs(r_range[0], 6)}-{round_to_sigfigs(r_range[-1], 6)}_{min_eta}.pickle')
        dump_dict_pkl(grouped_r_cdf, pkl_path, overwrite=False)

    if debug:
        print(f'You can find the CDFs here: {os.path.join(os.getcwd(), FOLDER_PATH)}')

def load_pkl(path):
    if os.path.isfile(path):
        with open(path, 'rb') as handle:
            obj = pickle.load(handle)
        return obj
    else:
        raise Exception("File does not exist, check the path again")
    
def dump_dict_pkl(obj, path, overwrite = False, debug=False):
    if overwrite:
        if debug:
            print("Overwriting existing file if it exists")
        with open(path, 'wb') as handle:
            pickle.dump(obj, handle)
    elif os.path.isfile(path):
        if debug:
            print("Appending to Existing File")
        with open(path, 'rb') as handle:
            existing_object = pickle.load(handle)
        obj = obj | existing_object
    else:
        if debug:
            print("Writing to new file")
        with open(path, 'wb') as handle:
            pickle.dump(obj, handle)

def find_n_fixed_pval_stat(ksstat: float, n: int, cutoff=0.05):
    """
    Finds the sample size 'n' required to achieve a target p-value 'cutoff' for a given Kolmogorov-Smirnov (KS) statistic 'ksstat'.

    Args:
        ksstat (float): The Kolmogorov-Smirnov statistic value.
        n (int): The initial sample size to start the search.
        cutoff (float, optional): The target p-value to achieve. Defaults to 0.05.

    Returns:
        int: The sample size 'n' required to achieve the target p-value 'cutoff' for the given 'ksstat'.

    Note:
        This function assumes the availability of the 'kstwo' function from a specific library (e.g., scipy.stats) 
        to calculate the survival function (sf) of the Kolmogorov-Smirnov distribution.
    """
    curr_pval = scipy.stats.kstwo(n).sf(ksstat)
    while not np.isclose(curr_pval, cutoff, atol=0.01):
        if np.isnan(curr_pval):
            print(f"Cannot compute pval with ksstat : {ksstat}, n : {n}")
            return -1
        if curr_pval < cutoff:
            n = int(n / 2)
            curr_pval = scipy.stats.kstwo(n).sf(ksstat)
        elif curr_pval > cutoff:
            n = int(n * 1.5)
            curr_pval = scipy.stats.kstwo(n).sf(ksstat)
    return n

def coord_descent_gengamma(sample, initial_param, r_depth, eta_depth, group, completed_r_depth = 1, completed_eta_depth = 1, debug = True, DATA_NAME = None, eng=None, use_matlab = False):
    '''
    Perform coordinate descent optimization to find the best parameters (r, eta) for a generalized gamma distribution
    that minimizes the Kolmogorov-Smirnov (KS) statistic for the given `sample`.
    
    Args:
       sample (numpy.ndarray): The sample data for which the KS statistic is computed.
       initial_param (tuple): The initial guess for the parameters (r, eta).
       r_depth (int): The number of decimal places to search for the optimal value of 'r'.
       eta_depth (int): The number of decimal places to search for the optimal value of 'eta'.
       group (int): The layer index for naming the intermediate pickles.
       completed_r_depth (int, optional): The number of decimal places already completed for 'r'. Defaults to 1.
       completed_eta_depth (int, optional): The number of decimal places already completed for 'eta'. Defaults to 1.
       
    Returns:
       tuple: The optimal values of (r, eta) that minimize the KS statistic for the given `sample`.

    Example:
    `coord_descent_gengamma(obs_x_dict[4], (0.8, 3), 3, 2, 4)` will search through
    r = range(0.70, 0.90, 0.01), eta = 3. Suppose best value is 0.80 (2 decimals)
    r = range(0.780, 0.800, 0.001), eta = 3. Suppose best value is r=0.803 (3 decimals)
    Then
    r = 0.803, eta = range(2.9, 3.1, 0.01). Suppose best value is eta=3.01 (2 decimals)

    returns 0.803, 3.01
    '''
    r_0, eta_0 = initial_param
    n_samples = 10000

    for d in np.arange(completed_r_depth, r_depth):
        if debug:
            print(f"Optimizing r, current depth {d} of {r_depth}, r = {r_0}")
        r_range = np.arange(r_0 - 10.0**(-d), r_0 + 10.0**(-d), 10.0**(-d-1)) 
        eta_range = [eta_0]
        add_cdfs(r_range=r_range, eta_range=eta_range, n_samples=n_samples, folder_name=f'{DATA_NAME}_group{group}_{n_samples}', eng=eng, use_matlab=False)
        layer_cdfs = combine_pickles(f'{DATA_NAME}_group{group}_{n_samples}')
        ksstats, best_param, min_stat = gridsearch(sample, layer_cdfs)
        r_0 = round_to_sigfigs(best_param[0], d+1)

    for d in np.arange(completed_eta_depth, eta_depth):
        if debug:
            print(f"Optimizing eta, current depth {d} of {eta_depth}, eta = {eta_0}")
        r_range = [r_0]
        eta_range = np.arange(max(eta_0 - 10.0**(-d), 0), eta_0 + 10.0**(-d), 10.0**(-d-1)) 
        add_cdfs(r_range=r_range,eta_range=eta_range, n_samples=n_samples, folder_name=f'{DATA_NAME}_group{group}_{n_samples}', eng=eng, use_matlab=False)
        layer_cdfs = combine_pickles(f'{DATA_NAME}_group{group}_{n_samples}')
        ksstats, best_param, min_stat = gridsearch(sample, layer_cdfs)
        eta_0 = round_to_sigfigs(best_param[1], d+1)

    return (r_0, eta_0)

def change_params_power(r, eta, k):
    r_new = r/k
    eta_new = (eta+1.5)/k - 1.5
    return r_new, eta_new

def variance_prior(r, eta, scale=1):
    beta = (eta+1.5)/r
    var_prior = scale * scipy.special.gamma(beta + 1/r)/scipy.special.gamma(beta)
    return var_prior

def MAD_prior(r, eta, scale = 1):
    beta = (eta+1.5)/r
    return np.sqrt(2/np.pi) * scale ** (1/2) * scipy.special.gamma(beta + 0.5/r)/scipy.special.gamma(beta)

def get_rescale_val(r, eta, sample_var):
    return sample_var/variance_prior(r, eta, scale = 1)

def kurtosis_prior(r, eta, scale=1, fisher=True):
    beta = (eta+1.5)/r
    kurtosis = scale*3*scipy.special.gamma(beta + 2/r)*scipy.special.gamma(beta)/scipy.special.gamma(beta+1/r)**2 
    if fisher:
        return kurtosis - 3
    else:
        return kurtosis 
    
def bootstrap_metric(x, metric=None, n_bootstrap=1000, bootstrap_size = 10000, ci=0.99, replace=True):
    metric_values = []
    for _ in tqdm(range(n_bootstrap)):
        resampled = np.random.choice(x, size=bootstrap_size, replace=replace)
        metric_values.append(metric(resampled))
        
    metric_point_estimate = metric(x)
    ci_lower = np.percentile(metric_values, (1 - ci) / 2 * 100)
    ci_upper = np.percentile(metric_values, (1 + ci) / 2 * 100)
    
    return metric_point_estimate, ci_lower, ci_upper, metric_values

def gen_gamma_mean(r, eta):
    return scipy.special.gamma((eta+2.5)/r) / scipy.special.gamma((eta+1.5)/r)

def gen_gamma_variance(r, eta):
    mean = gen_gamma_mean(r, eta)
    second_moment = scipy.special.gamma((eta+3.5)/r) / scipy.special.gamma((eta+1.5)/r)
    return second_moment - mean**2

def find_eta_for_target_mean(r, target_mean):
    def objective(eta):
        return (gen_gamma_mean(r, eta) - target_mean)**2
    result = scipy.optimize.minimize_scalar(objective)
    return result.x

def create_kurt_var_ksstat_df(cdf_dict):
    cdfs_df = pd.DataFrame({'(r,eta),cdf' : sorted(cdf_dict.items())})
    cdfs_df['r'] = pd.Series(cdfs_df["(r,eta),cdf"].str[0].str[0])
    cdfs_df['eta'] = pd.Series(cdfs_df["(r,eta),cdf"].str[0].str[1])
    cdfs_df['cdf'] = pd.Series(cdfs_df["(r,eta),cdf"].str[1])
    cdfs_df['variance'] = np.nan_to_num(cdfs_df.apply(lambda row : variance_prior(row.loc['r'], row.loc['eta']), axis = 1))
    cdfs_df['kurtosis'] = cdfs_df.apply(lambda row : kurtosis_prior(row.loc['r'], row.loc['eta']), axis = 1)
    # cdfs_df['MAD'] = cdfs_df.apply(lambda row : MAD_prior(row.loc['r'], row.loc['eta']), axis = 1)
    # cdfs_df['MAD'] = cdfs_df['MAD'].fillna(0)
    return cdfs_df

def add_tests_to_df(cdfs_df, group, var_kurt_df, ksstats):

    cdfs_df['pass_var'] = (cdfs_df['variance'] > var_kurt_df.loc[group, 'var_lower']) & (cdfs_df['variance'] < var_kurt_df.loc[group, 'var_upper'])
    cdfs_df['pass_kurt'] = (cdfs_df['kurtosis'] > var_kurt_df.loc[group, 'kurt_lower']) & (cdfs_df['kurtosis'] < var_kurt_df.loc[group,'kurt_upper'])
    cdfs_df['ksstat'] = ksstats
    cutoff = stats.kstwo(n=var_kurt_df.loc[group, 'total_samples']).isf(0.05)
    cdfs_df['pass_kstest'] = cdfs_df['ksstat'].apply(lambda x: True if x < cutoff else False)
    return cdfs_df

def remove_directory(directory_path):
    for root, dirs, files in os.walk(directory_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(directory_path)