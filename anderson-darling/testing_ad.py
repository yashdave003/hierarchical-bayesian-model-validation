import scipy
import numpy as np

def compute_adstat(x, F, true_n=None, sorted_sample=True, eps=1e-16):
    '''
    x.shape = (sample_size, ...) 
    F = cdf of the null distribution
    true_n = size of the full sample, if the dataset passed in is a subsample
    '''
    if not sorted_sample:
        x = np.sort(x, axis=0)
    z = np.clip(F(x), eps, 1 - eps)

    n = x.shape[0]
    i = np.arange(1, 2 * n, 2).reshape((n,) + (1,) * (len(x.shape) - 1))
    S = np.mean(i * (np.log(z) + np.log1p(-z[::-1])), axis=0)
    rescale_factor = 1 if true_n is None else true_n / n
    return rescale_factor * (-n - S)

@np.vectorize
def ad_asymptotic_cdf(z, N=10):
    '''
    asymptotic cdf of A^2 under H_0 at z using first N terms in expansion
    https://doi.org/10.1214/aoms/1177729437 (pp. 204)
    '''
    if z > 25:
        return 1 - ad_asymptotic_surv(z, N)

    @np.vectorize
    def term(j):
        t_j = (4*j+1)**2 * np.pi**2 / (8*z)
        integrand = lambda w: np.exp(z / (8*(1+w**2)) - w**2 * t_j)
        integral = scipy.integrate.quad(integrand, 0, np.inf)[0]
        return integral * scipy.special.binom(-.5, j) * (4*j+1) * np.exp(-t_j)
    return np.sqrt(2 * np.pi) / z * np.sum(term(np.arange(N)))

@np.vectorize
def ad_asymptotic_surv(z, N=10):
    if z > 25:
        y, dy = -26.639351139190342, -1.0192986056281121
        return np.exp(y + dy * (z - 25))
    return 1 - ad_asymptotic_cdf(z, N)


def gridsearch_ad(sample, all_cdfs, true_n=None, top_k=1, debug=False, scales=None):
    '''
    Takes in a sample and list of CDFs,
    Returns the AD statistic computed with respect to each CDF, the top-k minimizing parameters and the corresponding distances
    '''
    cdf_keys = sorted(all_cdfs)
    cdf_splines = [all_cdfs[key] for key in cdf_keys]
    num_cdfs = len(cdf_keys)
    adstats = np.zeros(num_cdfs)

    if debug:
        from tqdm.notebook import tqdm
        loop = tqdm(range(num_cdfs))
    else:
        loop = range(num_cdfs)
    for i in loop:
        if scales is not None:
            scale = scales[i]
            adstats[i] = compute_adstat(sample / np.sqrt(scale), cdf_splines[i], true_n=true_n)
        else:
            adstats[i] = compute_adstat(sample, cdf_splines[i], true_n=true_n)

    if debug:
        print(f"Finding Minimum after computing {num_cdfs} CDFs")
    if top_k > 1:
        min_k = np.zeros(top_k, dtype=int)
        adstats_copy = adstats.copy()
        for i in np.arange(top_k):
            min_k[i] = np.argmin(adstats_copy)
            adstats_copy[min_k[i]] = np.inf
        return adstats, [cdf_keys[j] for j in min_k], adstats[min_k]
    else:
        return adstats, cdf_keys[np.argmin(adstats)], np.min(adstats)


def add_tests_to_df_ad(cdfs_df, group, var_kurt_df, adstats, ad_cutoff):
    cdfs_df['pass_var'] = (cdfs_df['variance'] > var_kurt_df.loc[group, 'var_lower']) & (cdfs_df['variance'] < var_kurt_df.loc[group, 'var_upper'])
    cdfs_df['pass_kurt'] = (cdfs_df['kurtosis'] > var_kurt_df.loc[group, 'kurt_lower']) & (cdfs_df['kurtosis'] < var_kurt_df.loc[group, 'kurt_upper'])
    cdfs_df['adstat'] = adstats
    # cdfs_df['ksstat'] = adstats  # alias for combo_test_plot which hardcodes 'ksstat'
    cdfs_df['pass_adtest'] = cdfs_df['adstat'] < ad_cutoff
    return cdfs_df