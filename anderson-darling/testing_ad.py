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

# TODO: determine N automatically via stability condition
@np.vectorize
def ad_asymptotic_cdf(z, N=5):
    '''
    asymptotic cdf of A^2 under H_0 at z using first N terms in expansion
    https://doi.org/10.1214/aoms/1177729437 (pp. 204)
    '''
    @np.vectorize
    def term(j):
        t_j = (4*j+1)**2 * np.pi**2 / (8*z)
        integrand = lambda w: np.exp(z / (8*(1+w**2)) - w**2 * t_j)
        integral = scipy.integrate.quad(integrand, 0, np.inf)[0]
        return integral * scipy.special.binom(-.5, j) * (4*j+1) * np.exp(-t_j)
    return np.sqrt(2 * np.pi) / z * np.sum(term(np.arange(N)))