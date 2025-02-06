%% generate pdf for gaussian with generalized gamma variances
%matlab.engine.shareEngine
%% set parameters
function res = compute_cdf_using_gengamma_with_scale(r,beta, scale, x)
    
    gen_gamma_cdf = @(x) gammainc((x./scale).^r, beta);
    gauss_density = @(z) (1./(sqrt(2*pi))).*exp(-0.5*(z.^2));

    integrand = @(z) gauss_density(z) .* (1 - gen_gamma_cdf((x./z).^2));
    res = quadgk(integrand, -Inf, 0);
    if x > 0
        res = 1 - res;
    end
    
warning('off', 'all');





