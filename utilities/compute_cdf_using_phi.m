%% generate pdf for gaussian with generalized gamma variances
%matlab.engine.shareEngine
%% set parameters
function res = compute_cdf_using_phi(r,eta, x)
    
    beta = (eta + 1.5)/r; % change to standard parametrization
    scale = 1;

    
    
    
    %% define integrands
    
    
    gen_gamma_density = @(theta) (abs(r)/gamma(beta))*(1/scale)*(theta/scale).^(r*beta - 1).*...
        exp(-(theta/scale).^r);
    %% integrate
    
    integrand = @(theta) normcdf(x./sqrt(theta)) .* gen_gamma_density(theta);
    res = integral(integrand, 0, Inf);




