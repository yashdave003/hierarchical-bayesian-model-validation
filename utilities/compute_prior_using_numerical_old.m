%% generate pdf for gaussian with generalized gamma variances
%matlab.engine.shareEngine
%% set parameters
function res = compute_prior(r,eta,x)
    
    beta = (eta + 1.5)/r; % change to standard parametrization
    scale = 1;

    
    %% pick discretization range and scale
    n_samples = 10^3;
    x_max = 100;
    xs = linspace(-x_max,x_max,n_samples);
    
    
    %% preallocate
    prior_pdf = nan(size(xs));
    
    %% loop over xs
    
    
        %% define integrands
    gauss_density = @(theta) (1./(sqrt(2*pi)*sqrt(theta))).*exp(-0.5*(x.^2)./theta);
    gen_gamma_density = @(theta) (abs(r)/gamma(beta))*(1/scale)*(theta/scale).^(r*beta - 1).*...
        exp(-(theta/scale).^r);
    integrand = @(theta) gauss_density(theta).*gen_gamma_density(theta);
    
    %% integrate
    res = integral(integrand,0,Inf);



