%% generate pdf for gaussian with generalized gamma variances
function res = FullPriorDensity(r, eta)
    %% set parameters
    beta = (eta + 1.5)/r; % change to standard parametrization
    scale = 1;
    
    %% pick discretization range and scale
    n_samples = 10^5;
    x_max = 362420;
    xs = linspace(-x_max,x_max,n_samples);
    
    
    %% preallocate
    prior_pdf = nan(size(xs));
    
    %% loop over xs
    for j = 1:length(xs)
    x = xs(j);
    
    %% define integrands
    gauss_density = @(theta) (1./(sqrt(2*pi)*theta)).*exp(-0.5*(x./theta).^2);
    gen_gamma_density = @(theta) (r/gamma(beta))*(1/scale)*(theta/scale).^(r*beta - 1).*...
        exp(-(theta/scale).^r);
    integrand = @(theta) gauss_density(theta).*gen_gamma_density(theta);
    
    %% integrate
    prior_pdf(j) = integral(integrand,0,Inf);
    
    end

    sum(prior_pdf)
    trapz(xs,prior_pdf)
    res = prior_pdf;
