function res = compute_cdf_using_gengamma_vectorized(r, beta, x_vec)
    % Define functions
    gen_gamma_cdf = @(x) gammainc(x.^r, beta);
    gauss_density = @(z) (1./sqrt(2*pi)) .* exp(-0.5*z.^2);

    % Preallocate result
    res = zeros(size(x_vec));

    % Compute CDF for each x
    res = arrayfun(@(x) compute_single_cdf(x, gen_gamma_cdf, gauss_density), x_vec);
    
    % Nested function to compute single CDF
    function p = compute_single_cdf(x, gen_gamma_cdf, gauss_density)
        integrand = @(z) gauss_density(z) .* (1 - gen_gamma_cdf((x./z).^2));
        p = quadgk(integrand, -Inf, 0);
        if x > 0
            p = 1 - p;
        end
    end
end






