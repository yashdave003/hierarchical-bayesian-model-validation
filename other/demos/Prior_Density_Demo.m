%% generate pdf for gaussian with generalized gamma variances

%% set parameters
r = 0.2;
eta = 1.4;
beta = (eta + 1.5)/r; % change to standard parametrization
scale = 1;

%% pick discretization range and scale
n_samples = 10^3;
x_max = 10000;
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
%% display
figure(1)
%clf

subplot(1,2,1)
hold on
plot(xs,prior_pdf,'Linewidth',2)
grid on
xlabel('$x$','FontSize',18,'Interpreter','latex');
ylabel('Prior Density: $\pi(x)$','FontSize',18,'Interpreter','latex');

subplot(1,2,2)
hold on
plot(xs,prior_pdf,'Linewidth',2)
grid on
xlabel('$x$','FontSize',18,'Interpreter','latex');
ylabel('Prior Density: $\pi(x)$ (Log Scale)','FontSize',18,'Interpreter','latex');
set(gca,'yscale','log')

