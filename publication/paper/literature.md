https://iopscience.iop.org/article/10.1088/1361-6420/ada17f/pdf

Introduction
Recovering a sparse parameter vector from indirect, incomplete, and noisy observations is a
common yet challenging problem in a variety of applications. The task is often modeled as a
linear inverse problem
y = Fx + e, (1.1)
where y ∈ R
M is a vector of observations, x ∈ R
N
symbolizes the unknown parameter vector,
F ∈ R
M×N
is the known linear forward operator, and e ∈ R
M corresponds to the noise component. Comprehensive discussions on inverse problems may be found in [31, 32, 57] and related
references. In particular, (1.1) may be associated with signal or image reconstruction [35, 52].
If F is ill-conditioned or if the data are significantly distorted by noise, then (1.1) becomes
ill-posed and pathologically hard to solve.


Prior knowledge about the otherwise unknown parameter vector x is often leveraged to
overcome the associated challenges. In this regard, using a Bayesian approach [17, 40, 54],
which models the parameter and observation vectors as random variables, is known to be
highly successful. In a nutshell, the sought-after posterior distribution for the parameters of
interest is characterized using Bayes’ theorem, which connects the posterior density to the
prior and likelihood densities. The prior encodes information available on the parameters of
interest before any data are observed. At the same time, the likelihood density incorporates the
data model (1.1) and a stochastic description of the measurements.

the solution to the imaging problem is not a single image but a distribution of images, and it is possible to analyze the uncertainties due to measurement errors and to explicitly stated prior beliefs." Calvetti et. al. 2008