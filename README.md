
# Variational inference for approximate reference priors using neural networks

## Guidelines

The folder Multiseed_AJ contains saved *a posteriori* estimates for the probit statistical model.

The posterior is derived from the Jeffreys prior or the constrained Jeffreys prior of the model:
    $$\pi(\theta|\mathbf{X}) \propto J(\theta)L_N(\mathbf{X}|\theta) \quad\text{or}\quad \pi(\theta|\mathbf{X}) \propto J(\theta)\theta_{2}^{\kappa/\alpha}L_N(\mathbf{X}|\theta)$$
with
    $$J(\theta)=\sqrt{|\det\mathcal{I}(\theta)|};\quad \mathcal{I}(\theta)=-\int_{\mathcal{X}^N}\frac{\partial^2\log L_N}{\partial\theta^2}(\mathbf{X}|\theta)\cdot L_N(\mathbf{X}|\theta)d\mathbf{X}.$$

We rely, for the computation of these estimates, on the code provided in the repository [bayes_frag](https://github.com/vbkantoine/bayes_frag).
The derivation of the Jeffreys prior is conducted via numerical integrations; and the generation of *a posteriori* estimates is done thanks to a Metropolis Hasting algorithm.

For reproducibility, the jupyter notebook `probit_runs_with_Jeffreys.ipynb` proposed in the same folder allows to re-generate the results.
