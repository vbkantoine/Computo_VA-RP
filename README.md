
# Variational inference for approximate reference priors using neural networks

Nils Baillie, Antoine Van Biesbroeck and Clément Gauchy

## Guidelines

The main notebook is obtained by default by loading every necessary data file without doing the computations, some options in the beginning of the code can be changed if one wants to re-train the networks or re-sample the posterior distributions. These files are mainly there to generate the plots rapidly, they can be found in the folder "plots_data".

For the probit model, the computations were made on many different RNG seeds, the output files and the corresponding scripts can be found in the folder "data_probit/Multiseed_VARP". The folder "data_probit/Multiseed_AJ" contains saved *a posteriori* estimates for the probit statistical model.

The posterior is derived from the Jeffreys prior or the constrained Jeffreys prior of the model:
    $$\pi(\theta|\mathbf{X}) \propto J(\theta)L_N(\mathbf{X}|\theta) \quad\text{or}\quad \pi(\theta|\mathbf{X}) \propto J(\theta)\theta_{2}^{\kappa/\alpha}L_N(\mathbf{X}|\theta)$$
with
    $$J(\theta)=\sqrt{|\det\mathcal{I}(\theta)|};\quad \mathcal{I}(\theta)=-\int_{\mathcal{X}^N}\frac{\partial^2\log L_N}{\partial\theta^2}(\mathbf{X}|\theta)\cdot L_N(\mathbf{X}|\theta)d\mathbf{X}.$$

We rely, for the computation of these estimates, on the code provided in the repository [bayes_frag](https://github.com/vbkantoine/bayes_frag).
The derivation of the Jeffreys prior is conducted via numerical integrations; and the generation of *a posteriori* estimates is done thanks to a Metropolis Hasting algorithm.

For reproducibility, the python file `probit_runs_with_Jeffreys.py` proposed in the same folder allows to re-generate the results. Since the presence of this file lead to a failure of the build, here are 
its contents : 

```python
# %% [markdown]
# # Estimation of parameters in the probit model with an 'exact' Jeffreys
# 
# This script allows the conduction of *a posteriori* sampling of the paramter in the probit statistical model with a finely approximated Jeffreys prior.
# 
# 
# The derivation of the Jeffreys prior for the probit model is proposed in the repository [bayes_frag](https://github.com/vbkantoine/bayes_frag). That external code also allows to directly generate samples from the posterior yielded by that approximated Jeffreys prior.
# 

# %% [markdown]
# ### 1. Clone the bayes_frag github

# %%
# 1. clone github bayes_frag

import os
import sys

if not os.path.exists('bayes_frag') :
    !git clone "https://github.com/vbkantoine/bayes_frag.git"

sys.path.append(os.path.join(os.getcwd(),'bayes_frag'))

# %% [markdown]
# ### 2. Compute the Jeffreys prior for this model
# 
# We recall that the probit model is defined as a parametrized model where $\theta=(\theta_1,\theta_2)\in(0,\infty)^2$ is the parameter and the observed variable is $(Z,a)$ where 
# $$\left\lbrace\begin{array}{l}a\sim\mathrm{Log-}\mathcal{N}(\mu_a,\sigma_a^2)\\
# Z\sim\mathrm{Bernoulli}(P_f(a))\end{array}\right.,$$
# with $P_f(a)=\Phi\left(\frac{\log a-\log\theta_1}{\theta_2}\right)$, and $\Phi$ denoting the c.d.f. of a standard Gaussian.
# 
# The Jeffreys prior of this model depends on the distribution of $a$, i.e. it depends on $\mu_a$ and $\sigma_a$.
# 
# That external repository proposes a code to compute the Jeffreys prior given a distribution of $a$. Actually, it derives a fine numerical approximation of the Fisher information matrix that is stored in a file called `fisher`.
# 
# The exection of that code is generally very long given the complex expression of the Fisher information matrix.
# For this reason, we suggest to download the one that we have already computed and that we provide online on [OSF](https://osf.io/gvqw4/files/osfstorage/678a826e9b2975f377dd6f3f). The dowloaded file `fisher` can be placed at the root of the current directory.
# 
# We also have provided a lighter approximation of the Fisher information matrix based on a less thin derivation. It is stored in the file called `fisher_light`, that can be renamed by `fisher` to be used. 
# 
# If one wants to do the computations instead of downloading the `fisher` file or renaming the `fisher_light`file, the last line of the following cell must be uncommented. 
# 

# %%
# 2. compute and save a fine mesh of Jeffreys prior

import util.create_fisher_artificial as cfa

dat = cfa.Data_simplified_big()

def save_fisher_computation():
    save_fisher_arr = cfa.dict_save_fisher['save_fisher_arr'] 
    # selecting the dictionnary cfa.dict_save_fisher2 above lead to a lighter appproximation of Fisher 
    # (the one that we have provided and called `fisher_light`)

    alpha_min = save_fisher_arr.alpha_min
    alpha_max = save_fisher_arr.alpha_max
    beta_min = save_fisher_arr.beta_min
    beta_max = save_fisher_arr.beta_max
    num = save_fisher_arr.num_alpha

    theta_tab1 = save_fisher_arr.alpha_tab
    theta_tab2 = save_fisher_arr.beta_tab

    function = cfa.fisher.fisher_function("simpson", dat)
    I = cfa.save_fisher(cfa.save_path, function, theta_tab1, theta_tab2)
    
# # save_fisher_computation()


# %% [markdown]
# ### 3. Conduction of *a posteriori* sampling
# 
# In the following, we import data from the file `../tirages_data` to derive a posterior that is used to generate samples of the parameter $\theta$.
# Then, the results are saved on different files according to the prior: Jeffreys or the constrained Jeffreys.
# 
# 

# %%
import os
import numpy as np
import pickle
from numba import jit

from bayes_frag import stat_functions as stat_f
from bayes_frag.data import Data
from bayes_frag.model import Model
from bayes_frag import config
from bayes_frag.reference_curves import Reference_known_MLE
from bayes_frag.extract_saved_fisher import Approx_fisher

from util.create_fisher_artificial import dict_save_fisher

# %%
class Data_simplified(Data) :
    """
       This class serves the import of data contained in an external file 
    """
    def __init__(self, i, pickle_path):
        """
        Args:
            i (int): in the generated array, id to take into account in this run
        """
        file = open(pickle_path, mode='rb')
        array_A_Z = pickle.load(file)
        self.A = np.array(array_A_Z[0][:,i,np.newaxis], dtype=np.float, order='C')
        self.Z = np.array(array_A_Z[1][:,i,np.newaxis], dtype=np.float, order="C")
        self.Y = np.ones_like(self.A)
        self.a_tab = None
        self.h_a = None
        self._set_a_tab()
        self.f_A = None
        self.f_A_tab = None
        self._compute_f_A()
        self.increasing_mode = True

# %%

kappa = 1/8
alpha = 1/2
gamma = kappa/alpha

save_folder = './'
path_tirages_data = '../tirages_probit/'

theta_vrai = np.array([3.37610525, 0.43304097])


def experiment(N) :
    init_seed = 0
    SEED = init_seed + int(N)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    assert not os.path.exists(os.path.join(save_folder, 'model_J_constraint_{}'.format(N-1))), 'existing run no {}'.format(N)

    i_data = N-1

    data = Data_simplified(i_data, os.path.join(path_tirages_data, 'tirages_data'))

    # print(data.A.shape)
    # print(data.Z)
    error_draw = np.arange(1,41)*5
    n_est = 5000

    ##
    # parameters models
    linear = False
    approx_fisher_class = Approx_fisher(dict_save_fisher['save_fisher_path'], dict_save_fisher['save_fisher_arr'], fisher_file_path_is_personalized=True)
    fisher =  approx_fisher_class.fisher_approx
    likelihood = stat_f.log_vrais

    iter_HM = 15000
    sigma_prop = np.array([[0.1,0],[0,0.095]])
    t0 = np.array([3,0.3])
    HM_fun = stat_f.adaptative_HM


    @jit
    def prior(th) :
        I = fisher(th)
        return 1/2 * np.log(np.nan_to_num(np.abs(I[:,0,0]*I[:,1,1]-I[:,0,1]**2)))

    # log_post for this case :
    @jit(nopython=True)
    def log_post(z,a, theta) :
        return stat_f.log_post_jeff_adapt(theta,z,a, Fisher=fisher)

    @jit
    def prior_adapted(th) :
        return prior(th) + gamma * np.log(th[:,1]*th[:,1])

    @jit(nopython=True)
    def log_post_adapted(z,a,theta) :
        return log_post(z,a,theta) + gamma*np.log(theta[:,1])

    ref = Reference_known_MLE(data, theta_vrai)
    
    HM_post_simul_constraint = stat_f.Post_HM_Estimator(HM_fun, t0, log_post_adapted, pi_log=True, max_iter=iter_HM, sigma0=sigma_prop)
    model_J_constraint = Model(prior_adapted, likelihood, data, HM_post_simul_constraint, linear=linear, ref=ref)

    HM_post_simul = stat_f.Post_HM_Estimator(HM_fun, t0, log_post, pi_log=True, max_iter=iter_HM, sigma0=sigma_prop)
    model_J = Model(prior, likelihood, data, HM_post_simul, linear=linear, ref=ref)

    
    ## run everything

    model_J_constraint.run_simulations(error_draw, n_est, sim=['post'], print_fo=True)
    model_J.run_simulations(error_draw, n_est, sim=['post'], print_fo=True)

    pickle.dump({'logs': model_J_constraint.logs, 'A':model_J_constraint.A, 'S':model_J_constraint.S, 'seed':SEED}, open(os.path.join(save_folder, "model_J_constraint_{}".format(i_data)), "wb"))
    pickle.dump({'logs': model_J.logs, 'A':model_J.A, 'S':model_J.S, 'seed':SEED}, open(os.path.join(save_folder, "model_J_{}".format(i_data)), "wb"))




# %% [markdown]
# The above function can be ran several time to get samples given different samples of the data.
# The following permits to conduct $10$ of these experiments. 
# It re-generates the files `model_J_*` that are provided in the directory.
# We warn the user that this execution can take a long time.

# %%
# 4. (long code) all the following run section 3. for 10 different seeds
for N in range(1,11) :
    print('****** Experiment no {}/10 ******'.format(N))
    experiment(N)

# %%



```
