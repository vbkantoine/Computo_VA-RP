########################  Probit experiment  ########################
# Prior : improper and not explicit (no criteria)
# Posterior : proper and not explicit (no MLE fitting)

# Python files
import os 
import sys
import pickle

current_file = os.path.abspath(__file__)

# Set the working directory to the project root (going up 4 levels)
project_root = os.path.abspath(os.path.join(current_file, "../../../../"))
os.chdir(project_root)

py_dir = os.path.join(project_root, "python_files")
if py_dir not in sys.path:
    sys.path.append(py_dir)

from aux_optimizers import *
from stat_models_torch import *
from neural_nets import *
from variational_approx import *
from div_metrics_torch import *
from constraints import *

probit_path = os.path.join(project_root, "data_probit")
tirages_path = os.path.join(probit_path, "tirages_probit")
int_jeffreys_path = os.path.join(probit_path, "Multiseed_AJ")

use_cluster = True
disable_tqdm = False
init_seed = 0
n_exp = 100 - init_seed

use_constraints = True

def experiment(ntask):
    seed_all(ntask)

    ############  Parameters and classes  ############
    p = 50     # latent space dimension
    q = 2      # parameter space dimension
    N = 500     # number of data samples
    J = 500   # nb samples for MC estimation in MI
    T = 50     # nb samples MC marginal likelihood
    input_size = p  
    output_size = q
    mu_a, sigma2_a = 0, 1
    #mu_a, sigma2_a = 8.7 * 10**-3, 1.03
    Probit = torch_ProbitModel(use_log_normal=True, mu_a=mu_a, sigma2_a=sigma2_a, set_beta=None, alt_scaling=True)
    low = 0.0001          # lower bound 
    upp = 1 + low
    n_samples_prior = 10**6

    ########################  Prior ########################

    ############  Neural network  ############
    NN = SingleLinear(input_size, output_size, m1=0, s1=0.1, b1=0, act1=nn.Identity())
    #NN = DoubleLinear(input_size, input_size, output_size, m1=0, s1=0.1, b1=0, m2=0, s2=0.1, b2=0, act1=nn.PReLU(), act2=nn.Identity())
    NN = DifferentActivations(NN, [torch.exp, nn.Softplus()])
    #NN = DifferentActivations(NN, [torch.exp, torch.exp])
    NN = AffineTransformation(NN, low, upp)
    VA = VA_NeuralNet(neural_net=NN, model=Probit)
    #Div = DivMetric_NeuralNet(va=VA, T=T, use_alpha=False, use_log_lik=True)
    alpha = 0.5
    Div = DivMetric_NeuralNet(va=VA, T=T, use_alpha=True, alpha=alpha, use_log_lik=True)

    if not use_constraints : 
        ############  Optimization  ############
        num_epochs = 10000
        loss_fct = "LB_MI"
        optimizer = torch_Adam
        num_samples_MI = 100
        freq_MI = 500
        save_best_param = True
        learning_rate = 0.001
        weight_decay = 0.0
        num_samples_grad = 1
        MI, range_MI, lower_MI, upper_MI = Div.Partial_autograd(J, N, num_epochs, loss_fct, optimizer, num_samples_MI, freq_MI, 
                                                                save_best_param, learning_rate, weight_decay=weight_decay,
                                                                num_samples_grad=num_samples_grad,
                                                                disable_tqdm=disable_tqdm, momentum=True)

        seed_all(0)
        theta_sample = Div.va.implicit_prior_sampler(n_samples_prior)
        with torch.no_grad():
            theta_numpy = theta_sample.numpy()
            theta_sample = theta_numpy
        jeffreys_sample = []

    else : 
        ############  Constrained Optimization (assuming the same unconstrained training for all seeds) ############
        # Constraints parameters
        kappa = 1/8
        constr_val = 0.839594841003418
        alpha_constr = 1.040394
        beta = torch.tensor([kappa])
        b = torch.tensor([[alpha_constr,constr_val]]) 
        T_cstr = 100000
        eta_augm = torch.tensor([[0.,1.]])
        eta = torch.tensor([[0.,1.]])
        Constr = Constraints_NeuralNet(div=Div, betas=beta, b=b, T_cstr=T_cstr, 
                                    objective='LB_MI', lag_method='augmented', eta_augm=eta_augm, rule='SGD')
 
        with torch.no_grad():
            theta_sample_init = Div.va.implicit_prior_sampler(n_samples_prior).numpy()

        num_epochs = 10000
        optimizer = torch_Adam
        num_samples_MI = 100
        freq_MI = 500
        save_best_param = False
        learning_rate = 0.0005
        freq_augm = 100

        ### Training loop
        MI, constr_values, range_MI, lower_MI, upper_MI = Constr.Partial_autograd(J, N, eta, num_epochs, optimizer, num_samples_MI, 
                                                                freq_MI, save_best_param, learning_rate, momentum=True,
                                                                freq_augm=freq_augm, max_violation=0.005, update_eta_augm=2.)

        seed_all(0)
        with torch.no_grad():
            theta_sample = Constr.va.implicit_prior_sampler(n_samples_prior).numpy()
        jeffreys_sample = []

    ########################  Posterior  ########################
    seed_all(0)  # Reset RNG so the effect on the seed is only on the optimization part of the algorithm

    N = 50
    n_samples_post = 5000
    T_mcmc = 5*10**4 + 1
    file_path = os.path.join(tirages_path, 'tirages_data')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    data_A, data_Z = data[0], data[1]

    i = 2      # Choice of the (non-degenerate !) dataset 
    Xstack = np.stack((data_Z[:N, i],data_A[:N, i]),axis=1)
    X = torch.tensor(Xstack)
    D = X.unsqueeze(1)
    Probit.data = D
    sigma2_0 = torch.tensor(1.)
    eps_0 = torch.randn(p)
    eps_MH, batch_acc = VA.MH_posterior(eps_0, T_mcmc, sigma2_0, target_accept=0.4, adap=True, Cov=True, disable_tqdm=disable_tqdm)
    theta_MH = NN(eps_MH)
    with torch.no_grad():
        theta_MH = theta_MH.detach().numpy()
    theta_post = theta_MH[-n_samples_post:,-n_samples_post:]
    if not use_constraints :
        file_path = os.path.join(int_jeffreys_path, f'model_J_{i}')
    else :
        file_path = os.path.join(int_jeffreys_path, f'model_J_constraint_{i}')

    with open(file_path, 'rb') as file:
        model_J = pickle.load(file)
    jeffreys_post = model_J['logs']['post'][N]
    jeffreys_post = np.reshape(jeffreys_post, (n_samples_post, q))

    ########################  Save values  ########################

    if not use_constraints : 
        name_file = f'probit_seed{ntask}.pkl'
    else : 
        name_file = f'probit_constr_seed{ntask}.pkl'
    prior_samples = {'VA' : theta_sample, 'Jeffreys' : jeffreys_sample}
    post_samples = {'VA' : theta_post, 'Jeffreys' : jeffreys_post}
    saved_values = {'prior' : prior_samples, 'post' : post_samples}

    with open(name_file, 'wb') as file:
        pickle.dump(saved_values, file)

    #os.environ.clear()
    print('End of experiment')

if use_cluster :
    ntask = int(sys.argv[1]) + init_seed
    experiment(ntask)
else :
    for ntask in range(1 + init_seed, n_exp + 1 + init_seed):
        print(f'Seed number : {ntask}')
        experiment(ntask)
    print('Done !')