# Python files
import os 
import sys
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

use_cluster = False
disable_tqdm = False
use_constraints = True
num_M = 10


def compute(M):
    # Parameters and classes
    p = 50     # latent space dimension
    q = 2      # parameter space dimension
    N = 500    # number of data samples
    J = 500    # nb samples for MC estimation in MI
    T = 50     # nb samples MC marginal likelihood
    input_size = p  
    output_size = q
    low = 0.0001          # lower bound 
    upp = 1 + low
    n_samples_prior = 10**6

    mu_a, sigma2_a = 0, 1
    Probit = torch_ProbitModel(use_log_normal=True, mu_a=mu_a, sigma2_a=sigma2_a, set_beta=None, alt_scaling=True)

    kappa = 1/8
    constr_val = 0.839594841003418
    alpha_constr = 1.040394

    # Constraints
    beta = torch.tensor([kappa])
    b = torch.tensor([[alpha_constr,constr_val]]) 
    T_cstr = 100000
    eta_augm = torch.tensor([[0.,1.]])
    eta = torch.tensor([[0.,1.]])

    seed_all(0)
    NN = SingleLinear(input_size, output_size, m1=0, s1=0.1, b1=0, act1=nn.Identity())
    NN = DifferentActivations(NN, [torch.exp, nn.Softplus()])
    NN = AffineTransformation(NN, low, upp)
    VA = VA_NeuralNet(neural_net=NN, model=Probit)
    Div = DivMetric_NeuralNet(va=VA, T=T, use_alpha=True, alpha=0.5, use_log_lik=True)

    if not use_constraints : 
        # Correspond to the NN parameters obtained in "Probit_results_unconstrained.pkl" in "plots_data"
        all_params = torch.tensor([-0.2251, -0.3482, -0.3293,  0.3671,  0.1878,  0.3218, -0.1059, -0.0419,
                0.1192,  0.4353, -0.1900, -0.0502,  0.1812, -0.0683,  0.3681,  0.1530,
                -0.1607, -0.0196, -0.2561, -0.3685, -0.1274, -0.0556, -0.1713, -0.2469,
                0.5001,  0.1387,  0.1375,  0.0660,  0.0477, -0.1016,  0.0180,  0.0108,
                0.1951, -0.1063,  0.1140, -0.0090,  0.0730, -0.1845, -0.0025,  0.1369,
                -0.0313,  0.0246,  0.0377,  0.1101, -0.1143,  0.0038,  0.2696,  0.1236,
                -0.0201, -0.0118, -0.4036, -0.1407,  0.1627,  0.0172, -0.1612, -0.0479,
                0.0157,  0.0385,  0.0574,  0.0998,  0.0544,  0.0079,  0.0863, -0.0019,
                0.0761,  0.0618, -0.0299, -0.0188,  0.1916,  0.0690, -0.2322, -0.1196,
                0.0241, -0.1396,  0.0114,  0.1105, -0.3106, -0.0663, -0.1375,  0.0500,
                0.3553,  0.0676,  0.1155, -0.0653, -0.2178,  0.0799, -0.2201, -0.2170,
                0.1683, -0.0262,  0.2586,  0.0022, -0.1334,  0.1742, -0.0539, -0.1406,
                -0.0515,  0.0910, -0.0562,  0.2148,  0.1945, -1.4981])
        assign_parameters_to_NN(NN, all_params)

    else : 

        Constr = Constraints_NeuralNet(div=Div, betas=beta, b=b, T_cstr=T_cstr, 
                                    objective='LB_MI', lag_method='augmented', eta_augm=eta_augm, rule='SGD')
        
        # Correspond to the NN parameters obtained in "Probit_results_constrained.pkl" in "plots_data"
        all_params = torch.tensor([-0.2129, -0.1363, -0.1263,  0.3551,  0.2177,  0.1545, -0.1633, -0.1499,
         0.2277, -0.0730,  0.0843, -0.1268, -0.0578,  0.0833,  0.2679,  0.1115,
        -0.0923, -0.0637, -0.0424, -0.0701, -0.0503, -0.2664,  0.1334, -0.2465,
         0.1625,  0.1387,  0.1375,  0.0660,  0.0477, -0.1016,  0.0180,  0.0108,
         0.1951, -0.1063,  0.1140, -0.0090,  0.0730, -0.1845, -0.0025,  0.1369,
        -0.0313,  0.0246,  0.0377,  0.1101, -0.1143,  0.0038,  0.2696,  0.1236,
        -0.0201, -0.0118, -0.2425, -0.1407,  0.1627,  0.0172, -0.1612, -0.0479,
         0.0157,  0.0385,  0.0574,  0.0998,  0.0544,  0.0079,  0.0863, -0.0019,
         0.0761,  0.0618, -0.0299, -0.0188,  0.1916,  0.0690, -0.2322, -0.1196,
         0.0241, -0.1396,  0.0114,  0.1105, -0.2489,  0.0133, -0.0110,  0.1777,
         0.2390,  0.1181,  0.0553, -0.1010,  0.0345, -0.0352, -0.1398, -0.2927,
        -0.0316, -0.1553,  0.1273,  0.0790, -0.0073,  0.1075, -0.0262,  0.1288,
        -0.0428, -0.1288,  0.1752,  0.1638, -0.0076, -1.2811])
        assign_parameters_to_NN(NN, all_params)

    seed_all(1)
    theta_vrai = np.array([3.37610525, 0.43304097])
    import pickle
    file_path = os.path.join(tirages_path, 'tirages_data')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    N_max = 200
    N_init = 5
    tab_N = np.arange(N_init, N_max+N_init, 5)
    num_MC = 5000
    erreur = np.zeros(len(tab_N))
    data_A, data_Z = data[0], data[1]
    j = 0

    for j in tqdm(range(len(tab_N)), desc=f'Iterations', disable=disable_tqdm) : 
        N = tab_N[j]
        resultats_theta_post_N = np.zeros((num_MC, 2))
        i = M
        Xstack = np.stack((data_Z[:N, i],data_A[:N, i]),axis=1)
        X = torch.tensor(Xstack)
        D = X.unsqueeze(1)
        Probit.data = D
        n_samples_post = num_MC
        T_mcmc = 5*10**4 + 1
        sigma2_0 = torch.tensor(1.)
        eps_0 = torch.randn(p)
        eps_MH, batch_acc = VA.MH_posterior(eps_0, T_mcmc, sigma2_0, target_accept=0.4, adap=True, Cov=True, disable_tqdm=True)
        theta_MH = NN(eps_MH)
        theta_post = theta_MH[-n_samples_post:,-n_samples_post:]
        resultats_theta_post_N = theta_post.detach().numpy() 
        erreur[j] = np.mean(np.linalg.norm(resultats_theta_post_N - theta_vrai[np.newaxis], axis=1))
        j = j + 1

    if not use_constraints : 
        name_file = f'Quad_error_post{M}.pkl'
    else : 
        name_file = f'Quad_error_post_constr{M}.pkl'

    with open(name_file, 'wb') as file:
        pickle.dump(erreur, file)

if use_cluster :
    ntask = int(sys.argv[1]) 
    compute(ntask)
else :
    for M in range(1, num_M + 1):
        print(f'Dataset number : {M}')
        compute(M)
    print('Done !')