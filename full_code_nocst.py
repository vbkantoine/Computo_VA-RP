import os
import sys
# base_dir = r'C:\Users\AV273338\Documents\GitHub\Computo_VA-RP'  # the path
base_dir = r'/Users/antoinevanbiesbroeck/Documents/GitHub/Nils/Computo_VA-RP'
# {must be set on VA-RP
py_dir = os.path.join(base_dir, "python_files")
if py_dir not in sys.path:
    sys.path.append(py_dir)
#print(sys.path)
from aux_optimizers import *
from stat_models_torch import *
from neural_nets import *
from variational_approx import *
from div_metrics_torch import *
from constraints import *

# Packages used
import scipy.special as spc
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import gridspec
import pickle
from bayes_frag import plot_functions



# rc('text', usetex=False)
# rc('lines', linewidth=2)
rougeCEA = "#b81420"
vertCEA = "#73be4b"

path_plot = os.path.join(base_dir, "plots_data")
probit_path = os.path.join(base_dir, "data_probit")
tirages_path = os.path.join(probit_path, "tirages_probit")
int_jeffreys_path = os.path.join(probit_path, "Multiseed_AJ")
varp_probit_path = os.path.join(probit_path, "Multiseed_VARP")

### Load parameter for each main computation
# If True, loads the different results without re-doing every computation
# If False, all computations will be done and saved (and possibly overwrite the data files !)


load_results_Probit_nocstr = True
load_results_Probit_cstr = True


a_tab = np.linspace(10**-3, 15, num=100)

frag_curve = lambda a, theta : 1/2+1/2*spc.erf(np.log(a[np.newaxis]/theta[:,0,np.newaxis])/theta[:,1,np.newaxis])
theta_vrai = np.array([3.37610525, 0.43304097])
ref = 1/2+1/2*spc.erf(np.log(a_tab/theta_vrai[0])/theta_vrai[1])


def plot_frag_cred(theta_post, ax, color, label=None) :
    curves = frag_curve(a_tab, theta_post)
    q1 = np.quantile(curves, 1-0.05/2, axis=0)
    q2 = np.quantile(curves, 0.05/2, axis=0)
    # med = np.median(curves, axis=0)
    ax.fill_between(a_tab, q1, q2, color=color, alpha=0.6)
    if not label is None :
        ax.plot(a_tab, q1, color=color, alpha=0.6, label=label)
    # ax.plot(a_tab, med)
    ax.legend()
    return ax




###


#Parameters and classes
p = 50
q = 2
N = 50
J = 500
T = 50
input_size = p
output_size = q
low = 0.0001
upp = 1 + low
mu_a, sigma2_a = 0, 1
#mu_a, sigma2_a = 8.7 * 10**-3, 1.03
Probit = torch_ProbitModel(use_log_normal=True, mu_a=mu_a, sigma2_a=sigma2_a, set_beta=None, alt_scaling=True)
n_samples_prior = 10**6
alpha = 0.5
name_file = 'Probit_results_unconstrained.pkl'
file_path = os.path.join(path_plot, name_file)

seed_all(0)
sqrt2 = np.sqrt(2)
act_bet = lambda x: torch.log(1- 0.5*(1+torch.erf(x/sqrt2)) )
NN = NetProbitSeparate(input_size, m1=0, s1=0.1, b1=0, pre_act=[nn.Identity(), act_bet], act1=[torch.exp, torch.exp])

name_file_w = 'Probit_results_unconstrained_torch_state_good'
file_path_w = os.path.join(path_plot, name_file_w)
NN.load_state_dict(torch.load(file_path_w, weights_only=True))
# NN = DifferentActivations(NN, [torch.exp, nn.Softplus()])
# NN = AffineTransformation(NN, low, upp)
VA = VA_NeuralNet(neural_net=NN, model=Probit)
#print(f'Number of parameters : {VA.nb_param}')
Div = DivMetric_NeuralNet(va=VA, T=T, use_alpha=True, alpha=alpha, use_log_lik=True)
Div.penalize_norm_param=0
Div.save_params = True

with torch.no_grad():
    Thetas_ = Div.va.implicit_prior_sampler(n_samples_prior)
    theta_sample_init = Thetas_.numpy()
    Thetas_ = delete_nan_elements(Thetas_[-100:])
    # MI_list = np.array([Div.MI(theta, J, N).item() for theta in Thetas_])
    MI_list = Div.MI(Thetas_, J, N).numpy()
    MI_list = remove_nan_and_inf(MI_list)
    MI_current = np.mean(MI_list)

num_epochs = 5000
loss_fct = "LB_MI"
optimizer = torch_Adam
num_samples_MI = 10
freq_MI = 10
save_best_param = False
learning_rate = 0.001

if not load_results_Probit_nocstr :
    MI, range_MI, lower_MI, upper_MI = Div.Partial_autograd(J, N, num_epochs, loss_fct, optimizer, num_samples_MI,
                                                            freq_MI, save_best_param, learning_rate,num_samples_grad=40, momentum=True)
    all_params = torch.cat([param.view(-1) for param in NN.parameters()])

    # with torch.no_grad() :
    #     NN.netbeta.singl.fc1.weight[0,0] += -NN.netbeta.singl.fc1.weight[0,0] +1/0.1
    #     NN.netbeta.singl.fc1.weight[0,1] += -NN.netbeta.singl.fc1.weight[0,1] +4

    seed_all(0)
    theta_sample = Div.va.implicit_prior_sampler(n_samples_prior)
    with torch.no_grad():
        theta_sample_prior = theta_sample.numpy()

    with open(os.path.join(tirages_path, 'tirages_data'), 'rb') as file:
        data = pickle.load(file)
    data_A, data_Z = data[0], data[1]
    theta_true = np.array([3.37610525, 0.43304097])
    N = 50  # non-degenerate
    i = 2

    # seed_all(0)
    Xstack = np.stack((data_Z[:N, i],data_A[:N, i]),axis=1)
    X = torch.tensor(Xstack)
    D = X.unsqueeze(1)
    Probit.data = D
    n_samples_post = 5000
    T_mcmc = 4*10**5 + 1
    sigma2_0 = torch.tensor(1.)
    eps_0 = torch.randn(p)
    #eps_0 = 10 * torch.ones(p)
    eps_MH, batch_acc = VA.MH_posterior(eps_0, T_mcmc, sigma2_0, target_accept=0.4, adap=True, Cov=True, disable_tqdm=False)
    theta_MH = NN(eps_MH)
    with torch.no_grad():
        theta_MH = theta_MH.detach().numpy()
    theta_post_nocstr = theta_MH[-n_samples_post:,-n_samples_post:]

    # Saves all relevant quantities
    Mutual_Info = {'values' : MI, 'range' : range_MI, 'lower' : lower_MI, 'upper' : upper_MI, 'weights':Div.keep_params}
    weights_hist = Div.keep_params
    Prior = {'samples' : theta_sample_prior, 'jeffreys' : None, 'params' : all_params}
    Posterior = {'samples' : theta_post_nocstr, 'jeffreys' : None}
    Probit_results_nocstr = {'MI' : Mutual_Info, 'prior' : Prior, 'post' : Posterior}
    with open(file_path, 'wb') as file:
        pickle.dump(Probit_results_nocstr, file)


    # torch.save(NN.state_dict(),   os.path.join(path_plot, 'Probit_results_unconstrained_torch_state_1'))

else :
    with open(file_path, 'rb') as file:
        res = pickle.load(file)
        MI_dic = res['MI']
        Prior = res['prior']
        Posterior = res['post']
        MI, range_MI, lower_MI, upper_MI = MI_dic['values'], MI_dic['range'], MI_dic['lower'], MI_dic['upper']
        theta_sample_prior, jeffreys_sample, all_params = Prior['samples'], Prior['jeffreys'], Prior['params']
        theta_post_nocstr, jeffreys_post = Posterior['samples'], Posterior['jeffreys']
        weights_hist = MI_dic['weights']

with torch.no_grad():
    mult_w2 = NN.netbeta.singl.fc1.weight.detach().numpy()
w2_neg = 1/mult_w2[mult_w2<0]
w2_pos = 1/mult_w2[mult_w2>0]
print('multiplicative weights on beta:')
print('neagtives belong to [{},{}]'.format(w2_neg.min(), w2_neg.max()))
print('positives belong to [{},{}]'.format(w2_pos.min(), w2_pos.max()))






####


plt.figure(figsize=(4, 3))
plt.plot(range_MI, MI, '-', color='#8D725B')
plt.plot(range_MI, MI, '*', color='#8D725B')
# for i in range(len(range_MI)):
#     plt.plot([range_MI[i], range_MI[i]], [lower_MI[i], upper_MI[i]], color='black')
plt.plot(range_MI, upper_MI, '-', color='lightgrey')
plt.plot(range_MI, lower_MI, '-', color='lightgrey')
plt.fill_between(range_MI, lower_MI, upper_MI, color='lightgrey', alpha=0.5)
plt.xlabel(r"Epochs")
plt.ylabel(r"Generalized mutual information")
plt.grid()
#plt.yscale('log')
plt.tight_layout()
plt.show()

print('MI fistr value: {}'.format(MI_current))
# MI_list = np.concatenate( [Div.MI(torch.tensor(theta_post_nocstr[-100*(i+1):-100*(i)-1]), J, N).numpy()  for i in range(10)] )
with torch.no_grad():
    MI_list = Div.MI(torch.tensor(theta_post_nocstr[-100:]), J, N).numpy()
#  MI_list = remove_nan_and_inf(MI_list)
MI_last = np.nanmean(MI_list)
print('MI last value: {}'.format(MI_last))

del_infs_np = lambda x : x[np.isfinite(x)]

plt.figure(figsize=(5, 4))
plt.hist(del_infs_np(theta_sample_init[:,1]), density=True, bins="rice", color="red", label=r"Initial prior in $\beta$", alpha=0.4)
plt.hist(del_infs_np(theta_sample_prior[:,1][theta_sample_prior[:,1]<50]), bins= 1000, density=True, label=r"Fitted prior", alpha=0.4)
plt.legend()
plt.xlim(0,10)


target_w1 = lambda w1: np.linalg.norm(w1, axis=-1)
target_b1 = lambda b1: b1
target_w2_1 = lambda w2: np.max(1/w2 -1e10*(w2>0), axis=-1)
target_w2_2 = lambda w2: np.min(1/w2 +1e10*(w2<0), axis=-1)
plt.figure(figsize=(4, 3))
plt.plot(range_MI, target_w1(np.array(weights_hist['w1']).squeeze()), label=r'$\|\mathbf{w}_1\|_2$', color='#f0a94c')
plt.plot(range_MI, target_b1(weights_hist['b1']), label=r'$b_1$')
plt.plot(range_MI, target_w2_1(np.array(weights_hist['w2']).squeeze()), label=r'$w_{2l}^{-1}$', color='#8D725B')
plt.plot(range_MI, target_w2_2(np.array(weights_hist['w2']).squeeze()), label=r'$w_{2j}^{-1}$', color='#7D1B65')
plt.plot(range_MI, np.ones_like(range_MI)*2, '--', color='#7D1B65', alpha=0.8, linewidth=1)
plt.plot(range_MI, np.array(weights_hist['b2']).squeeze(), color='green')
# plt.plot(range_MI, np.ones_like(range_MI)*-0, '--', color='#8D725B', alpha=0.8, linewidth=1)
# plt.grid(alpha=0.35)
plt.xlabel('Epochs')
plt.grid(alpha=0.35)
plt.legend(loc='upper left')




######


N = 50  # non-degenerate
i = 2
file_path = os.path.join(int_jeffreys_path, f'model_J_{i}')
with open(file_path, 'rb') as file:
    model_J = pickle.load(file)
theta_J = model_J['logs']['post'][N]

x1 = theta_post_nocstr[:, 0]
y1 = theta_post_nocstr[:, 1]
# y1 = np.clip(y1, a_min=None, a_max=2)
x2 = theta_J[:, 0]
y2 = theta_J[:, 1]

kde_x1 = gaussian_kde(x1)
kde_y1 = gaussian_kde(y1)
kde_x2 = gaussian_kde(x2)
kde_y2 = gaussian_kde(y2)

# Create figure and gridspec layout
fig = plt.figure(figsize=(6.5, 5))
gs = gridspec.GridSpec(4, 5)

# Main scatter plot
ax_main = fig.add_subplot(gs[1:4, 0:3])
ax_main.scatter(x1, y1, alpha=0.5, s=20, marker='.', label='VARP posterior',zorder=2, color='red')
ax_main.scatter(x2, y2, alpha=0.5, s=20, marker='.', label='AJ posterior',zorder=1, color='blue')
ax_main.set_xlabel(r'$\alpha$')
ax_main.set_ylabel(r'$\beta$')
ax_main.legend(loc='upper left', fontsize=11)
ax_main.grid()

# Marginal histogram / KDE for alpha
ax_x_hist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
ax_x_hist.hist(x1, bins='rice', alpha=0.4, label='VARP histogram',density=True, color='red')
ax_x_hist.hist(x2, bins='rice', alpha=0.4, label='AJ histogram',density=True,color='blue')
x_vals = np.linspace(min(x1.min(), x2.min()), max(x1.max(), x2.max()), 100)
ax_x_hist.plot(x_vals, kde_x1(x_vals), color='red', lw=2)
ax_x_hist.plot(x_vals, kde_x2(x_vals), color='blue', lw=2)
ax_x_hist.set_ylabel(r'Marginal $\alpha$')
ax_x_hist.tick_params(axis='x', labelbottom=False)
ax_x_hist.legend()
ax_x_hist.grid()

# Marginal histogram / KDE for beta
ax_y_hist = fig.add_subplot(gs[1:4, 3:5], sharey=ax_main)
ax_y_hist.hist(y1, bins='rice', orientation='horizontal', alpha=0.4, label='VARP histogram',density=True, color='red')
ax_y_hist.hist(y2, bins='rice', orientation='horizontal', alpha=0.4, label='AJ histogram',density=True, color='blue')
y_vals = np.linspace(min(y1.min(), y2.min()), max(y1.max(), y2.max()), 100)
ax_y_hist.plot(kde_y1(y_vals), y_vals, color='red', lw=2)
ax_y_hist.plot(kde_y2(y_vals), y_vals, color='blue', lw=2)
ax_y_hist.set_xlabel(r'Marginal $\beta$')
ax_y_hist.tick_params(axis='y', labelleft=False)
ax_y_hist.legend()
ax_y_hist.grid()

plt.tight_layout()
plt.show()



# curves

fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(111)

plot_frag_cred(theta_post_nocstr, ax, 'red', label='VA-RP')
plot_frag_cred(theta_J, ax, 'blue', label='AJ')


















