
import os

# N1 = os.environ['SLURM_NTASKS']
# print(N1)
# N = os.environ['N_RUN']
# gamma = os.environ['GAMMA']
import sys

try :
    N = int(sys.argv[1])
    gamma = float(sys.argv[2])/10
    print(gamma)
except :
    N = os.environ['N_RUN']
    gamma = float(os.environ['GAMMA'])/10
    print('N is os.environ[SLURM_NTASKS]')
    print('GAMMA is os.environ[GAMMA]')




import pickle
from tqdm import tqdm
import numpy as np
from numba import jit
import scipy.special as spc
import matplotlib.pyplot as plt
import scipy.stats as stat
try :
    from scipy.integrate import simpson
except :
    from scipy.integrate import simps as simpson

import inspect
directory = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])) # get script's path
# os.chdir(directory)
ch_directory = os.path.join(directory, r'./../')
os.chdir(ch_directory)

sys.path.append(ch_directory)
sys.path.append(directory)

from bayes_frag import plot_functions
from bayes_frag.reference_curves import Reference_curve, Reference_saved_MLE, Reference_known_MLE, Ref_main
from bayes_frag.model import Model
from bayes_frag.data import Data, Data_toy
from bayes_frag.extract_saved_fisher import get_fisher_IM
from bayes_frag import stat_functions as stat_f
from bayes_frag.planning.as_model import Model_AS, Model_AS_1run, Adaptative_Acquisition, Minimizer
from bayes_frag import config

path = r''

num_mean = 10


# # # mod and configs:
"""
1. ASG_no_lin (10**4) 100x (1-26) : PGA - ASG - 15
                                    sa_ - ASG - 14
2. ASG_no_lin (10**4) 400x (-20-) : PGA - ASG - 19
                                    sa_ - ASG - 18
3. toy         (g0.5) 100x (1-26) : sa_5hz - toy - 30
   toy_cut 0.9 (g0.5) 100x (1-26) : sa_5hz - toy_cut - 32
   toy_cut 0.8 (g0.5) 100x (1-26) : sa_5hz - toy_cut - 34
4. ASG_no_lin (80000) 100x (1-26) : PGA - ASG - 41
                                    PSA - ASG - 42
5. ASG_no_lin (10**5) ...
"""

IM = "PGA"
mod_name = 'ASG'
num_run = 41
def m_data() :
    # if 'ref' in
    if mod_name=='ASG' :
        data = Data(IM, csv_path='Res_ASG_SA_PGA_RL_RNL_80000.csv', quantile_C=0.9, name_inte='rot_nlin', shuffle=True)
    elif mod_name=='ASG_lin':
        data = Data(IM, csv_path='Res_ASG_Lin.csv', quantile_C=0.9, name_inte='rot', shuffle=True)
    elif mod_name=='toy' :
        data = Data_toy(2, 0, 3, 0.3)
        data._set_a_tab(max_a=np.exp(2), num_a=200)
    elif mod_name=='toy_cut' :
        data = Data_toy(2, 0, 3, 0.3)
        q = data.get_curve_opt_threshold(0.9)
        ids_q = data.A<q 
        print('prop seismse conserves: ', ids_q.mean())
        data.Z = data.Z[ids_q][:,np.newaxis] + 0
        data.Y = data.Y[ids_q][:,np.newaxis] + 0
        data.A = data.A[ids_q][:,np.newaxis] + 0
        data._set_a_tab(max_a=np.exp(2), num_a=200)
        # data._set_a_tab(max_a=data.A.max(), num_a=200)
    else :
        data = Data(IM, shuffle=True)
        # data = Data(IM, csv_path='Res_ASG_Lin.csv', C=1.6, name_inte='rot', shuffle=True)
    # data = Data(IM, csv_path='Res_ASG.csv', quantile_C=0.9, name_inte='rot_nlin', shuffle=True)
    return data
data = m_data()

if not 'toy' in mod_name :
    biais = True
    if mod_name == 'ASG' :
        ref = Reference_saved_MLE(data, os.path.join(config.data_path, 'ref_MLE_ASG_80000_{}'.format(IM)))
    else :
        ref = Reference_saved_MLE(data, os.path.join(config.data_path, 'ref_MLE_{}_{}'.format(mod_name, IM)))
else :
    biais= None
    ref = Reference_known_MLE(data, np.array([data.alpha_star, data.beta_star]))
# ref_curve = ref.curve_MLE

if IM=="PGA" :
    # num_clust = 15
    num_clust = 25 # 80000
elif IM=="sa_5hz" :
    num_clust = 20
if mod_name=='ASG_lin' :
    num_clust = 35 # a verifier
 
if 'toy' in mod_name :
    # ref_curve = ref.curve_MLE
    ref._compute_MC_data_tab()
else : 
    ref._compute_empirical_curves(num_clust) 
    ref._compute_MC_data_tab()

    # ref_curve = ref.curve_MC_data_tabs[0]


theta_n_J = []
theta_n_Ja = []
theta_s_Ja = []
theta_s_J = []
A_n_J = []
A_n_Ja = []
A_s_Ja = []
A_s_J = []
S_n_J = []
S_n_Ja = []
S_s_Ja = []
S_s_J = []

n_est = 5000

# ks_ = np.arange(1,50)*5
ks_ = np.arange(1,26)*10
# ks_ = np.array([20])
transfo_i = lambda i: i


"""
    4 modeles :
        "model_noSUR_J_{}" :      points generes aleatoirement, prior=J
        "model_noSUR_J_adpt_{}" : points generes aleatoirement, prior=J_adap
        "model_SUR_adpt_{}" :     points generes par SUR,       prior=J_adap
        "model_SUR_origJ_{}" :    points du SUR et J_a, cfrag par prior=J
"""

disable_tqdm=True

save_parent = r'/mnt/beegfs/workdir/antoine.van-biesbroeck/results_SUR_bin_alpha_impact/'
save_child = r'M-G{}-{}_SUR_bin_{}-(1-26)_{}'.format(gamma, mod_name, IM, num_run)
path = os.path.join(save_parent, save_child)
assert os.path.exists(path), "path donnot exist {}".format(path)

print('simulations extraction')
for k in tqdm(range(num_mean), disable=disable_tqdm) :
    # f_J = open(os.path.join(file_paths, "exec_probits_task_newC_{}_r8/model_bin_J".format(k+1)), 'rb')
    # f_J = open(os.path.join(file_paths, "exec_probits_task_{}_r1/model_bin_J".format(k+1)), 'rb')
    # f_J = open(os.path.join(file_paths, "2022-01-12-est_ASG", "exec_probits_task_ASG_{}_r1/model_bin_J".format(k+1)), 'rb')
    #try :
    KK = k+1
    while KK<=400 :
        try :
            f_n_J = open(os.path.join(path, "model_noSUR_J_{}".format(KK)), 'rb')
            f_n_Ja = open(os.path.join(path, "model_noSUR_J_adpt_{}".format(KK)), 'rb')
            f_s_Ja = open(os.path.join(path, "model_SUR_adpt_{}".format(KK)), 'rb')
            f_s_J = open(os.path.join(path, "model_SUR_origJ_{}".format(KK)), 'rb')
        except :
            KK += 100
        else :
            break
    # f_J = open(os.path.join(file_paths, "2023-02-17-est_ASG_Lin_PGA", "exec_probits_task_ASG_Lin_PGA_{}_r1/model_bin_J".format(k+1)), 'rb')
    model_n_J = pickle.load(f_n_J)
    model_n_Ja = pickle.load(f_n_Ja)
    model_s_Ja = pickle.load(f_s_Ja)
    model_s_J = pickle.load(f_s_J)
    #
    f_n_J.close()
    f_n_Ja.close()
    f_s_Ja.close()
    f_s_J.close()
    #
    theta_n_J.append(np.array(list(model_n_J['logs']['post'].values())))
    A_n_J.append(model_n_J['A'])
    S_n_J.append(model_n_J['S'])
    # theta_J[:,k] = np.array(list(model_J['logs']['post'].values()))
    
    # theta_MLE[:,k] = np.array(list(model_J['logs']['MLE'].values()))
    # f_st = open(os.path.join(file_paths, "2022-01-12-est_ASG", "exec_probits_task_ASG_{}_r1/model_bin_st".format(k+1)), 'rb')
    # f_st = open(os.path.join(file_paths, "exec_probits_task_newC_{}_r8/model_bin_st".format(k+1)), 'rb')
    # f_st = open(os.path.join(file_paths, "exec_probits_task_{}_r1/model_bin_st".format(k+1)), 'rb')
    # f_st = open(os.path.join(path, "model_SUR_{}".format(k+1)), 'rb')
    # f_st = open(os.path.join(file_paths, "2023-02-17-est_ASG_Lin_PGA", "exec_probits_task_ASG_Lin_PGA_{}_r1/model_bin_st".format(k+1)), 'rb')
    # model_st = pickle.load(f_st)
    theta_n_Ja.append(np.array(list(model_n_Ja['logs']['post'].values())))
    # A_n_Ja.append(model_n_Ja['A'])
    # S_n_Ja.append(model_n_Ja['S'])
    # assert np.all(np.array(A_n_J)==np.array(A_n_Ja)) and np.all(np.array(S_n_Ja)==np.array(S_n_J))
    # theta_st[:,k] = np.array(list(model_st['logs']['post'].values()))
    theta_s_Ja.append(np.array(list(model_s_Ja['logs']['post'].values())))
    # A_s_Ja.append(model_s_Ja['A'])
    # S_s_Ja.append(model_s_Ja['S'])
    #
    theta_s_J.append(np.array(list(model_s_J['logs']['post'].values())))
    A_s_J.append(model_s_J['A'])
    S_s_J.append(model_s_J['S'])
    assert np.all(np.array(A_s_J)==np.array(A_s_Ja)) and np.all(np.array(S_s_Ja)==np.array(S_s_J))


del model_n_J 
del model_n_Ja
del model_s_Ja
del model_s_J
    # except :
    #     pass

# to_keep = np.arange(1,25)*2+1
to_keep = np.arange(num_mean)

print(np.array(theta_n_J).shape)
theta_n_J = np.transpose(np.array(theta_n_J)[to_keep], axes=[1,0,2,3])
theta_n_Ja = np.transpose(np.array(theta_n_Ja)[to_keep], axes=[1,0,2,3])
theta_s_Ja = np.transpose(np.array(theta_s_Ja)[to_keep], axes=[1,0,2,3])
theta_s_J = np.transpose(np.array(theta_s_J)[to_keep], axes=[1,0,2,3])
A_n_J = np.array(A_n_J).squeeze()[to_keep]
# A_n_Ja = np.array(A_n_Ja).squeeze()
# A_s_Ja = np.array(A_s_Ja).squeeze()
A_s_J = np.array(A_s_J).squeeze()[to_keep]
S_n_J = np.array(S_n_J).squeeze()[to_keep]
# S_n_Ja = np.array(S_n_Ja).squeeze()
# S_s_Ja = np.array(S_s_Ja).squeeze()
S_s_J = np.array(S_s_J).squeeze()[to_keep]
# kmax = min(A_n_J.shape[1], A_n_Ja.shape[1], A_s_J.shape[1], A_s_Ja.shape[1])
# k_ot = max(A_n_J.shape[1], A_n_Ja.shape[1], A_s_J.shape[1], A_s_Ja.shape[1])
kmax = min(A_n_J.shape[1], A_s_J.shape[1])
k_ot = max(A_n_J.shape[1], A_s_J.shape[1])
# assert kmax==k_ot, 'kmax!=kot : {} / {}'.format(kmax,k_ot)
A_n_J = A_n_J[:,:kmax]+0
# A_n_Ja = A_n_Ja[:,:kmax]+0
A_s_J = A_s_J[:,:kmax]+0
# A_s_Ja = A_s_Ja[:,:kmax]+0
S_n_J = S_n_J[:,:kmax]+0
# S_n_Ja = S_n_Ja[:,:kmax]+0
S_s_J = S_s_J[:,:kmax]+0
# S_s_Ja = S_s_Ja[:,:kmax]+0

# ks_ = (np.arange(1,25)+1)*10
ks_=ks_+0

assert theta_n_J.shape == (len(ks_),num_mean,n_est,2)


a_qu1 = ref.probit_ref.get_curve_opt_threshold(10**-2)
data_pga_asg = Data('PGA', csv_path='Res_ASG_SA_PGA_RL_RNL_80000.csv', quantile_C=0.9, name_inte='rot_nlin', shuffle=True)
ref_pga_asg = Reference_saved_MLE(data_pga_asg, os.path.join(config.data_path, 'ref_MLE_ASG_80000_PGA'))
kmeans = ref_pga_asg._compute_empirical_curves(20)
a_asg_qu2 = ref_pga_asg.a_tab_MC[-1] #get_curve_opt_threshold(ref.theta_MLE[0], ref.theta_MLE[1], 10**-2)
qu2 = ref_pga_asg.probit_ref(a_asg_qu2)
a_qu2 = ref.probit_ref.get_curve_opt_threshold(qu2)
data._set_a_tab(min_a=a_qu1, max_a=a_qu2)
ref._compute_MLE_curve()
ref._compute_MC_data_tab()
if 'toy' in mod_name :
    ref_curve = ref.curve_MLE
else : 
    ref_curve = ref.curve_MC_data_tabs[0]


##

def errors_L2A(theta, ks_, a_tab=data.a_tab, f_A_tab=data.f_A_tab, ref_curve=ref_curve, conf=0.05, transfo_i=lambda x:x):
    curves_post = np.zeros((len(ks_), len(a_tab), num_mean, n_est))
    err_quad_cond= np.zeros((len(ks_), num_mean))
    err_conf_cond = np.zeros((len(ks_), num_mean))
    err_med_cond = np.zeros((len(ks_), num_mean))
    for i,k in enumerate(ks_) :
        # print(i)
        idi = transfo_i(i)

        for n in range(num_mean):
            curves_post = 1/2 + 1/2*spc.erf(np.log(a_tab[...,np.newaxis]/theta[i,n,np.newaxis,:,0])/theta[i,n,np.newaxis,:,1])
            # curves_post = curve_3var(a_tab, theta[idi,n]).T
            # err_conf_cond[i,n] = simpson((np.quantile(curves_post, 1-conf/2, axis=-1)- np.quantile(curves_post, conf/2, axis=-1))**2* f_A_tab, a_tab)
            err_med_cond[i,n] = simpson((np.median(curves_post, axis=-1)-ref_curve)**2* f_A_tab, a_tab)

            # err_quad_cond[i,n] = simpson((curves_post-ref_curve[:,np.newaxis])**2 *f_A_tab[:,np.newaxis], a_tab, axis=0).mean(axis=-1)

    #         curves_post[i,:,n,:] = 1/2 + 1/2*spc.erf(np.log(a_tab[...,np.newaxis]/theta[i,n,np.newaxis,:,0])/theta[i,n,np.newaxis,:,1])
    # # curves_post = curves_post.reshape(len(ks_), len(self.data.a_tab), -1)
    #         err_conf[i] += simpson((np.quantile(curves_post, 1-conf/2, axis=-1)- np.quantile(curves_post, conf/2, axis=-1))**2 *f_A_tab[np.newaxis,:,np.newaxis], a_tab, axis=1)
    #         err_med_cond[i] += simpson((np.median(curves_post, axis=-1)-ref_curve[np.newaxis,:,np.newaxis])**2 *f_A_tab[np.newaxis,:,np.newaxis], a_tab, axis=1)
    # # err_conf_cond = simpson((np.quantile(curves_post, 1-conf/2, axis=-1)- np.quantile(curves_post, conf/2, axis=-1))**2 *f_A_tab[np.newaxis,:,np.newaxis], a_tab, axis=1)
    # # err_med_cond = simpson((np.median(curves_post, axis=-1)-ref_curve[np.newaxis,:,np.newaxis])**2 *f_A_tab[np.newaxis,:,np.newaxis], a_tab, axis=1)
    # print('copute err_quad...')
    # err_quad_cond = simpson((curves_post-ref_curve[np.newaxis,:,np.newaxis,np.newaxis])**2 *f_A_tab[np.newaxis,:,np.newaxis,np.newaxis], a_tab, axis=1).mean(axis=-1)
    # print('done')

    err_conf = err_conf_cond.mean(-1)
    err_med = err_med_cond.mean(-1)
    # err_conf /= num_mean
    # err_med /= num_mean
    err_quad = err_quad_cond.mean(-1)

    return {'err_conf':err_conf, 'err_med':err_med, 'err_quad':err_quad, 'err_med_cond':err_med_cond}

def errors(theta, ks_, a_tab=data.a_tab, ref_curve=ref_curve, conf=0.05, transfo_i=lambda x:x):
    return errors_L2A(theta, ks_, a_tab, np.ones_like(a_tab), ref_curve, conf, transfo_i)






def quant_quant(theta, ks_, f_A_tab=data.f_A_tab, a_tab=data.a_tab, conf=0.05, varia_conf=0.05, s=0, transfo_i=lambda x:x, to_div=data.a_tab.max()) :
    curves_post = np.zeros((len(ks_), len(a_tab), num_mean, n_est))
    err_conf_cond = np.zeros((len(ks_), num_mean))
    quant1_conf_zone = np.zeros((len(ks_)))
    quant2_conf_zone = np.zeros((len(ks_)))
    var_conf_zone = np.zeros((len(ks_)))
    mean_conf_zone = np.zeros((len(ks_)))
    for i,k in enumerate(ks_) :
        # print(i)
        idi = transfo_i(i)
        for n in range(num_mean):
            curves_post = 1/2 + 1/2*spc.erf(np.log(a_tab[...,np.newaxis]/theta[i,n,np.newaxis,:,0])/theta[i,n,np.newaxis,:,1])
            # curves_post = curve_3var(a_tab, theta[idi,n]).T
            err_conf_cond[i,n] = simpson((np.quantile(curves_post, 1-conf/2, axis=-1)- np.quantile(curves_post, conf/2, axis=-1))**2* f_A_tab, a_tab)
        # 1. variance de err_conf_cond
        var_conf_zone[i] = err_conf_cond[i].var()
        # 2. quantiles de err_conf_cond
        # print(err_conf_cond[i][err_conf_cond[i]>0.05].shape, err_conf_cond[i].min())
        try :
            quant1_conf_zone[i] = np.quantile(err_conf_cond[i][err_conf_cond[i]>s], 1-varia_conf/2)
            quant2_conf_zone[i] = np.quantile(err_conf_cond[i][err_conf_cond[i]>s], varia_conf/2)
            mean_conf_zone[i] = err_conf_cond[i][err_conf_cond[i]>s].mean()
        except :
            quant1_conf_zone[i] = np.quantile(err_conf_cond[i], 1-varia_conf/2)
            quant2_conf_zone[i] = np.quantile(err_conf_cond[i], varia_conf/2)
            mean_conf_zone[i] = err_conf_cond[i][err_conf_cond[i]>s].mean()
    return {"quant1_conf_zone":quant1_conf_zone/to_div, "quant2_conf_zone":quant2_conf_zone/to_div, "var_conf_zone":var_conf_zone/to_div, 'mean_conf_zone':mean_conf_zone/to_div}


def quad_err_tubes(theta, ks_, f_A_tab=data.f_A_tab, a_tab=data.a_tab, ref_curve=ref_curve, variaconf=0.05, transfo_i=lambda x:x, to_div=data.a_tab.max()) :
    err_quad_cond= np.zeros((len(ks_), num_mean))
    for i,k in enumerate(ks_) :
        # print(i)
        idi = transfo_i(i)
        for n in range(num_mean):
            # curves_post = curve_3var(a_tab, theta[idi,n]).T
            curves_post = 1/2 + 1/2*spc.erf(np.log(a_tab[...,np.newaxis]/theta[i,n,np.newaxis,:,0])/theta[i,n,np.newaxis,:,1])
            err_quad_cond[i,n] = simpson((curves_post-ref_curve[:,np.newaxis])**2 *f_A_tab[:,np.newaxis], a_tab, axis=0).mean(axis=-1)
    return {'quant1_quad':np.quantile(err_quad_cond, 1-variaconf/2, axis=-1)/to_div, 'quant2_quad':np.quantile(err_quad_cond, variaconf/2, axis=-1)/to_div, 'mean_quad':err_quad_cond.mean(-1)/to_div, 'quad_cond':err_quad_cond/to_div}



def cred_interv_ok(theta, ks_, a_tab=data.a_tab, ref_curve=ref_curve, conf=0.05, transfo_i=lambda x:x) :
    # il faut compter combien de fois que Pref tombe dans l'intervalle de cred
    curves_post = np.zeros((len(ks_), len(a_tab), num_mean, n_est))
    # err_conf_cond = np.zeros((len(ks_), num_mean))
    credib_cond = np.zeros((len(ks_), num_mean))
    for i,k in enumerate(ks_) :
        # print(i)
        idi = transfo_i(i)
        for n in range(num_mean):
            # curves_post = curve_3var(a_tab, theta[idi,n]).T
            curves_post = 1/2 + 1/2*spc.erf(np.log(a_tab[...,np.newaxis]/theta[i,n,np.newaxis,:,0])/theta[i,n,np.newaxis,:,1])
    # for i,k in enumerate(ks_) :
    #     print(i)
    #     idi = transfo_i(i)
    #     curves_post[:,:,:] = stat.norm.cdf((theta[idi,:,np.newaxis,:,2]*np.log(a_tab[np.newaxis,:,np.newaxis]) + theta[idi,:,np.newaxis,:,0] )/theta[idi,:,np.newaxis,:,1] )
            credib_cond[i,n] = np.sum(1*(np.quantile(curves_post, 1-conf/2, axis=-1)>=ref_curve)*(np.quantile(curves_post, conf/2, axis=-1)<=ref_curve) /a_tab.max()*(a_tab[1]-a_tab[0]))
    return credib_cond.mean(-1), credib_cond


def proba_degen(A_t, S_t) :
    probs = np.ones(A_t.shape[1]-1)
    for k in range(1, A_t.shape[1]) :
        # print(A_t[:,:k][S_t[:,:k]==0].shape)
        probs[k-1] = (( (A_t[:,:k]*(S_t[:,:k]==0)).max(axis=1) < (A_t[:,:k] + 10**4*(S_t[:,:k]==0)).min(axis=1) ) | ( (A_t[:,:k]*(S_t[:,:k]==1)).max(axis=1) < (A_t[:,:k] + 10**4*(S_t[:,:k]==1)).min(axis=1) ) ).mean()
    return probs



def Pfmeds(theta, ks_=ks_, a_tab=data.a_tab, transfo_i=lambda x:x) :
    Pfmeds = np.zeros((len(ks_), num_mean, len(a_tab)))
    for i,k in enumerate(ks_) :
        idi = transfo_i(i)
        for n in range(num_mean):
            # curves_post = curve_3var(a_tab, theta[idi,n]).T
            curves_post = 1/2 + 1/2*spc.erf(np.log(a_tab[...,np.newaxis]/theta[i,n,np.newaxis,:,0])/theta[i,n,np.newaxis,:,1]) #shape (a,MC)
            Pfmeds[i,n] = np.median(curves_post, axis=-1)
    return Pfmeds


path_save = path


import errors_fig_linear as err_fig 

err_dict = {}

def task_1() :
    # run errors et plot med
    to_div = data.a_tab.max()-data.a_tab.min()
    err_dict['errors_L2_n_J'] = errors(theta_n_J, ks_, transfo_i=transfo_i)
    err_dict['errors_L2_n_Ja'] = errors(theta_n_Ja, ks_, transfo_i=transfo_i)
    err_dict['errors_L2_s_J'] = errors(theta_s_J, ks_, transfo_i=transfo_i)
    err_dict['errors_L2_s_Ja'] = errors(theta_s_Ja, ks_, transfo_i=transfo_i)
    if not biais is None :
        model_biais = simpson((ref.curve_MC_data_tabs[0]-ref.curve_MLE)**2, data.a_tab)/to_div #/data.a_tab.max()
    print("median error tube computation (L2U)")
    fig1 = plt.figure(11, figsize = (6,4))
    fig1.clf()
    ax1 = fig1.add_subplot(111)
    err_fig.plots_med_tube_L2U(err_dict['errors_L2_n_J'], ks_, ax1, 'J', color='blue', a_max=to_div)
    err_fig.plots_med_tube_L2U(err_dict['errors_L2_n_Ja'], ks_, ax1, r'$J_\gamma,\ \gamma={}$'.format(gamma), color='orange', a_max=to_div)
    if not biais is None :
        ax1.axhline(y=model_biais, color='black', linestyle='--', alpha=0.7, label='model bias')
    ax1.set_title('standard')
    ax1.set_ylabel(r'square bias $||m(P_{\theta})-P_{ref}||_{L^2}^2]$')
    ax1.legend()
    ax1.grid(alpha=0.35)
    fig2 = plt.figure(110, figsize = (6,4))
    fig2.clf()
    ax2 = fig2.add_subplot(111)
    err_fig.plots_med_tube_L2U(err_dict['errors_L2_s_J'], ks_, ax2, 'J', color='blue', a_max=to_div)
    err_fig.plots_med_tube_L2U(err_dict['errors_L2_s_Ja'], ks_, ax2, r'$J_\gamma,\ \gamma={}$'.format(gamma), color='orange', a_max=to_div)
    if not biais is None :
        ax2.axhline(y=model_biais, color='black', linestyle='--', alpha=0.7, label='model bias')
    ax2.set_title('with P.E.')
    ax2.set_ylabel(r'square bias $||m(P_{\theta})-P_{ref}||_{L^2}^2]$')
    ax2.legend()
    ax2.grid(alpha=0.35)

    xmin1, xmax1, ymin1, ymax1 = ax1.axis()
    xmin2, xmax2, ymin2, ymax2 = ax2.axis()
    ymin, ymax = min(ymin1,ymin2), max(ymax1,ymax2)
    ax1.set(ylim=(ymin, ymax))
    ax2.set(ylim=(ymin, ymax))
    fig1.savefig(os.path.join(path_save,"figure_nref_newL2_{}_noSUR.pdf".format(11)), format='pdf')
    fig2.savefig(os.path.join(path_save,"figure_nref_newL2_{}_SUR.pdf".format(11)), format='pdf')


def task_2() :
    # run quant quant and plot var conf
    to_div = data.a_tab.max()-data.a_tab.min()
    err_dict['variations_conf_n_J_U'] = quant_quant(theta_n_J, ks_, f_A_tab=np.ones_like(data.a_tab), s=0.01, transfo_i=transfo_i, to_div=to_div)
    err_dict['variations_conf_n_Ja_U'] = quant_quant(theta_n_Ja, ks_, f_A_tab=np.ones_like(data.a_tab), s=0.01, transfo_i=transfo_i, to_div=to_div)
    err_dict['variations_conf_s_J_U'] = quant_quant(theta_s_J, ks_, f_A_tab=np.ones_like(data.a_tab), s=0.01, transfo_i=transfo_i, to_div=to_div)
    err_dict['variations_conf_s_Ja_U'] = quant_quant(theta_s_Ja, ks_, f_A_tab=np.ones_like(data.a_tab), s=0.01, transfo_i=transfo_i, to_div=to_div)
    
    fig1 = plt.figure(12, figsize = (6,4))
    figtemp = plt.figure()
    fig1.clf()
    axes1 = [fig1.add_subplot(111), figtemp.add_subplot(111)]
    # plot_variations(variations_conf_MLE_U, ks_, 'MLE', L2set='' , axes=axes, color='red')
    err_fig.plot_variations(err_dict['variations_conf_n_J_U'], ks_, 'J', L2set='', axes=axes1, color='blue')
    err_fig.plot_variations(err_dict['variations_conf_n_Ja_U'], ks_, r'$J_\gamma,\ \gamma={}$'.format(gamma), L2set='', axes=axes1, color='orange')
    axes1[0].set_title('standard')
    axes1[0].set_ylabel(r'credibility width $\|q_{r/2}-q_{1-r/2}\|_{L^2}^2$,')
    axes1[0].grid(alpha=0.35)
    plt.close(figtemp)

    fig2 = plt.figure(120, figsize = (6,4))
    figtemp = plt.figure()
    fig2.clf()
    axes2 = [fig2.add_subplot(111), figtemp.add_subplot(111)]
    # plot_variations(variations_conf_MLE_U, ks_, 'MLE', L2set='' , axes=axes, color='red')
    err_fig.plot_variations(err_dict['variations_conf_s_J_U'], ks_, 'J', L2set='', axes=axes2, color='blue')
    err_fig.plot_variations(err_dict['variations_conf_s_Ja_U'], ks_, r'$J_\gamma,\ \gamma={}$'.format(gamma), L2set='', axes=axes2, color='orange')
    axes2[0].set_title('with P.E.')
    axes2[0].set_ylabel(r'credibility width $\|q_{r/2}-q_{1-r/2}\|_{L^2}^2$,')
    axes2[0].grid(alpha=0.35)
    plt.close(figtemp)

    xmin1, xmax1, ymin1, ymax1 = axes1[0].axis()
    xmin2, xmax2, ymin2, ymax2 = axes2[0].axis()
    ymin, ymax = min(ymin1,ymin2), max(ymax1,ymax2)
    axes1[0].set(ylim=(ymin, ymax))
    axes2[0].set(ylim=(ymin, ymax))
    fig1.savefig(os.path.join(path_save,"figure_nref_newL2_{}_noSUR.pdf".format(12)), format='pdf')
    fig2.savefig(os.path.join(path_save,"figure_nref_newL2_{}_SUR.pdf".format(12)), format='pdf')


def task_3() :
    # run quad tube and plot quad
    print('quad tube computations (L2U)...')
    # quad_tube_MLE_U = quad_err_tubes(theta_MLE, ks_, f_A_tab=np.ones_like(data.a_tab))
    to_div = data.a_tab.max()-data.a_tab.min()
    err_dict['quad_tube_n_J_U'] = quad_err_tubes(theta_n_J, ks_, f_A_tab=np.ones_like(data.a_tab), transfo_i=transfo_i, to_div=to_div)
    err_dict['quad_tube_n_Ja_U'] = quad_err_tubes(theta_n_Ja, ks_, f_A_tab=np.ones_like(data.a_tab), transfo_i=transfo_i, to_div=to_div)
    err_dict['quad_tube_s_J_U'] = quad_err_tubes(theta_s_J, ks_, f_A_tab=np.ones_like(data.a_tab), transfo_i=transfo_i, to_div=to_div)
    err_dict['quad_tube_s_Ja_U'] = quad_err_tubes(theta_s_Ja, ks_, f_A_tab=np.ones_like(data.a_tab), transfo_i=transfo_i, to_div=to_div)
    
    fig1 = plt.figure(10, figsize = (6,4))
    fig1.clf()
    ax1 = fig1.add_subplot(111)
    # plot_quad_tubes(quad_tube_MLE_U, ks_, 'MLE', L2set='', ax=ax, color='red')
    err_fig.plot_quad_tubes(err_dict['quad_tube_n_J_U'], ks_, 'J', L2set='', ax=ax1, color='blue')
    err_fig.plot_quad_tubes(err_dict['quad_tube_n_Ja_U'], ks_, r'$J_\gamma,\ \gamma={}$'.format(gamma), L2set='', ax=ax1, color='orange')
    ax1.set_title(r'standard')
    ax1.set_ylabel(r'quadratic error')
    ax1.grid(alpha=0.35)

    fig2 = plt.figure(100, figsize = (6,4))
    fig2.clf()
    ax2 = fig2.add_subplot(111)
    # plot_quad_tubes(quad_tube_MLE_U, ks_, 'MLE', L2set='', ax=ax, color='red')
    err_fig.plot_quad_tubes(err_dict['quad_tube_s_J_U'], ks_, 'J', L2set='', ax=ax2, color='blue')
    err_fig.plot_quad_tubes(err_dict['quad_tube_s_Ja_U'], ks_, r'$J_\gamma,\ \gamma={}$'.format(gamma), L2set='', ax=ax2, color='orange')
    ax2.set_title(r'with P.E.')
    ax2.set_ylabel(r'quadratic error')
    ax2.grid(alpha=0.35)

    xmin1, xmax1, ymin1, ymax1 = ax1.axis()
    xmin2, xmax2, ymin2, ymax2 = ax2.axis()
    ymin, ymax = min(ymin1,ymin2), max(ymax1,ymax2)
    ax1.set(ylim=(ymin, ymax))
    ax2.set(ylim=(ymin, ymax))
    fig1.savefig(os.path.join(path_save,"figure_nref_newL2_{}_noSUR.pdf".format(10)), format='pdf')
    fig2.savefig(os.path.join(path_save,"figure_nref_newL2_{}_SUR.pdf".format(10)), format='pdf')


def task_4() :
    # run cred ok and plot
    err_dict['cred_n_J'] = cred_interv_ok(theta_n_J, ks_, ref_curve=ref.curve_MLE, conf=0.05, transfo_i=lambda x:x)
    err_dict['cred_n_Ja'] = cred_interv_ok(theta_n_Ja, ks_, ref_curve=ref.curve_MLE, conf=0.05, transfo_i=lambda x:x)
    err_dict['cred_s_J'] = cred_interv_ok(theta_s_J, ks_, ref_curve=ref.curve_MLE, conf=0.05, transfo_i=lambda x:x)
    err_dict['cred_s_Ja'] = cred_interv_ok(theta_s_Ja, ks_, ref_curve=ref.curve_MLE, conf=0.05, transfo_i=lambda x:x)

    fig1 = plt.figure(13, figsize = (6,4))
    fig1.clf()
    ax1 = fig1.add_subplot(111)
        # plot_variations(variations_conf_MLE_U, ks_, 'MLE', L2set='' , axes=axes, color='green')
    err_fig.plot_cred_ok(err_dict['cred_n_J'], ks_, 'J', ax=ax1, color='blue')
    err_fig.plot_cred_ok(err_dict['cred_n_Ja'], ks_, r'$J_\gamma,\ \gamma={}$'.format(gamma), ax=ax1, color='orange')
    ax1.set_title(r'standard')
    ax1.set_ylabel(r'coverage probability')
    ax1.grid(alpha=0.35)

    fig2 = plt.figure(130, figsize = (6,4))
    fig2.clf()
    ax2 = fig2.add_subplot(111)
        # plot_variations(variations_conf_MLE_U, ks_, 'MLE', L2set='' , axes=axes, color='green')
    err_fig.plot_cred_ok(err_dict['cred_s_J'], ks_, 'J', ax=ax2, color='blue')
    err_fig.plot_cred_ok(err_dict['cred_s_Ja'], ks_, r'$J_\gamma,\ \gamma={}$'.format(gamma), ax=ax2, color='orange')
    ax2.set_title(r'with P.E.')
    ax2.set_ylabel(r'coverage probability')
    ax2.grid(alpha=0.35)

    xmin1, xmax1, ymin1, ymax1 = ax1.axis()
    xmin2, xmax2, ymin2, ymax2 = ax2.axis()
    ymin, ymax = min(ymin1,ymin2), max(ymax1,ymax2)
    ax1.set(ylim=(ymin, ymax))
    ax2.set(ylim=(ymin, ymax))
    fig1.savefig(os.path.join(path_save,"figure_nref_newL2_{}_noSUR.pdf".format(13)), format='pdf')
    fig2.savefig(os.path.join(path_save,"figure_nref_newL2_{}_SUR.pdf".format(13)), format='pdf')

def task_5() :
    err_dict['degen_n_J'] = proba_degen(A_n_J, S_n_J)
    err_dict['degen_s_Ja'] = proba_degen(A_s_J, S_s_J)

    fig = plt.figure(15, figsize= (6,4))
    fig.clf()
    ax = fig.add_subplot(111)
    err_fig.plot_proba_degen(err_dict['degen_n_J'], 'standard', ax=ax, color='blue')
    err_fig.plot_proba_degen(err_dict['degen_s_Ja'], 'with P.E.', ax=ax, color='orange')
    ax.grid(alpha=0.35)

    fig.savefig(os.path.join(path_save, "figure_nref_newL2_{}.pdf".format(15)), format='pdf')


def task_6() :
    # distrib of A
    num_A_keep = 100
    err_dict['A_s'] = A_s_J[:,-num_A_keep:].flatten()
    err_dict['A_n'] = A_n_J[:,-num_A_keep:].flatten()
    try :
        fig = plt.figure(18, figsize=(6,4))
        fig.clf()
        ax = fig.add_subplot(111)
        ax.hist(np.log(A_n_J[:,-num_A_keep:].flatten()), bins=80, color='blue', density=True, alpha=0.4, label='standard')
        ax.hist(np.log(A_s_J[:,-num_A_keep:].flatten()), bins=80, color='red', density=True, alpha=0.4, label='with P.E.')
        xmin, xmax, ymin, ymax = ax.axis()
        ax.vlines(np.log(ref.theta_MLE[0]), ymin, ymax)
        ax2 = ax.twinx()
        ax2.plot(np.log(data.a_tab), ref.curve_MLE, color='darkcyan', label='ref MLE')
        ax2.grid(alpha=0.35)
        ax2.set_ylabel('Pf(a)', color='darkcyan')
        ax2.tick_params(axis='y', labelcolor='darkcyan')
        ax.set_xlim(0,np.log(10))
        ax.set_xlabel('log a')
        ax.set_title('distribution of A')
        ax.legend()
        fig.savefig(os.path.join(path_save, "figure_nref_newL2_{}.pdf".format(18)), format='pdf')
    except :
        pass

def task_7() :
    from utils_div_calc import build_index_f_div_post, build_index_f_div
    mu_a_ln = np.log(data.A).mean()
    sigma_a_ln = np.log(data.A).std()
    be = 1/2
    epsilon_beta = gamma*be #gamma=eps/be
    const_reg = 1/gamma + 1/(2-gamma)
    beta_invcdf = lambda u : (u<1/gamma/const_reg) * ( u*const_reg*gamma )**(1/gamma) + (u>1/gamma/const_reg) * ( (const_reg*u-const_reg)*(gamma-2) )**(1/(gamma-2))

    indexes_n = np.zeros((num_mean, len(ks_)))
    indexes_s = np.zeros((num_mean, len(ks_)))
    divergence_n = np.zeros((num_mean, len(ks_)))
    divergence_s = np.zeros((num_mean, len(ks_)))
    divergence_post_n = np.zeros((num_mean, len(ks_)-1))
    divergence_post_s = np.zeros((num_mean, len(ks_)-1))
    index_post_n = np.zeros((num_mean, len(ks_)-1))
    index_post_s = np.zeros((num_mean, len(ks_)-1))
    for i,k in enumerate(ks_) :
        for n in range(num_mean) :
            index_s = build_index_f_div(A_s_J[n,:k-1], S_s_J[n,:k-1], mu_a_ln, sigma_a_ln, beta_invcdf)
            indexes_s[n,i], divergence_s[n,i] = index_s(A_s_J[n,k-1], S_s_J[n,k-1])
            index_n = build_index_f_div(A_n_J[n,:k-1], S_n_J[n,:k-1], mu_a_ln, sigma_a_ln, beta_invcdf)
            indexes_n[n,i], divergence_n[n,i] = index_n(A_n_J[n,k-1], S_n_J[n,k-1])
            if i!=len(ks_)-1 :
                index_post_s_f = build_index_f_div_post(A_s_J[n,k:k] , S_s_J[n,k:k] , theta_s_Ja[i,n]) #(len(ks_),num_mean,n_est,2) #theta post|A[:k]
                index_post_s[n,i], divergence_post_s[n,i] = index_post_s_f(A_s_J[n,k+1], S_s_J[n,k+1])
                index_post_n_f = build_index_f_div_post(A_n_J[n,k:k] , S_n_J[n,k:k] , theta_n_Ja[i,n]) #(len(ks_),num_mean,n_est,2) #theta post|A[:k]
                index_post_n[n,i], divergence_post_n[n,i] = index_post_n_f(A_n_J[n,k], S_n_J[n,k])
                

    err_dict['indices_n'] = indexes_n
    err_dict['indices_s'] = indexes_s
    err_dict['divergence_n'] = divergence_n
    err_dict['divergence_s'] = divergence_s
    err_dict['index_post_n'] = index_post_n
    err_dict['index_post_s'] = index_post_s
    err_dict['divergence_post_n'] = divergence_post_n
    err_dict['divergence_post_s'] = divergence_post_s
    try :
        fig = plt.figure(16, figsize= (6,4))
        fig.clf()
        ax = fig.add_subplot(111)
        ax.fill_between(ks_, np.quantile(indexes_n, 0.05/2, axis=0), np.quantile(indexes_n, 1-0.05/2, axis=0), color='blue', alpha=0.6)
        ax.plot(ks_, np.median(indexes_n, axis=0), '--', color='blue', label='standard', alpha=0.35)
        ax.fill_between(ks_, np.quantile(indexes_s, 0.05/2, axis=0), np.quantile(indexes_s, 1-0.05/2, axis=0), color='orange', alpha=0.6)
        ax.plot(ks_, np.median(indexes_s, axis=0), '--', color='orange', label='with P.E.', alpha=0.35)
        ax.legend()
        ax.set_ylabel('I(a)')
        ax.set_xlabel('k')
        fig.savefig(os.path.join(path_save, "figure_nref_newL2_{}.pdf".format(16)), format='pdf')

        vari_n = (indexes_n[:-1] - indexes_n[1:])/indexes_n[:-1]
        vari_s = (indexes_s[:-1] - indexes_s[1:])/indexes_s[:-1]
        fig = plt.figure(17, figsize= (6,4))
        fig.clf()
        ax = fig.add_subplot(111)
        ax.fill_between(ks_, np.quantile(vari_n, 0.05/2, axis=0), np.quantile(vari_n, 1-0.05/2, axis=0), color='blue', alpha=0.6)
        ax.plot(ks_, np.median(vari_n, axis=0), '--', color='blue', label='standard', alpha=0.35)
        ax.fill_between(ks_, np.quantile(vari_s, 0.05/2, axis=0), np.quantile(vari_s, 1-0.05/2, axis=0), color='orange', alpha=0.6)
        ax.plot(ks_, np.median(vari_s, axis=0), '--', color='orange', label='with P.E.', alpha=0.35)
        ax.legend()
        ax.set_ylabel('variation of I(a)')
        ax.set_xlabel('k')
        fig.savefig(os.path.join(path_save, "figure_nref_newL2_{}.pdf".format(17)), format='pdf')
    except Exception as e:
        print(e)
        pass

def task_8() :
    # variation L2 of Pf med
    # store the entire Pf med
    err_dict['Pfmeds_n_J'] = Pfmeds(theta_n_J)
    err_dict['Pfmeds_n_Ja'] = Pfmeds(theta_n_Ja)
    err_dict['Pfmeds_s_J'] = Pfmeds(theta_s_J)
    err_dict['Pfmeds_s_Ja'] = Pfmeds(theta_s_Ja)





if int(N)==1 :
    task_1()
if int(N)==2 :
    task_2()
if int(N)==3 :
    task_3()
if int(N)==4 :
    task_4()
if int(N)==5 :
    task_5()
if int(N)==5 : #two tasks for 5
    task_6()
if int(N)==5 :
    task_7()
    task_8()


# task_4()

f = open(os.path.join(path_save, "errors_new_newL2_{}".format(N)), "wb")
pickle.dump(err_dict, f)
f.close()











