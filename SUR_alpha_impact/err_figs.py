
import numpy as np
import matplotlib.pyplot as plt
import pickle

import os
import sys

prev_dir = os.getcwd()
import inspect
directory = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe() ))[0]))
sys.path.append(os.path.join(directory, '../'))

from bayes_frag import plot_functions

def uni_ylims(ax1,ax2) :
    yi1,ya1 = ax1.get_ylim()
    yi2,ya2 = ax2.get_ylim()
    yi,ya = min(yi1,yi2), max(ya1,ya2)
    ax1.set_ylim(yi, ya)
    ax2.set_ylim(yi, ya)


def plot_proba_degen(errs, name, ax=None, color='blue') :
    if ax is None :
        fig =plt.figure()
        ax = fig.add_subplot(111)
    krange = np.arange(1, errs.shape[0]+1)
    ax.plot(krange, errs, linestyle=(0, (5, 1)), label=name, color=color)
    ax.set_xlabel('Number of data')
    ax.set_title('Degeneracy probability')
    ax.legend()
    return ax

def plot_variations(variations_dict, ks_, name, varia_conf=0.05, conf=0.05, L2set=r'(\mathbb{P}_A)', axes=None, color='blue', transfo_i=lambda x:x) :
    if axes is None :
        fig = plt.figure()
        axes = [fig.add_subplot(1,2,j+1) for j in range(2)]
    axes[0].fill_between(ks_, variations_dict['quant1_conf_zone'], variations_dict['quant2_conf_zone'], color=color, alpha=0.6)
    axes[0].plot(ks_, variations_dict['mean_conf_zone'], linestyle=(0, (5, 1)), color=color, label=name, alpha=0.35)
    # axes[0].plot(ks_, variations_dict['quant2_conf_zone'], '--', color=color)
    axes[1].plot(ks_, variations_dict['var_conf_zone'], color=color, label=name)
    axes[0].set_xlabel('Number of data')
    axes[0].set_ylabel('error variation')
    axes[0].set_title(r'{} confidence interval for the variable $E=\|q_[r/2]^[\theta]-q_[1-r/2]^[\theta]\|_[L^2{}]^2$, $r={}$'.format(1-varia_conf, L2set, conf).replace('[','{').replace(']', '}'))
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[1].set_xlabel('Number of data')
    axes[1].set_ylabel('error variation')
    axes[1].set_title(r'variance $\mathrm[Var]\,E$, with $E=\|q_[r/2]^[\alpha,\beta]-q_[1-r/2]^[\alpha,\beta]\|_[L^2{}]^2$, $r={}$'.format(L2set, conf).replace('[','{').replace(']', '}'))
    axes[1].set_yscale('log')
    axes[1].legend()
    return axes



def plot_quad_tubes(quad_dict, ks_, name, varia_conf=0.05, L2set=r'(\mathbb{P}_A)', ax=None, color='blue', transfo_i=lambda x:x) :
    if ax is None :
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.fill_between(ks_, quad_dict['quant1_quad'], quad_dict['quant2_quad'], color=color, alpha=0.6)
    ax.plot(ks_, quad_dict['mean_quad'], linestyle=(0, (5, 1)), color=color, label=name, alpha=0.35)
    # axes[0].plot(ks_, variations_dict['quant2_conf_zone'], '--', color=color)
    ax.set_xlabel('Number of data')
    ax.set_ylabel('error')
    ax.set_yscale('log')
    ax.set_title(r'${}$ confidence for'.format(varia_conf)+r' quadratic error $\mathbb{E}[||P_{\theta}-P_{ref}||_{L^2'+r'{}'.format(L2set)+r'}^2]$')
    ax.legend()
    return ax


def plots_med_tube_L2U(errors, ks_, name, ax=None, color='blue', varia_conf=0.05, transfo_i=lambda x:x, a_max=1) :

    err_med_cond = errors['err_med_cond']/a_max
    q1 = np.quantile(err_med_cond, varia_conf/2, axis=-1)
    q2 = np.quantile(err_med_cond, 1-varia_conf/2, axis=-1)


    ax.fill_between(ks_, q1, q2, color=color, alpha=0.6)
    ax.plot(ks_, err_med_cond.mean(-1), linestyle=(0, (5, 1)), color=color, label=name, alpha=0.45)
    # axes[0].plot(ks_, variations_dict['quant2_conf_zone'], '--', color=color)
    ax.set_xlabel('Number of data')
    ax.set_ylabel('error')
    ax.set_yscale('log')
    ax.set_title(r'${}$ confidence for'.format(varia_conf)+r' bias error $||m(P_{\theta})-P_{ref}||_{L^2}^2]$')
    ax.legend()


def plot_cred_ok(cred_arr, ks_, name, conf=0.05, color="blue", ax=None, transfo_i=lambda x:x, quant=False) :
    if ax is None :
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if quant :
        ax.fill_between(ks_, np.quantile(cred_arr[1], conf/2, axis=-1), np.quantile(cred_arr[1], 1-conf/2, axis=-1), color=color, alpha=0.6)
    ax.plot(ks_, cred_arr[0], linestyle=(0, (5, 1)), color=color, label=name, alpha=0.35)
    # ax.plot(ks_, cred_arr, label=name, color=color)
    ax.set_xlabel('Number of data')
    ax.set_ylabel('credibility')
    ax.set_title(r'average credibility $\mathbb{E}\|\mathbf{1}_{[q_{r/2}, q_{1-r/2}]}(P_{ref})\|_{'+r'L^2}$, '+'$r={}$'.format(conf))
    ax.legend()
    return ax


if __name__=="__main__" :
    ks_ = np.arange(1,26)*10
    # path_save = r"./Runs_linear/"
    path_saved_runs = r'/Users/antoinevanbiesbroeck/Documents/data_ran/SUR_alph_imp/mnt/beegfs/workdir/antoine.van-biesbroeck/results_SUR_bin_alpha_impact'

    from bayes_frag.data import Data
    from bayes_frag.reference_curves import Reference_saved_MLE
    from bayes_frag import config
    from scipy.integrate import simpson

    def m_data(IM) :
        # return Data(IM, csv_path='Res_ASG_Lin.csv', quantile_C=0.9, name_inte='rot', shuffle=True)
        da = Data(IM, csv_path='Res_ASG_SA_PGA_RL_RNL_80000.csv', quantile_C=0.9, name_inte='rot_nlin', shuffle=True)
        return da
    def m_ref(data, IM, model = 'ASG') :
        ref = Reference_saved_MLE(data, os.path.join(config.data_path, 'ref_MLE_{}_80000_{}'.format(model, IM)))
        num_clust = 25 if IM=='PGA' else 15 #12 PGA 104
        ref._compute_empirical_curves(num_clust)

        a_qu1 = ref.probit_ref.get_curve_opt_threshold(10**-2)
        data_pga_asg = m_data('PGA')
        ref_pga_asg = Reference_saved_MLE(data_pga_asg, os.path.join(config.data_path, 'ref_MLE_ASG_80000_PGA'))
        kmeans = ref_pga_asg._compute_empirical_curves(20)
        a_asg_qu2 = ref_pga_asg.a_tab_MC[-1] #get_curve_opt_threshold(ref.theta_MLE[0], ref.theta_MLE[1], 10**-2)
        qu2 = ref_pga_asg.probit_ref(a_asg_qu2)
        a_qu2 = ref.probit_ref.get_curve_opt_threshold(qu2)
        data._set_a_tab(min_a=a_qu1, max_a=a_qu2)
        ref._compute_MLE_curve()
        ref._compute_MC_data_tab()
        return ref

    def generic_err_fig(IM, ax, err_num=3, err_name='quad_tube_', err_name_post='_U', func_plot=plot_quad_tubes, mean_name='mean_quad', tube1_name='quant1_quad', tube2_name='quant2_quad', to_divide=1) :

        # IM = 'PGA'

        n_fold = 41 if IM=='PGA' else 42

        folds_son = lambda g : 'M-G{}-ASG_SUR_bin_{}-(1-26)_{}'.format(g, IM, n_fold)

        u = 0 # which take as n_s_J

        color_g = ['red']*9 + ['orange']
        g_labels = {0:r'$\gamma=0$', 1.9:r'$\gamma=1.9$'}

        # 1: plot l'err quadra: tout les gamma
        g_tab = (np.arange(10)*2+1)/10
        errors_3 = []
        err_quad_tube1 = []
        err_quad_tube2 = []
        for i,gamma in enumerate(g_tab) :
            errors_3.append(pickle.load(open(os.path.join(path_saved_runs, folds_son(gamma), 'errors_new_newL2_{}'.format(err_num)), 'rb')))
            if tube1_name is None :
                err_med_cond = errors_3[-1][err_name+'s_Ja'+err_name_post][tube2_name]/to_divide
                err_quad_tube1.append(np.quantile(err_med_cond, 0.05/2, axis=-1))
                err_quad_tube2.append(np.quantile(err_med_cond, 1-0.05/2, axis=-1))
            else :
                err_quad_tube1.append(errors_3[-1][err_name+'s_Ja'+err_name_post][tube1_name])
                err_quad_tube2.append(errors_3[-1][err_name+'s_Ja'+err_name_post][tube2_name])
        #gmma = 0
        if tube1_name is None :
            err_med_cond = errors_3[u][err_name+'s_J'+err_name_post][tube2_name]/to_divide
            err_quad_tube1.append(np.quantile(err_med_cond, 0.05/2, axis=-1))
            err_quad_tube2.append(np.quantile(err_med_cond, 1-0.05/2, axis=-1))
        else :
            err_quad_tube1.append(errors_3[u][err_name+'s_J'+err_name_post][tube1_name])
            err_quad_tube2.append(errors_3[u][err_name+'s_J'+err_name_post][tube2_name])
        err_quad_tube1 = np.array(err_quad_tube1)
        err_quad_tube2 = np.array(err_quad_tube2)

        if to_divide!=1 :
            func_plot(errors_3[u][err_name+'n_J'+err_name_post], ks_, 'standard', ax=ax, a_max=to_divide)
        else :
            try :
                func_plot(errors_3[u][err_name+'n_J'+err_name_post], ks_, 'standard', ax=ax)
            except :
                figtemp = plt.figure()
                ax2 = figtemp.add_subplot()
                func_plot(errors_3[u][err_name+'n_J'+err_name_post], ks_, 'standard', axes=[ax,ax2])
                plt.close(figtemp)

        ax.fill_between(ks_, np.maximum(np.max(err_quad_tube1, axis=0), np.max(err_quad_tube2, axis=0) ), np.minimum(np.min(err_quad_tube1, axis=0), np.min(err_quad_tube2, axis=0) ), color='red', alpha=0.45 )
        for i,gamma in enumerate(g_tab) :
            line = ax.plot(ks_, errors_3[i][err_name+'s_Ja'+err_name_post][mean_name]/to_divide,':', color=color_g[i], alpha=0.6 if gamma!=1.9 else 0.9)
            # if gamma in g_labels.keys() :
            #     line.set_label('Label via method')
        # gmma = 0
        ax.plot(ks_, errors_3[u][err_name+'s_J'+err_name_post][mean_name]/to_divide,':', color='purple', alpha=0.9)
        ax.plot([0],[0], ':', color='black', alpha=0.4, label='with DoE')
        ax.set_xlim(ks_.min(), ks_.max())
        ax.legend()
        ax.set_title(IM)
        ax.set_xlabel(r'Number of observations $k$')

    #1. quad PGA
    IM = 'PGA'
    data = m_data(IM)

    fig = plt.figure(1, figsize=(4.5, 3))
    fig.clf()
    ax = fig.add_subplot(111)

    generic_err_fig(IM, ax)

    ax.set_title(r'quadratic error $\mathcal{E}^{|\mathbf{z}^k,\mathbf{a}^k}$')
    ax.set_ylabel('')


    #2. quad PSA
    IM = 'sa_5hz'
    data = m_data(IM)

    fig = plt.figure(2, figsize=(4.5, 3))
    fig.clf()
    ax2 = fig.add_subplot(111)

    generic_err_fig(IM, ax2)

    ax2.set_title(r'quadratic error $\mathcal{E}^{|\mathbf{z}^k,\mathbf{a}^k}$')
    ax2.set_ylabel('')

    uni_ylims(ax,ax2)


    #3. Bias PGA
    IM = 'PGA'
    data = m_data(IM)
    ref = m_ref(data, IM)
    to_div = data.a_tab.max()-data.a_tab.min()
    model_biais = simpson((ref.curve_MC_data_tabs[0]-ref.curve_MLE)**2, data.a_tab)/to_div

    fig = plt.figure(3, figsize=(4.5, 3))
    fig.clf()
    ax = fig.add_subplot(111)

    generic_err_fig(IM, ax, 1, err_name='errors_L2_', err_name_post='', func_plot=plots_med_tube_L2U, mean_name='err_med', tube1_name=None, tube2_name='err_med_cond', to_divide=to_div)

    ax.axhline(y=model_biais, color='black', linestyle='-', alpha=0.3, label='model bias')
    ax.set_title(r'square bias $\mathcal{B}^{|\mathbf{z}^k,\mathbf{a}^k}$')
    ax.legend()
    ax.set_ylabel('')

    #4. Bias PSA
    IM = 'sa_5hz'
    data = m_data(IM)
    ref = m_ref(data, IM)
    to_div = data.a_tab.max()-data.a_tab.min()
    model_biais = simpson((ref.curve_MC_data_tabs[0]-ref.curve_MLE)**2, data.a_tab)/to_div

    fig = plt.figure(4, figsize=(4.5, 3))
    fig.clf()
    ax2 = fig.add_subplot(111)

    generic_err_fig(IM, ax2, 1, err_name='errors_L2_', err_name_post='', func_plot=plots_med_tube_L2U, mean_name='err_med', tube1_name=None, tube2_name='err_med_cond', to_divide=to_div)

    ax2.axhline(y=model_biais, color='black', linestyle='-', alpha=0.3, label='model bias')
    ax2.set_title(r'square bias $\mathcal{B}^{|\mathbf{z}^k,\mathbf{a}^k}$')
    ax2.legend()
    ax2.set_ylabel('')

    uni_ylims(ax,ax2)

    #5. Width PGA
    IM = 'PGA'
    data = m_data(IM)

    fig = plt.figure(5, figsize=(4.5, 3))
    fig.clf()
    ax = fig.add_subplot(111)

    generic_err_fig(IM, ax, 2, err_name='variations_conf_', func_plot=plot_variations, mean_name='mean_conf_zone', tube1_name='quant1_conf_zone', tube2_name='quant2_conf_zone')

    ax.set_title(r'square credibility width $\mathcal{W}^{|\mathbf{z}^k,\mathbf{a}^k}$')
    ax.set_ylabel('')

    #5. Width PGA
    IM = 'sa_5hz'
    data = m_data(IM)

    fig = plt.figure(6, figsize=(4.5, 3))
    fig.clf()
    ax2 = fig.add_subplot(111)

    generic_err_fig(IM, ax2, 2, err_name='variations_conf_', func_plot=plot_variations, mean_name='mean_conf_zone', tube1_name='quant1_conf_zone', tube2_name='quant2_conf_zone')

    ax2.set_title(r'square credibility width $\mathcal{W}^{|\mathbf{z}^k,\mathbf{a}^k}$')
    ax2.set_ylabel('')

    uni_ylims(ax,ax2)

    ##

    def degen_prob(IM, ax) :
        # IM = 'PGA'

        n_fold = 41 if IM=='PGA' else 42

        folds_son = lambda g : 'M-G{}-ASG_SUR_bin_{}-(1-26)_{}'.format(g, IM, n_fold)

        u = 0 # which take as n_s_J

        color_g = ['purple'] + ['red']*8 + ['orange']
        g_labels = {0:r'$\gamma=0$', 1.9:r'$\gamma=1.9$'}

        # 1: plot l'err quadra: tout les gamma
        g_tab = (np.arange(10)*2+1)/10
        errors_3 = []
        for i,gamma in enumerate(g_tab) :
            errors_3.append(pickle.load(open(os.path.join(path_saved_runs, folds_son(gamma), 'errors_new_newL2_{}'.format(5)), 'rb')))

        plot_proba_degen(errors_3[u]['degen_n_J'], 'standard', ax=ax, color='blue')
        krange = np.arange(1, errors_3[-1]['degen_n_J'].shape[0]+1)


        for i,gamma in enumerate(g_tab) :
            if i>0 :
                ax.plot(krange, errors_3[i]['degen_s_Ja'],':', color=color_g[i], alpha=0.6 if gamma not in [0.1,1.9] else 1)
            # if gamma in g_labels.keys() :
            #     line.set_label('Label via method')
        # # gmma = 0.1
        gamma,i = 0.1,0
        ax.plot(krange, errors_3[i]['degen_s_Ja'],':', color=color_g[i], alpha=0.6 if gamma not in [0.1,1.9] else 0.9)
        # ax.plot(ks_, errors_3[u][err_name+'s_J'+err_name_post][mean_name]/to_divide,':', color='purple', alpha=0.9)
        ax.plot([0],[0], ':', color='black', alpha=0.4, label='with DoE')
        ax.set_xlim(krange.min(), krange.max())
        ax.set_ylim(0,1)

        ax.set_xlabel(r'Number of observations $k$')
        ax.legend()
        return ax


    # PGA
    IM='PGA'
    fig = plt.figure(7, figsize=(4.5, 3))
    fig.clf()
    ax = fig.add_subplot(111)
    degen_prob(IM,ax)
    ax.set_title('Degeneracy probability with PGA')
    ax.grid(alpha=0.3)

    #PSA
    IM='sa_5hz'
    fig = plt.figure(8, figsize=(4.5, 3))
    fig.clf()
    ax = fig.add_subplot(111)
    degen_prob(IM,ax)
    ax.set_title('Degeneracy probability with PSA')
    ax.grid(alpha=0.3)





