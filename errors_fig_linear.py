
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from bayes_frag import plot_functions


def plot_proba_degen(errs, name, ax=None, color='blue') :
    if ax is None :
        fig =plt.figure()
        ax = fig.add_subplot(111)
    krange = np.arange(1, errs.shape[0]+1)
    ax.plot(krange, errs, '--', label=name, color=color)
    ax.set_xlabel('Number of data')
    ax.set_title('Degeneracy probability')
    ax.legend()
    return ax

def plot_variations(variations_dict, ks_, name, varia_conf=0.05, conf=0.05, L2set=r'(\mathbb{P}_A)', axes=None, color='blue', transfo_i=lambda x:x) :
    if axes is None :
        fig = plt.figure()
        axes = [fig.add_subplot(1,2,j+1) for j in range(2)]
    axes[0].fill_between(ks_, variations_dict['quant1_conf_zone'], variations_dict['quant2_conf_zone'], color=color, alpha=0.6)
    axes[0].plot(ks_, variations_dict['mean_conf_zone'], '--', color=color, label=name, alpha=0.35)
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
    ax.plot(ks_, quad_dict['mean_quad'],'--', color=color, label=name, alpha=0.35)
    # axes[0].plot(ks_, variations_dict['quant2_conf_zone'], '--', color=color)
    ax.set_xlabel('Number of data')
    ax.set_ylabel('error')
    ax.set_yscale('log')
    ax.set_title(r'${}$ confidence for'.format(varia_conf)+r' quadratic error $\mathbb{E}[||P_{\theta}-P_{ref}||_{L^2'+r'{}'.format(L2set)+r'}^2]$')
    ax.legend()
    return ax


def plots_med_tube_L2U(errors, ks_, ax, name, color='blue', varia_conf=0.05, transfo_i=lambda x:x, a_max=1) :

    err_med_cond = errors['err_med_cond']/a_max
    q1 = np.quantile(err_med_cond, varia_conf/2, axis=-1)
    q2 = np.quantile(err_med_cond, 1-varia_conf/2, axis=-1)


    ax.fill_between(ks_, q1, q2, color=color, alpha=0.6)
    ax.plot(ks_, err_med_cond.mean(-1),'--', color=color, label=name, alpha=0.35)
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
    ax.plot(ks_, cred_arr[0], '--', color=color, label=name, alpha=0.35)
    # ax.plot(ks_, cred_arr, label=name, color=color)
    ax.set_xlabel('Number of data')
    ax.set_ylabel('credibility')
    ax.set_title(r'average credibility $\mathbb{E}\|\mathbf{1}_{[q_{r/2}, q_{1-r/2}]}(P_{ref})\|_{'+r'L^2}$, '+'$r={}$'.format(conf))
    ax.legend()
    return ax


if __name__=="__main__" :
    ks_ = np.arange(1,20)*5
    # path_save = r"./Runs_linear/"
    path_save = r'/Users/antoinevanbiesbroeck/Documents/New code/runs_SURbin/2024-31-01-SUR_bin-(1-20)_2'
    errors = pickle.load(open(os.path.join(path_save, 'errors'), 'rb'))
    # pickle.dump({'errors_L2_J':errors_L2_J, 'errors_L2_st':errors_L2_st, 'variations_conf_J_U':variations_conf_J_U, 'variations_conf_st_U':variations_conf_st_U, 'quad_tube_J_U':quad_tube_J_U, 'quad_tube_st_U':quad_tube_st_U}, open(os.path.join(path_save, "errors"), "wb"))


    fig = plt.figure(12, figsize = (4.5,4))
    figtemp = plt.figure()
    fig.clf()
    axes = [fig.add_subplot(111), figtemp.add_subplot(111)]
    # plot_variations(variations_conf_MLE_U, ks_, 'MLE', L2set='' , axes=axes, color='red')
    plot_variations(errors['variations_conf_st_U'], ks_, 'SUR', L2set='', axes=axes, color='orange')
    plot_variations(errors['variations_conf_J_U'], ks_, 'no SUR', L2set='', axes=axes, color='blue')
    plt.close(figtemp)

    # fig = plt.figure(6)
    # fig.clf()
    # axes = [fig.add_subplot(1,2,j+1) for j in range(2)]
    # plot_variations(variations_conf_MLE_U, ks_, 'MLE', L2set='' , axes=axes, color='green')


    fig = plt.figure(10, figsize = (4.5,4))
    fig.clf()
    ax = fig.add_subplot(111)
    # plot_quad_tubes(quad_tube_MLE_U, ks_, 'MLE', L2set='', ax=ax, color='red')
    plot_quad_tubes(errors['quad_tube_st_U'], ks_, 'SUR', L2set='', ax=ax, color='orange')
    plot_quad_tubes(errors['quad_tube_J_U'], ks_, 'no SUR', L2set='', ax=ax, color='blue')


    print("median error tube computation (L2U)")
    fig = plt.figure(11, figsize = (4.5,4))
    fig.clf()
    ax = fig.add_subplot(111)
    plots_med_tube_L2U(errors['errors_L2_J'], ks_, ax, 'no SUR', color='blue')
    plots_med_tube_L2U(errors['errors_L2_st'], ks_, ax, 'SUR', color='orange')
