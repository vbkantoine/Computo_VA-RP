# code stat_functions
# needed functions for statistical use
# from cProfile import label
# from symbol import return_stmt
import numpy as np
import numpy.random as rd
import math
from numpy.linalg import cholesky
import numba
from numba import jit, prange
import matplotlib.pyplot as plt

from utils import simpson_numb, jrep0, jrep1
# from fisher import Jeffreys_rectangles, Jeffreys_simpson, Jeffreys_simpson_numba, Fisher_Simpson, Fisher_Simpson_Numb
from distributions import gen_log_norm_cond, log_norm_conditionnal


## class HM estimator :
class HM_Estimator() :
    """HM estimator class that runs an HM_algorithm and stores the logs"""
    def __init__(self, HM_func, z0, pi, *args, **kwargs) :
        """HM_func: an HM algorithm to generate the simulations
            z0: init point for HM
            pi: pdf to simulate
            *args, **kwargs: to put in the HM when ran"""
        self.args = args
        self.kwargs = kwargs
        self.HM_func = HM_func
        self.pi = pi
        self.z0 = z0
        self.logs = list()

    def simulate(self, n_est) :
        """generate n_est estimations via self.HM_func"""
        res = self.HM_func(self.z0, self.pi, *self.args, **self.kwargs)
        self.logs.append(res)
        return res[1][-n_est:]
    
    def analyse_HM(self, k, axes=None, start=0, kept_plot_ids=None) :
        """generate analysis plot of previous geenreations
            k = logs id to plot
            axes are created if None, list(3 ax)
            start = from which HM iteration to start the plots
            kept_plot_ids = if fewer ids to keep in the plot"""
        # HM is supposed to return (t_fin, t_tot, acc_tot)
        s_fin, t_tot, acc_tot = self.logs[k]
        true_acc = np.minimum(acc_tot, 1)
        num = t_tot[start:].shape[0]
        evol_acc = true_acc[start:].cumsum()/np.arange(start+1,num+1)
        alpha_evol = t_tot[start:,0]
        alpha_mean = alpha_evol.cumsum()/np.arange(start+1,num+1)
        beta_evol = t_tot[start:,1]
        beta_mean = beta_evol.cumsum()/np.arange(start+1,num+1)
        if kept_plot_ids is None :
            kept_plot_ids = np.arange(start, num)
        if axes is None :
            fig = plt.figure()
            axes = list()
            axes.append(fig.add_subplot(131))
            axes.append(fig.add_subplot(132))
            axes.append(fig.add_subplot(133))
        axes[0].plot(kept_plot_ids, evol_acc)
        axes[0].set_ylim(0,1)
        axes[1].plot(kept_plot_ids, alpha_evol, label=r'$\alpha$')
        axes[1].plot(kept_plot_ids, alpha_mean, label=r'$\mathbb{E}\alpha$')
        axes[1].legend()
        axes[2].plot(kept_plot_ids, beta_evol, label=r'$\beta$')
        axes[2].plot(kept_plot_ids, beta_mean, label=r'$\mathbb{E}\beta$')
        axes[2].legend()
        return axes

    def clear_logs(self) :
        """clear the logs from previous generations"""
        self.logs = list()

class Post_HM_Estimator(HM_Estimator) :
    """sub HM Estimator class to generate a posteriori data"""
    def __init__(self, HM_func, z0, post, *args, **kwargs) :
        """post=func(s,a,theta)"""
        super(Post_HM_Estimator, self).__init__(HM_func, z0, post, *args, **kwargs)
        self.post = post

    def __call__(self, a, s) :
        """when called, update the posterior according to prompted data and simulates"""
        self.pi = self.func_log_post(a,s)
        return self.simulate

    def func_log_post(self,a,s) :
        full_post = self.post
        if numba.core.registry.CPUDispatcher==type(self.pi) :
            @jit(nopython=True)
            def post(theta) :
                return full_post(s, a, theta) 
        else :
            def post(theta) :
                return full_post(s, a, theta) 
        return post







## HM :

@jit(nopython=True)
def HM_gauss(z0, pi, max_iter=5000, sigma_prop=10**-2*np.eye(2)) :
    zk = z0+0
    for k in range(max_iter) :
        z = zk + sigma_prop@rd.randn(d)
        log_alpha = np.log(pi(zk))-np.log(pi(z))
        rand = np.log(rd.rand())<log_alpha
        zk += rand*(z-zk)
    return rk

@jit(nopython=True)
def HM_k(z0, pi, k, pi_log=False, max_iter=5000, sigma_prop=10**-2) :
    d = z0.shape[0]
    z_v = jrep0(z0, k)
    z_tot = np.zeros((max_iter,k,2))
    u = len(np.asarray(sigma_prop).shape)
    alpha_tab = np.zeros((max_iter,k))
    if u==0 :
        sig = sigma_prop*np.eye(2)
    else :
        sig = sigma_prop+0
    for n in range(max_iter) :
        pi_zv = pi(z_v)
        z = np.zeros_like(z_v)
        # for i,z_vi in enumerate(z_v) :
        for i in range(k) :
            z[i] = z_v[i] + sig@rd.randn(d)
            # z[i] = np.asarray(z_vi) + sig@rd.randn(d)
        pi_z = pi(z)
        if pi_log :
            log_alpha = pi_z - pi_zv
        else :
            log_alpha = np.log(pi_z)-np.log(pi_zv)
        rand = np.log(rd.rand(k))<log_alpha
        alpha_tab[n] = np.exp(log_alpha)
        # alpha = pi_z / pi_zv
        # rand = rd.rand(k)<alpha
        #pi_z_vi = pi(z_vi)
        #pi_zi = pi(zi)
        #log_alpha = np.log(pi_z_vi)-np.log(pi_zi)
        #rand = np.log(rd.rand())<log_alpha
        z_v += jrep1(rand, d)*(z-z_v)
        z_tot[n] = z_v + 0
    return z_v, z_tot, alpha_tab


@jit(nopython=True)
def adaptative_HM_k(z0, pi, k, pi_log=False, max_iter=5000, sigma0=0.1*np.eye(2), b=0.05) :
    d = z0.shape[0]
    z_v = jrep0(z0, k)
    z_tot = np.zeros((max_iter,k,2))
    # u = len(np.asarray(sigma0).shape)
    alpha_tab = np.zeros((max_iter,k))
    step = 40*d
    # if u==0 :
    # sig = sigma0*np.eye(d)
    # else :
    sig = sigma0+0
    sig_emp = sig+0
    for n in range(max_iter) :
        pi_zv = pi(z_v)
        z = np.zeros_like(z_v)
        # for i,z_vi in enumerate(z_v) :
        if n>=step :
            for i in prange(k) :
                z[i] = z_v[i] + (1-b)*2.38*sig_emp@rd.randn(d)/np.sqrt(d) + b* sig@rd.randn(d)/np.sqrt(d)
                # z[i] = np.asarray(z_vi) + sig@rd.randn(d)
        else :
            for i in prange(k) :
                z[i] = z_v[i] + sig@rd.randn(d)/np.sqrt(d)
        pi_z = pi(z)
        if pi_log :
            log_alpha = pi_z - pi_zv
        else :
            log_alpha = np.log(pi_z)-np.log(pi_zv)
        rand = np.log(rd.rand(k))<log_alpha
        alpha_tab[n] = np.exp(log_alpha)
        # alpha = pi_z / pi_zv
        # rand = rd.rand(k)<alpha
        #pi_z_vi = pi(z_vi)
        #pi_zi = pi(zi)
        #log_alpha = np.log(pi_z_vi)-np.log(pi_zi)
        #rand = np.log(rd.rand())<log_alpha
        z_v += jrep1(rand, d)*(z-z_v)
        z_tot[n] = z_v + 0
        # tocov = np.expand_dims(z_tot[:n+1,:,0].flatten(), axis=0)
        # be = np.expand_dims(z_tot[:n+1,:,1].flatten(), axis=0)
        tocov = np.stack((z_tot[:n+1,:,0].flatten(), z_tot[:n+1,:,1].flatten()), axis=0)
        sig_emp = cholesky(np.cov(tocov)+10**-10*np.eye(d))
        # if sig_emp.shape!=(2,2) :
        #     return sig_emp, z_tot, alpha_tab
    return z_v, z_tot, alpha_tab

@jit(nopython=True)
def adaptative_HM(z0, pi, pi_log=False, max_iter=5000, sigma0=0.1*np.eye(2), b=0.05, step=500) :
    d = 2
    z_v = z0.reshape(1,2)
    z_tot = np.zeros((max_iter,2))
    # u = len(np.asarray(sigma0).shape)
    alpha_tab = np.zeros((max_iter,1))
    # if u==0 :
    # sig = sigma0*np.eye(d)
    # else :
    sig = sigma0+0
    sig_emp = sig+0
    for n in range(max_iter) :
        pi_zv = pi(z_v)
        z = np.zeros_like(z_v)
        if n>=step :
            z[0] = z_v[0] + (1-b)*2.38*sig_emp@rd.randn(d)/np.sqrt(d) + b* sig@rd.randn(d)/np.sqrt(d)
        else :
            z[0] = z_v[0] + sig@rd.randn(d)/np.sqrt(d)
        pi_z = pi(z)
        if pi_log :
            log_alpha = pi_z - pi_zv
        else :
            log_alpha = np.log(pi_z)-np.log(pi_zv)
        rand = np.log(rd.rand())<log_alpha
        alpha_tab[n] = np.exp(log_alpha)
        z_v += rand*(z-z_v)
        z_tot[n] = z_v[0] + 0
        if n>=step :
            tocov = np.transpose(z_tot[:n+1])
            sig_emp = cholesky(np.cov(tocov)+10**-9*np.eye(d))
    return z_v, z_tot, alpha_tab.flatten()

@jit(nopython=True)
def adapt_rate_HM(z0, pi, pi_log=False, max_iter=5000, sigma0=np.array([0.1]), target_accept=0.4, batch_size=50) :
    d = z0.shape[0]
    z_tot = np.zeros((max_iter,2))
    z_v = z0.reshape(1,d)
    alpha_tab = np.zeros((max_iter,1))
    sig = sigma0+0
    # # l_sig = sigma0*np.eye(d)
    j = 0
    accept = np.zeros(1)
    matr = cholesky(0.5*np.eye(d)+0.5*np.ones((d,d)))
    for n in range(max_iter):
        pi_zv = pi(z_v)
        z = np.zeros_like(z_v)
        z[0] = z_v[0] + sig*matr@rd.randn(d)
        pi_z = pi(z)
        if pi_log :
            log_alpha = pi_z - pi_zv
        else :
            log_alpha = np.log(pi_z)-np.log(pi_zv)
        rand = np.log(rd.rand())<log_alpha
        alpha_tab[n] = np.exp(log_alpha)
        z_v += rand*(z-z_v)
        accept = accept+rand 
        z_tot[n] = z_v[0] + 0
        if (n+1)%batch_size==0 :
            j = j+1
            delta_j = min(0.01, j**-0.5)
            sig = sig * np.exp( np.sign( accept/batch_size - target_accept)*delta_j/2) 
            # l_sig = sig*np.eye(d)
            # batch_acc.append(accept/batch_size)
            accept = np.zeros(1)
    return z_v, z_tot, alpha_tab.flatten()


@jit(nopython=True)
def log_adaptative_HM(z0, pi, pi_log=False, max_iter=5000, sigma0=0.1*np.eye(2), b=0.05, step=500) : # to simulate log(alpha), log(beta) instead of alpha, beta
    z_v, z_tot, alpha_tab = adaptative_HM(z0, pi, pi_log, max_iter, sigma0, b, step)
    return np.exp(z_v), np.exp(z_tot), alpha_tab

@jit(nopython=True)
def adaptative_HM_1d(z0, pi, pi_log=False, max_iter=5000, sigma0=0.1, b=0.05, step=500) :
    # d = z0.shape[0]
    z_v = np.zeros(1)
    z_v = z0+0
    z_tot = np.zeros((max_iter,1))
    # u = len(np.asarray(sigma0).shape)
    alpha_tab = np.zeros((max_iter,1))
    # if u==0 :
    # sig = sigma0*np.eye(d)
    # else :
    sig = sigma0+0
    sig_emp = sig+0
    for n in range(max_iter) :
        pi_zv = pi(z_v)
        z = np.zeros(1)
        if n>=step :
            z = z_v + (1-b)*2.38*sig_emp*rd.randn() + b* sig*rd.randn()
        else :
            z = z_v + sig*rd.randn()
        pi_z = pi(z)
        if pi_log :
            log_alpha = pi_z - pi_zv
        else :
            log_alpha = np.log(pi_z)-np.log(pi_zv)
        rand = np.log(rd.rand())<log_alpha
        alpha_tab[n] = np.exp(log_alpha)
        z_v += rand*(z-z_v)
        z_tot[n] = z_v + 0
        if n>=step :
            # tocov = np.transpose(z_tot[:n+1])
            # sig_emp = cholesky(np.cov(tocov)+10**-10*np.eye(d))
            sig_emp = np.sqrt(z_tot[:n+1].var())
    return z_v, z_tot[:,0], alpha_tab[:,0]






@jit(nopython=True, parallel=True)
def HM_k_log_norm(z0, pi, k, max_iter=5000, sigma_prop=10**-2) :
    d = z0.shape[0]
    z_v = jrep0(z0, k)
    for n in range(max_iter) :
        pi_zv = pi(z_v)
        z = np.zeros_like(z_v)
        for i in prange(k) :
            z[i] = gen_log_norm_cond(z_v[i], sigma_prop)
        pi_z = pi(z)
        log_alpha = np.log(pi_z)-np.log(pi_zv) + np.log(log_norm_conditionnal(z_v, z, sigma_prop)) - np.log(log_norm_conditionnal(z, z_v, sigma_prop))
        rand = np.log(rd.rand(k))<log_alpha
        #pi_z_vi = pi(z_vi)
        #pi_zi = pi(zi)
        #log_alpha = np.log(pi_z_vi)-np.log(pi_zi)
        #rand = np.log(rd.rand())<log_alpha
        z_v += jrep1(rand,d)*(z-z_v)
    return z_v



def HM_Fisher() :
    return True

def adaptative_HM_nonumb(z0, pi, pi_log=False, max_iter=5000, sigma0=0.1*np.eye(2), b=0.05, step=500, d=2) :
    # d = 2
    z_v = z0.reshape(1,d)
    z_tot = np.zeros((max_iter,d))
    # u = len(np.asarray(sigma0).shape)
    alpha_tab = np.zeros((max_iter,1))
    # if u==0 :
    # sig = sigma0*np.eye(d)
    # else :
    sig = sigma0+0
    sig_emp = sig+0
    for n in range(max_iter) :
        pi_zv = pi(z_v)
        z = np.zeros_like(z_v)
        if n>=step :
            z[0] = z_v[0] + (1-b)*2.38*sig_emp@rd.randn(d)/np.sqrt(d) + b* sig@rd.randn(d)/np.sqrt(d)
        else :
            z[0] = z_v[0] + sig@rd.randn(d)/np.sqrt(d)
        pi_z = pi(z)
        if pi_log :
            log_alpha = pi_z - pi_zv
        else :
            log_alpha = np.log(pi_z)-np.log(pi_zv)
        rand = np.log(rd.rand())<log_alpha
        alpha_tab[n] = np.exp(log_alpha)
        z_v += rand*(z-z_v)
        z_tot[n] = z_v[0] + 0
        if n>=step :
            tocov = np.transpose(z_tot[:n+1])
            sig_emp = cholesky(np.cov(tocov)+10**-6*np.eye(d))
    return z_v, z_tot, alpha_tab.flatten()





## KL :

@jit(nopython=True)
def Kullback_Leibler_MC_HM(p,q, dim, iter_p=1000, iter_HM=5000) :
    """KL div estimated via MCMC, with integration samples generated via HM"""
    p0 = np.ones(dim)
    p_simul = HM_k(p0, p, iter_p, max_iter=iter_HM)
    return np.sum(np.log(p(p_simul)/q(p_simul)))


def Kullback_Leibler_simpson(p_tab, q_tab, x_tab) :
    """KL div with integration computed via simpson method"""
    f = p_tab*np.log(p_tab/q_tab)
    return simpson(f, x_tab)

@jit(nopython=True, parallel=True)
def Kullback_Leibler_simpson_numb(p_tab, q_tab, x_tab) :
    """KL div with integgration computed via simpson method, in Numba mode """
    f = p_tab*np.log(p_tab/q_tab)
    return simpson_numb(f, x_tab)





## distribs :

def likelihood(z, a, theta) :
    """probit likelihood
        z,a = array(n) ; theta=array(l,2)
    return array(l) """
    l = theta.shape[0]
    n = a.shape[0]
    p = np.zeros((l,n))
    ind0 = np.zeros(l, dtype='int')
    for i in range(l) :
        if np.any(theta[i]<=0) :
            p[i] = 0
        else :
            for k in range(n) :
                phi = 1/2+1/2*math.erf((np.log(a[k]/theta[i,0])/theta[i,1]))
                if (phi<1e-11 and z[k]==0) or (phi>1-1e-11 and z[k]==1) :
                    p[i,k] = 1
                else :
                    if (phi<1e-11 and z[k]==1) or (phi>1-1e-11 and z[k]==0) :
                        phi = (phi<1e-11)*1e-11 + (phi>1-1e-11)*(1-1e-11)
                    p[i,k] = z[k]*phi + (1-z[k])*(1-phi)
    return p.prod(axis=1)



@jit(nopython=True, parallel=True, cache=True)
def p_z_cond_a_theta_binary(z,a,theta) :
    """probit likelihood
        z,a = array(n) ; theta=array(l,2)
        return array(l)
        computed as exp of the sum of logs.
        Numba mode
    """
    l = theta.shape[0]
    n = a.shape[0]
    logp = np.zeros((l,n,1))
    ind0 = np.zeros(l, dtype='int')
    #for i,t in enumerate(theta) :
    for i in range(l) :
        if np.any(theta[i]<=0) :
            ind0[i] = 1
        else :
            # for k,zk in enumerate(z) :
            for k in prange(n) :
                phi = 1/2+1/2*math.erf((np.log(a[k]/theta[i,0])/theta[i,1]))
                if phi<1e-11 or phi>1-1e-11 :
                    phi = (phi<1e-11)*1e-11 + (phi>1-1e-11)*(1-1e-11)
                logp[i,k] = z[k]*np.log(phi) + (1-z[k])*np.log((1-phi))
    p = np.exp(logp.sum(axis=1)).flatten()
    # print((p==0).sum())
    # p[ind0] = 0
    p = p*(1-ind0)
    # print(ind0)
    # print((p==0).sum())
    return p

def posterior(theta, z, a, prior, cond=p_z_cond_a_theta_binary) :
    """posterior computation from a prior and a likelihood law (theta,a,z=arrays)"""
    p = cond(z,a,theta)*prior(theta)
    return p

@jit(nopython=True, parallel=True, cache=True)
def posterior_numb(theta, z, a, prior, cond=p_z_cond_a_theta_binary) :
    """posterior computation from a prior and a likelihood law (theta,a,z=arrays)
    numba mode"""
    p = cond(z,a,theta)*prior(theta)
    return p

def get_opp_log_vr_probit(z,a) :
    def opp_log_vraiss(theta) :
        # a = A_tot+0
        # z = S_tot+0
        # l = theta.shape[0]
        n = a.shape[0]
        logp = np.zeros((n,1))
        # ind0 = np.zeros(l, dtype='int')
        #for i,t in enumerate(theta) :
        if np.any(theta<=0) :
            return np.inf
        else:
            # for k,zk in enumerate(z) :
            for k in range(n) :
                phi = 1/2+1/2*math.erf((np.log(a[k]/theta[0])/theta[1]))
                if phi<1e-11 or phi>1-1e-11 :
                    phi = (phi<1e-11)*1e-11 + (phi>1-1e-11)*(1-1e-11)
                logp[k] = z[k]*np.log(phi) + (1-z[k])*np.log((1-phi))
        # p = np.exp(logp.sum(axis=1)/n).flatten()
        # print((p==0).sum())
        # p[ind0] = 0
        # p = p*(1-ind0)
        # print(ind0)
        # print((p==0).sum())
        return -logp.mean(axis=0).flatten()
    return opp_log_vraiss


@jit(nopython=True, parallel=True, cache=True)
def log_vrais(z,a,theta) :
    """ log-likelihood in numba mode
    return array(theta.sahpe[0])"""
    l = theta.shape[0]
    n = a.shape[0]
    logp = np.zeros((l,n,1))
    ind0 = np.zeros(l, dtype=np.int64)
    #for i,t in enumerate(theta) :
    for i in range(l) :
        if np.any(theta[i]<=0) :
            ind0[i] = 1
        else :
            # for k,zk in enumerate(z) :
            for k in prange(n) :
                # print((np.log(a[k]/theta[i,0])/theta[i,1]))
                phi = 1/2+1/2*math.erf((np.log(a[k]/theta[i,0])/theta[i,1])[0])
                if phi<1e-11 or phi>1-1e-11 :
                    phi = (phi<1e-11)*1e-11 + (phi>1-1e-11)*(1-1e-11)
                logp[i,k] = z[k]*np.log(phi) + (1-z[k])*np.log((1-phi))
    # logp = logp - logp.max()
    lp = (logp.sum(axis=1)).flatten()
    # m = lp.max()
    # lp = lp - lp.max()
    lp = lp - ind0*1e11
    return lp


@jit(nopython=True, parallel=True, cache=True)
def log_post_jeff(theta, z, a) :
    """log posterior with jeffrey prior, numba mode"""
    l = theta.shape[0]
    # if Jeffrey :
    log_J = np.zeros(l)
    # for i, t in enumerate(theta) :
    for i in prange(l) :
        t = theta[i]
        alpha = t[0]
        beta = t[1]
        if alpha<=0 or beta<=0 :
            log_J[i] = -1e-11
        else :
            a_tab = np.exp(np.linspace(np.log(alpha)-4*beta, np.log(alpha)+4*beta, 40))
            I = Fisher_Simpson_Numb(np.array([alpha]), np.array([beta]), a_tab)
            log_J[i] = 1/2 * np.log(I[0,0,0,0]*I[0,0,1,1] - I[0,0,1,0]**2) #/a.shape[0]
    # else :
    #     log_J = prior(theta)/a.shape[0]
    vr = log_vrais(z,a,theta) #/a.shape[0]
    return vr + log_J


def log_post_jeff_notnumb(theta, z, a) :
    """log posterior with jeffreys prior"""
    l = theta.shape[0]
    # if Jeffrey :
    log_J = np.zeros(l)
    # for i, t in enumerate(theta) :
    for i in range(l) :
        t = theta[i]
        alpha = t[0]
        beta = t[1]
        a_tab = np.exp(np.linspace(np.log(alpha)-4*beta, np.log(alpha)+4*beta, 40))
        I = Fisher_Simpson(np.array([alpha]), np.array([beta]), a_tab)
        log_J[i] = 1/2 * np.log(I[0,0,0,0]*I[0,0,1,1] - I[0,0,1,0]**2)#/a.shape[0]
    # else :
    #     log_J = prior(theta)/a.shape[0]
    vr = log_vrais(z,a,theta) #/a.shape[0]
    return vr + log_J



@jit(nopython=True, parallel=True, cache=True)
def log_post_jeff_adapt(theta, z, a, Fisher) :
    """log posterior with  jeffreys prior, computed from Fisher function prompted (numba mode)"""
    I = Fisher(theta)
    log_J = 1/2 * np.log(I[:,0,0]*I[:,1,1] - I[:,1,0]**2)
    vr = log_vrais(z,a,theta)
    return vr + log_J



def get_log_vrais_lin(A,Y, coeff, C) :
    @jit(nopython=True)
    def log_vraiss_log_lin(th):
        # thh = np.zeros((1,2))
        thh = th.flatten()
        # arr = -1/2 * np.log(thh[1]*np.abs(np.log(C)/np.log(thh[0]))) - (np.log(Y) + np.log(A)*np.log(C)/np.log(thh[0])  )**2/2/thh[1]*np.abs(np.log(C)/np.log(thh[0])) +100
        # arr = -1/2 * np.log(thh[1]) - (np.log(Y) - np.log(A)+np.log(thh[0]))**2/2/thh[1]+10 #for normalized A,Y
        # arr = -1/2 * np.log(thh[1]) - (np.log(Y) - coeff*np.log(A)-inter+np.log(thh[0]))**2/2/thh[1]+1000 #for un-normalized A,Y
        # arr = -1/2 * np.log(thh[1]) - (np.log(Y) - coeff*np.log(A)-inter+np.log(thh[0])+np.log(C)/coeff)**2/2/thh[1]*coeff+100
        """below true formula for log normal model with coeff
            log(Y)/c = log(C)/c + N(log(a/alp), bet2)"""
        arr = -1/2 * np.log(thh[1]) - (np.log(Y)/coeff - np.log(C)/coeff - np.log(A) + np.log(thh[0]))**2/2/thh[1]**2+1000
        return arr.sum()
    return log_vraiss_log_lin

@jit(nopython=True)
def log_vr_lin_full(A,Y,th, coeff, C) :
    thh = th.flatten()
    # arr = -1/2 * np.log(thh[1]*np.abs(np.log(C)/np.log(thh[0]))) - (np.log(Y) + np.log(A)*np.log(C)/np.log(thh[0])  )**2/2/thh[1]*np.abs(np.log(C)/np.log(thh[0])) +100
    # arr = -1/2 * np.log(thh[1]) - (np.log(Y) - np.log(A)+np.log(thh[0]))**2/2/thh[1]+10 #for normalized A,Y
    # arr = -1/2 * np.log(thh[1]) - (np.log(Y) - coeff*np.log(A)-inter+np.log(thh[0]))**2/2/thh[1]+1000 #for un-normalized A,Y
    # arr = -1/2 * np.log(thh[1]) - (np.log(Y) - coeff*np.log(A)-inter+np.log(thh[0])+np.log(C)/coeff)**2/2/thh[1]*coeff+100
    """below true formula for log normal model with coeff
        log(Y)/c = log(C)/c + N(log(a/alp), bet2)"""
    arr = -np.log(thh[1]) - (np.log(Y)/coeff - np.log(C)/coeff - np.log(A) + np.log(thh[0]))**2/2/thh[1]**2+1000
    return arr.sum()

@jit(nopython=True)
def log_Jeffreys_lin(theta) :
    thh = np.zeros((1,2))
    thh[0] = theta.flatten()
    return -np.log(thh[:,0]) -  np.log(thh[:,1])
















