# python files
from aux_optimizers import *
from stat_models_numpy import *
from stat_models_torch import *
from variational_approx import *

# packages used
import numpy as np 
       
# MI : mutual information
# LB_MI : lower bound of mutual information (uses MLE instead of the marginal)
    
####################  Divergence Metrics (NumPy)  ####################

class DivMetric_OneLayer():
    def __init__(self, va, T, T_dp, use_alpha, alpha=None, use_log_lik=True):
        self.va = va   # variational approximation (with single linear layer)
        self.model = va.model  # statistical (numpy) model   
        self.use_alpha = use_alpha  # uses -log or alpha-div accordingly
        self.alpha = alpha  # alpha parameter if alpha-div is used
        self.T = T        # number of MC samples for the marginal
        self.T_dp = T_dp  # number of MC samples for the grad marginal
        self.use_log_lik = use_log_lik  # computations made with log-likelihood instead

    # The following functions are all noisy, ie one sample only instead of the expectation
    # Mutual information functions

    def MI(self, theta, w, J, N):   
        model = self.model
        va = self.va
        D = model.sample(theta, N, J)
        if self.use_log_lik :
            log_lik = model.log_likelihood(theta)
            theta_sample = va.implicit_prior_sampler(w, self.T)
            logs_lik = model.collec_log_lik(theta_sample)
            ratio_lik = np.mean(np.exp(logs_lik - log_lik), axis=0)   
        else :
            marg = va.MC_marg(w, self.T)
            lik = model.likelihood(theta)
            ratio_lik = marg/lik
        if self.use_alpha : 
            c = 1 / (self.alpha - 1)
            sommand = alpha_div(ratio_lik, self.alpha, c)
        else:
            sommand = - np.log(ratio_lik)
        return np.mean(sommand)
    
    def grad_MI(self, eps, w, J, N):
        model = self.model
        va = self.va
        theta = va.g(w, eps)
        D = model.sample(theta, N, J)
        if self.use_alpha :
            alpha = self.alpha
            F_func = lambda x : (1-x**alpha)/alpha
            f_prime = lambda x : (x**(alpha-1) - 1)/(alpha-1)
        gradient_log_lik = model.grad_log_lik(theta)
        if self.use_log_lik :
            log_lik = model.log_likelihood(theta)
            theta_sample = va.implicit_prior_sampler(w, self.T)
            logs_lik = model.collec_log_lik(theta_sample)
            ratio_lik = np.mean(np.exp(logs_lik - log_lik), axis=0)
        else :
            marg = va.MC_marg(w, self.T)
            lik = model.likelihood(theta)
            ratio_lik = marg/lik
        if not self.use_alpha :
            sommand = gradient_log_lik * -np.log(ratio_lik)
            return np.mean(sommand) * va.grad_g(w,eps)
        else :
            s1 = np.mean(gradient_log_lik * F_func(ratio_lik)) * va.grad_g(w,eps) 
            if self.use_log_lik :
                epsilon_sample = np.random.normal(size=(self.T_dp, va.p))
                theta_sample = va.g(w,epsilon_sample)
                grad_log_lik = model.collec_grad_log_lik(theta_sample)
                logs_lik = model.collec_log_lik(theta_sample)
                lik_term = np.exp(logs_lik[:,None,:] - log_lik)
                new_marg_grad = np.mean(grad_log_lik[:,None,:] * lik_term * va.grad_g(w,epsilon_sample)[:,:,None], axis=0)
                s2 = np.mean(new_marg_grad * f_prime(ratio_lik), axis=1)
            else :
                grad_marg = va.MC_grad_marg(w, self.T_dp)
                s2 = np.mean(grad_marg * f_prime(ratio_lik) / lik, axis=1)
            return s1 + s2
        
    
    def hess_MI(self, eps, w, J, N):  # not available for alpha-divergences
        model = self.model
        va = self.va
        theta = va.g(w, eps)
        D = model.sample(theta, N, J)
        # Quantities of interest
        marg = va.MC_marg(w, self.T)
        grad_marg = va.MC_grad_marg(w, self.T_dp)
        lik = model.likelihood(theta)
        gradient_log_lik = model.grad_log_lik(theta)
        hessian_log_lik = model.grad2_log_lik(theta)
        hessian_g, prod_grad_g = va.grad2_g(w, eps)
        # Final computations
        ratio_lik = lik/marg
        sommand1 = gradient_log_lik * np.log(ratio_lik)
        sommand2 = hessian_log_lik * np.log(ratio_lik) + (gradient_log_lik)**2 *(1 + np.log(ratio_lik))
        sommand3 = gradient_log_lik * grad_marg / marg
        return hessian_g * np.mean(sommand1) + prod_grad_g * np.mean(sommand2) - va.grad_g(w, eps) * np.mean(sommand3)
    
    # Lower bound functions
    def LB_MI(self, theta, w, J, N):
        model = self.model
        D = model.sample(theta, N, J)
        if self.use_log_lik :
            log_lik = model.log_likelihood(theta)
            max_log_lik = model.log_likelihood(model.MLE())
            if self.use_alpha :
                c = 1 / (self.alpha - 1)
                sommand = alpha_div(np.exp(max_log_lik - log_lik), self.alpha, c)
            else :
                sommand = - (max_log_lik - log_lik) 
        else:
            lik = model.likelihood(theta)
            max_lik = model.likelihood(model.MLE())
            if self.use_alpha :
                c = 1 / (self.alpha - 1)
                sommand = alpha_div(max_lik/lik, self.alpha, c)
            else :
                sommand = - np.log(max_lik/lik) 
        return np.mean(sommand)
    

    def grad_LB_MI(self, eps, w, J, N):
        model = self.model
        va = self.va
        alpha = self.alpha
        theta = va.g(w,eps)
        D = model.sample(theta, N, J)
        if self.use_alpha :
            F_func = lambda x : (1-x**alpha)/alpha
        else :
            F_func = lambda x : -np.log(x)
        gradient_log_lik = model.grad_log_lik(theta)
        if self.use_log_lik :
            log_lik = model.log_likelihood(theta)
            max_log_lik = model.log_likelihood(model.MLE())
            ratio_lik = np.exp(max_log_lik - log_lik)
        else :
            lik = model.likelihood(theta)
            max_lik = model.likelihood(model.MLE())
            ratio_lik = max_lik/lik
        sommand = gradient_log_lik * F_func(ratio_lik)
        return np.mean(sommand) * va.grad_g(w,eps)
    
    def hess_LB_MI(self, eps, w, J, N): 
        model = self.model
        va = self.va
        theta = va.g(w,eps)
        D = model.sample(theta, N, J)
        if self.use_alpha :
            alpha = self.alpha
            F_func = lambda x : (1 - x**alpha)/alpha
            F_prime = lambda x : -x**(alpha-1)
        else :
            F_func = lambda x : -np.log(x) + 1
            F_prime = lambda x : -1/x
        # Quantities of interest
        lik = model.likelihood(theta)
        max_lik = model.likelihood(model.MLE())
        gradient_log_lik = model.grad_log_lik(theta)
        hessian_log_lik = model.grad2_log_lik(theta)
        hessian_g, prod_grad_g = va.grad2_g(w,eps)
        # Final computations
        ratio_lik = max_lik/lik
        sommand1 = gradient_log_lik * F_func(ratio_lik)
        sommand2 = hessian_log_lik * F_func(ratio_lik) + (gradient_log_lik)**2 * F_func(ratio_lik) - (gradient_log_lik)**2 * F_prime(ratio_lik) / lik
        return hessian_g * np.mean(sommand1) + prod_grad_g * np.mean(sommand2)