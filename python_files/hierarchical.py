# python files
from aux_optimizers import *
from stat_models_numpy import *
from stat_models_torch import *
from neural_nets import *
from variational_approx import *
from div_metrics_numpy import *
from div_metrics_torch import *

# packages used
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

####################  Hierarchical class  ####################

# We build the following class onto the "torch_NormalModel_variance" statistical model class 
# Several methods have the "use_formula" argument which utilizes the fact that the Normal model is used. When set to True,
# the computations are faster thanks to vectorization, which would not be possible for a general, hence implicit implementation.

class Hierarchical_Normal():
    def __init__(self, div, T1):
        """ Defines the hierarchical normal model
        Args:
            div (DivMetric_NeuralNet): Divergence used for the lower level statistical model
            T1 (int): Number of MC samples from the lower level RP
        """
        self.va = div.va
        self.model = self.va.model
        self.data = None
        self.available_MLE = False
        self.T1 = T1


    def sample(self, theta2, N, J, use_formula=True):
        """ Samples from the second level likelihood
        Args:
            theta2 (tensor): Current value of the parameter
            N (int): Size of the sample
            J (int): Number of N-samples
            use_formula (bool, optional): Uses explicitly the model for faster sampling. Defaults to True.
        Returns:
            (N,J) tensor : Data sampled from the likelihood
        """
        self.model.mu = theta2
        thetas_1 = self.va.implicit_prior_sampler(J).squeeze(1)
        if use_formula :
            D = theta2 + torch.sqrt(thetas_1) * torch.randn(N, J)
        else :
            D = torch.zeros((N, J))
            for j in range(J):
                D[:, j] = self.model.sample(thetas_1[j], N, 1).squeeze(1)
        self.data = D
        self.model.data = D
        return D

    # Derivative on the FIRST level likelihood wrt to the second component (mu here)
    def grad_lik_1st_level(self, thetas_1, theta2):
        """ Computes the derivative of the Normal log-likelihood wrt to the mean parameter (mu)
        Args:
            thetas_1 (T1 tensor): Samples from the low level RP
            theta2 (tensor): Value of the parameter
        Returns:
            (T1,J) tensor 
        """
        D = self.data
        sum_data = torch.sum((D - theta2), dim=0)
        return (thetas_1[:, None]**-1) * sum_data[None, :]


    def likelihood(self, theta2):
        self.model.mu = theta2
        thetas_1 = self.va.implicit_prior_sampler(self.T1).squeeze(1)
        return torch.mean(self.model.collec_lik(thetas_1), dim=0)

    def log_likelihood(self, theta2):
        return torch.log(self.likelihood(theta2))


    def collec_lik(self, thetas2, use_formula=True):
        T = thetas2.size(0)
        N, J = self.data.size()
        if use_formula :
            D = self.data
            thetas_1 = self.va.implicit_prior_sampler(self.T1).squeeze(1)
            cste = torch.sqrt(2*math.pi*thetas_1)**-N
            sum_term = torch.sum((D[None,:,:] - thetas2[:,None,None])**2, dim=1)
            exp_term = torch.exp(-0.5*sum_term[None,:,:] / thetas_1[:,None,None])
            lik_2D = torch.mean(cste[:,None,None] * exp_term, dim=0)
        else :
            lik_2D = torch.zeros((T, J))
            for t in range(T):
                lik_2D[t, :] = self.likelihood(thetas2[t])
        return lik_2D

    def collec_log_lik(self, thetas2, use_formula=True):
        T = thetas2.size(0)
        J = self.data.size(1)
        if use_formula :
            log_lik_2D = torch.log(self.collec_lik(thetas2))
        else :
            log_lik_2D = torch.zeros((T, J))
            for t in range(T):
                log_lik_2D[t, :] = self.log_likelihood(thetas2[t])
        return log_lik_2D


    def grad_log_lik(self, theta2):
        self.model.mu = theta2
        thetas_1 = self.va.implicit_prior_sampler(self.T1).squeeze(1)     
        soft_m = F.softmax(self.model.collec_lik(thetas_1), dim=0)
        grad_lik_one = self.grad_lik_1st_level(thetas_1, theta2)
        grad_1D = torch.sum(soft_m * grad_lik_one, dim=0)
        return grad_1D




