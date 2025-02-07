# python files
from aux_optimizers import *
from stat_models_numpy import *
from stat_models_torch import *
from neural_nets import *

# packages used
import numpy as np 
import torch
from tqdm import tqdm

########## Variational Approximations ##########


##############################  Simple case in Numpy  ##############################

class VA_OneLayer():   # similar to a simple neural network (one linear layer, explicit gradient/hessian functions) 
    def __init__(self, p, model, act_func='sigmoid', use_bias=False):
        self.p = p           # dimension of latent/input space 
        self.model = model   # statistical (numpy) model used
        self.act_func = act_func  # choice of activation function
        self.use_bias = use_bias  # if True, add bias as a variable
        self.bias = 0
        if self.act_func == 'sigmoid' :  # leads to Logit-Normal distribution
            self.g = self.sigmoid
            self.grad_g = self.grad_sigmoid
            self.grad2_g = self.grad2_sigmoid
        if self.act_func == 'exp' :      # leads to Log-Normal distribution
            self.g = self.exp
            self.grad_g = self.grad_exp
            self.grad2_g = self.grad2_exp

        if self.act_func == 'softplus' :  # no usual distribution 
            self.g = self.softplus
            self.grad_g = self.grad_softplus
            self.grad2_g = self.grad2_softplus
                
    # Activation functions and its 1st and 2nd order derivatives
    def sigmoid(self, w, eps):  # w : size = p , eps : shape = (E, p),  p' = p + use_bias
        return (np.tanh(eps @ w + self.bias) + 1) / 2   # output size = E
    
    def grad_sigmoid(self, w, eps): # w : size = p , eps : shape = (E, p)
        grad_weight = (eps.T * (1 - np.tanh(eps@w + self.bias)**2) / 2).T
        if self.use_bias :
            grad_bias = (1 - np.tanh(eps@w + self.bias)**2) / 2
            if np.size(grad_bias) == 1 :
                result = np.append(grad_weight,grad_bias)
            else :
                result = np.append(grad_weight,grad_bias[:,None],axis=1)
        else :
            result = grad_weight
        return result   # output shape = (E, p')
        
    def grad2_sigmoid(self, w, eps):  # w : size = p , eps : size = p 
        tanh = np.tanh(eps@w + self.bias)
        Eps = eps[None,:]*eps[:,None]
        if not self.use_bias :
            Matr = Eps
        else :
            Matr = np.block([[Eps,        eps[:,None]],
                             [eps[None,:],np.array([1])]])
        hessian = - Matr * tanh * (1-tanh**2)
        prod_grad = 0.25 * Matr * (1-tanh**2)**2
        return hessian, prod_grad   # output 2 matrices of shape (p', p')

    def exp(self, w, eps):  # w : size = p , eps : shape = (E, p),  p' = p + use_bias
        return np.exp(eps @ w + self.bias)  # output size = E

    def grad_exp(self, w, eps): # w : size = p , eps : shape = (E, p)
        grad_weight = (eps.T * np.exp(eps @ w + self.bias) ).T
        if self.use_bias :
            grad_bias = np.exp(eps @ w + self.bias)
            if np.size(grad_bias) == 1 :
                result = np.append(grad_weight,grad_bias)
            else :
                result = np.append(grad_weight,grad_bias[:,None],axis=1)
        else :
            result = grad_weight
        return result   # output shape = (E, p')

    def grad2_exp(self, w, eps):  # w : size = p , eps : size = p 
        g_term = np.exp(eps@w + self.bias)
        Eps = eps[None,:]*eps[:,None]
        if not self.use_bias :
            Matr = Eps
        else :
            Matr = np.block([[Eps,        eps[:,None]],
                             [eps[None,:],np.array([1])]])
        hessian = Matr * g_term
        prod_grad = Matr * g_term**2
        return hessian, prod_grad   # output 2 matrices of shape (p', p')

    def softplus(self, w, eps):  # w : size = p , eps : shape = (E, p),  p' = p + use_bias
        return np.log(1 + np.exp(eps @ w + self.bias))  # output size = E
    
    def grad_softplus(self, w, eps): # w : size = p , eps : shape = (E, p)
        sigm_term = (1+np.tanh(0.5*eps @ w + 0.5*self.bias))/2
        grad_weight = (eps.T * sigm_term ).T
        if self.use_bias :
            grad_bias = sigm_term
            if np.size(grad_bias) == 1 :
                result = np.append(grad_weight,grad_bias)
            else :
                result = np.append(grad_weight,grad_bias[:,None],axis=1)
        else :
            result = grad_weight
        return result   # output shape = (E, p')
    
    def grad2_softplus(self, w, eps):  # w : size = p , eps : size = p 
        sigm_term = (1+np.tanh(0.5*eps @ w + 0.5*self.bias))/2
        tanh2_term = 0.25 * (1 - np.tanh(0.5*eps @ w + 0.5*self.bias)**2)
        Eps = eps[None,:]*eps[:,None]
        if not self.use_bias :
            Matr = Eps
        else :
            Matr = np.block([[Eps,        eps[:,None]],
                             [eps[None,:],np.array([1])]])
        hessian = Matr * tanh2_term
        prod_grad = Matr * sigm_term**2
        return hessian, prod_grad   # output 2 matrices of shape (p', p')

    # Sampling functions
    def implicit_prior_sampler(self, w, T):
        eps = np.random.normal(size=(T, self.p))
        return self.g(w,eps)     # output theta sample of size = T
    
    def MC_marg(self, w, T):  # MC estimation of marginal likelihood
        model = self.model
        theta_sample = self.implicit_prior_sampler(w, T)
        lik = model.collec_lik(theta_sample)
        return np.mean(lik, axis=0)   # output size = J, where (N, J) = shape(model.data)
    
    def MC_grad_marg(self, w, T_dp):  # MC estimation of gradient marginal likelihood
        model = self.model
        epsilon_sample = np.random.normal(size=(T_dp,self.p))
        theta_sample = self.g(w,epsilon_sample)
        grad_log_lik = model.collec_grad_log_lik(theta_sample)
        lik = model.collec_lik(theta_sample)
        marg_grad = np.mean(grad_log_lik[:,None,:] * lik[:,None,:] * self.grad_g(w,epsilon_sample)[:,:,None], axis=0)
        return marg_grad   # output shape = (p, J)
    
    def MC_posterior_stat(self, w, T, phi = lambda x : x):   # default stat : mean
        model = self.model
        theta_sample = self.implicit_prior_sampler(w, T)
        lik = model.collec_lik(theta_sample)
        numerator = phi(theta_sample)[:,None] * lik
        return np.mean(numerator, axis=0) / np.mean(lik, axis=0)
    
    
    
##############################  Variational Approximations with neural nets in Torch  ##############################
    

class VA_NeuralNet():  
    def __init__(self, neural_net, model):
        self.neural_net = neural_net  # neural network used
        self.model = model   # statistical (torch) model used
        self.p = neural_net.input_size   # (int)
        self.q = neural_net.output_size   # (int)
        self.nb_param = sum(param.numel() for param in neural_net.parameters() if param.requires_grad) # number of parameters

    # Sampling functions
    def implicit_prior_sampler(self, T):
        """ Samples from the implicit prior (defined by the neural network)
        Args:
            T (int): Number of samples
        Returns:
           (T,q) tensor: Tensor containing the theta samples
        """
        eps = torch.randn(T, self.p)
        return self.neural_net(eps)
    
    def MC_marg(self, T):  
        """ Monte-Carlo estimation of the marginal likelihood
        Args:
            T (int): Number of MC samples
        Returns:
            J tensor: Marginal estimation for every N-sample in self.data (of shape (N,J) or (N,J,q))
        """
        model = self.model
        theta_sample = self.implicit_prior_sampler(T)
        if self.q == 1 :
            theta_sample = theta_sample.squeeze()
        lik = model.collec_lik(theta_sample)
        return torch.mean(lik, dim=0)  
    
    def MLE_approx(self, T):
        """ Approximates the MLE from T samples of theta (dim q) : maximazises the log-likelihood over {1,...,T},
        then, takes the average over every one of the J MLEs
        Args:
            T (int): Number of theta samples used
        Returns:
            q tensor : The final average of the approximate MLEs
        """
        model = self.model
        theta_sample = self.implicit_prior_sampler(T)
        if self.q == 1 :
            theta_sample = theta_sample.squeeze()
        log_lik = model.collec_log_lik(theta_sample)
        indices = torch.argmax(log_lik, dim=0)
        if self.q == 1 :
            MLEs = theta_sample[indices]
        else :
            MLEs = theta_sample[indices, :]
        return torch.mean(MLEs, dim=0)

    # def MC_grad_marg(self, T_dp, grad_tensor):  # MC estimation of gradient marginal likelihood
    #     model = self.model
    #     epsilon_sample = torch.randn(T_dp,self.p)
    #     theta_sample = self.neural_net(epsilon_sample)
    #     grad_log_lik = model.collec_grad_log_lik(theta_sample)
    #     lik = model.collec_lik(theta_sample)
    #     marg_grad = torch.mean(grad_log_lik[:,None,:] * lik[:,None,:] * grad_tensor[:,:,None], dim=0)
    #     return marg_grad   # output shape = (p, J)
    
    def MC_posterior_stat(self, T, phi = lambda x : x):   # (to be tested)
        """ Computes the expectancy of phi(theta) for theta sampled from the posterior (by MC estimation)
        Args:
            T (int): Number of MC samples
            phi (function, optional): Fonction in the expectancy, defaults to 'mean'.
        Returns:
            J tensor : MC estimation for every N-sample in self.data
        """
        model = self.model
        theta_sample = self.implicit_prior_sampler(T)
        lik = model.collec_lik(theta_sample)
        numerator = phi(theta_sample) * lik
        return torch.mean(numerator, dim=0) / torch.mean(lik, dim=0)
    

    ##########  MCMC, Metropolis-Hastings Random Walk  ########## 

    # The adaptive scheme is inspired from the one used in an exercice of the course "Computational Statistics" (Master MVA)
    def MH_posterior(self, eps_0, T_mcmc, sigma2_0, target_accept=0.4, adap=True, batch_size=50, Cov=False, disable_tqdm=False):
        """ MCMC sampler using the Metropolis-Hastings Random Walk algorithm
        on the latent variable (eps)
        Args:
            eps_0 (p tensor): Initial point
            T_mcmc (int): Number of MCMC iterations
            sigma2_0 (tensor): Initial variance for proposal distribution
            adap (bool, optional): Specify if adaptive MH is wanted. Defaults to True.
            batch_size (int, optional): Frequence of adaptive updates of the variance. Defaults to 50.
            Cov (bool) : Uses a non-diagonal covariance matrix for the proposal. Defaults to False.
        Returns:
            ((T_mcmc,p) tensor, list): The Markov chain of samples and the acceptance rate
        """
        net = self.neural_net
        model = self.model
        eps = torch.zeros((T_mcmc,self.p))
        eps[0,:] = eps_0
        accept = 0
        batch_acc = []
        j = 0
        for t in tqdm(range(0,T_mcmc-1),desc='Metropolis-Hastings iterations',disable=disable_tqdm):
            if not Cov : 
                eps_cand = eps[t,:] + torch.sqrt(sigma2_0)*torch.randn(self.p)
            else :
                cov_matrix = sigma2_0 *(torch.full((self.p, self.p), 0.5)).fill_diagonal_(1.0)
                eps_cand = torch.distributions.MultivariateNormal(eps[t,:], covariance_matrix=cov_matrix).sample()
            log_ratio_prior = -0.5*(torch.sum(eps_cand**2) - torch.sum(eps[t,:]**2))
            log_ratio_lik = model.log_likelihood(net(eps_cand)) - model.log_likelihood(net(eps[t,:]))
            log_ratio = log_ratio_prior + log_ratio_lik
            logU = torch.log(torch.rand(1))
            if logU < log_ratio : 
                eps[t+1,:] = eps_cand
                accept = accept + 1
            else :
                eps[t+1,:] = eps[t,:]
            if adap :
                    if (t+1) % batch_size == 0 :
                        j = j + 1
                        delta_j = min(0.01, j**-0.5)
                        sigma2_0 = sigma2_0 * torch.exp(torch.sign(torch.tensor(accept/batch_size - target_accept))*delta_j)
                        batch_acc.append(accept/batch_size)
                        accept = 0
        return eps, batch_acc
    


    ##########  Var. Approx., Gaussian parametric posterior on latent space  ##########

    def ELBO_post(self, mean, logvar, q_eps):
        net = self.neural_net
        model = self.model
        expec = torch.mean(model.collec_log_lik(net(q_eps).flatten()), dim=0)
        KL = 0.5 * torch.sum(self.p * logvar.exp() + mean**2 - 1 - logvar)
        return expec - KL

    
    def VA_posterior(self, VA_net, num_epochs, lr, n_samples_eps, disable_tqdm=False):
        optimizer = torch.optim.Adam(VA_net.parameters(), lr)
        VA_net.train()
        Loss_list = np.zeros(num_epochs)
        for epoch in tqdm(range(num_epochs), desc='Posterior training', disable=disable_tqdm):
            optimizer.zero_grad()
            mean, logvar = VA_net(self.model.data.flatten())
            q_eps = VA_net.sample_q(n_samples_eps)
            loss = - self.ELBO_post(mean, logvar, q_eps)
            loss.backward()
            optimizer.step()
            Loss_list[epoch] = loss.item()
        return Loss_list