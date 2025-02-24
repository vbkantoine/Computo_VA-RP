# python file
from aux_optimizers import * 

# packages used 
import numpy as np 
import torch
import math
import pandas


##############################  Statistical models (Torch)  ##############################
# Every float (in numpy file) in argument is replaced by a torch tensor of size one


##### Binomial likelihood #####

class torch_BinomialModel():
    def __init__(self, n, numpy_to_torch=False):
        self.n = torch.tensor([n], requires_grad=False)
        self.data = None  # binomial sample in memory of shape = (N, J)
        self.numpy_to_torch = numpy_to_torch  # If True, samples from numpy then convert to torch
                                              # If False, samples using torch.distributions
        self.available_MLE = True

    def sample(self, theta, N, J):
        """ Samples from the likelihood for a fixed theta value
        Args:
            theta (tensor): parameter (in (0,1))
            N (int): Number of data samples
            J (int): Number of N-samples (used for MC estimation afterwards)
        Returns:
            (N,J) tensor : Matrix containing all samples, also kept in memory in the class (accessed by self.data)
        """
        if self.numpy_to_torch : 
            D = torch.tensor(np.random.binomial(n=(self.n).item(), p=theta.item(), size=(N, J)),dtype=torch.float32)
        else :
            distrib = torch.distributions.Binomial(total_count=self.n, probs=theta)
            D = distrib.sample(sample_shape=(N, J))
        D = torch.reshape(D, (N, J))
        self.data = D
        return D
    
    def sample_Jeffreys(self, n_samples):
        """ Samples from the Jeffreys prior associated with the likelihood
        Args:
            n_samples (int): Number of samples
        Returns:
            (n_samples) tensor : Tensor containing the samples
        """
        jeffreys_sample = torch.tensor(np.random.beta(0.5, 0.5, size=n_samples), dtype=torch.float32)
        return jeffreys_sample
    
    def sample_post_Jeffreys(self, X, n_samples):
        """ Samples from the posterior given an array of data X when using the Jeffreys prior
        Args:
            X (tensor): IID samples of distribution Binom(self.n, theta)
            n_samples (int): Number of samples
        Returns:
            tensor : Tensor containing the samples (Beta distribution)
        """
        N = X.size(dim=0)
        a = 0.5 + torch.sum(X)
        b = 0.5 + N*self.n - torch.sum(X)
        post_samples = torch.distributions.Beta(a, b).sample(sample_shape=(n_samples,))
        return post_samples.squeeze(dim=-1)
    
    def MLE(self, m=1e-6, M=1. - 1e-6): 
        """ Computes the average Maximum Likelihood Estimator of self.data
        Returns:
            tensor (size 1): Mean of the MLE of each Xj where Xj is the j-th N-sample in data
        """
        mle = (1/self.n) * torch.mean(self.data)
        return m + (M-m) * mle   # avoid NaN values due to MLE = 0. or 1.
    
    def likelihood(self, theta):
        """ Computes the likelihood for one theta value (with the data sample in memory)
        Args:
            theta (tensor): parameter (in (0,1))
        Returns:
            J tensor : likelihood value for each N-sample in self.data at theta
        """
        D = self.data
        lik_1D = torch.prod(torch_binomialcoeff(self.n, D) * theta**D * (1 - theta)**(self.n - D), dim=0)
        return lik_1D
    
    def log_likelihood(self, theta):
        """ Computes the log-likelihood for one theta value (with the data sample in memory)
        without the additive constant
        Args:
            theta (tensor): parameter (in (0,1))
        Returns:
            J tensor : log-likelihood value for each N-sample in self.data at theta
        """
        D = self.data
        log_lik_1D = torch.sum(D*torch.log(theta) + (self.n - D)*torch.log(1-theta), dim=0)
        return log_lik_1D

    def collec_lik(self, thetas):
        """ Computes the likelihood for several theta values
        Args:
            thetas (T tensor): A sample of T theta values
        Returns:
            (T,J) tensor : likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        lik_2D = torch.prod(torch_binomialcoeff(self.n, D[None,:,:]) * thetas[:, None, None]**D[None,:,:] * (1 - thetas[:, None, None])**(self.n - D[None,:,:]), dim=1)
        return lik_2D
    
    def collec_log_lik(self, thetas):
        """ Computes the log-likelihood for several theta values
        Args:
            thetas (T tensor): A sample of T theta values
        Returns:
            (T,J) tensor : log-likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        log_lik_2D = torch.sum(D[None,:,:]*torch.log(thetas[:,None,None]) +  (self.n-D[None,:,:])*torch.log(1-thetas[:,None,None]), dim=1)
        return log_lik_2D
    
    def grad_log_lik(self, theta):
        """ Computes the 1st order derivative of the log-likelihood for one theta value
        Args:
            theta (tensor): parameter (in (0,1))
        Returns:
            J tensor : grad log-likelihood value for each N-sample in self.data at theta
        """
        D = self.data
        grad_1D = torch.sum((D / theta) - ((self.n - D) / (1 - theta)), dim=0)
        return grad_1D
    
    def collec_grad_log_lik(self, thetas):
        """ Computes the 1st order derivative of the log-likelihood for several theta values
        Args:
            thetas (T tensor): A sample of T theta values
        Returns:
            (T,J) tensor : grad log-likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        grad_2D = torch.sum((D[None,:,:] / thetas[:, None, None]) - ((self.n - D[None,:,:]) / (1 - thetas[:, None, None])), dim=1)
        return grad_2D
    
    def grad2_log_lik(self, theta):
        """ Computes the 2nd order derivative of the log-likelihood for one theta value
        Args:
            theta (tensor): parameter (in (0,1))
        Returns:
            J tensor : grad2 log-likelihood value for each N-sample in self.data at theta
        """
        D = self.data
        grad2_1D = torch.sum(-(D / theta**2) - ((self.n - D) / (1 - theta)**2), dim=0)
        return grad2_1D
    
    def collec_grad2_log_lik(self, thetas):
        """ Computes the 2nd order derivative of the log-likelihood for several theta values
        Args:
            thetas (T tensor): A sample of T theta values
        Returns:
            (T,J) tensor : grad2 log-likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        grad2_2D = torch.sum(-(D[None,:,:] / (thetas[:, None, None])**2) - ((self.n - D[None,:,:]) / (1 - thetas[:, None, None])**2), dim=1)
        return grad2_2D
    
###### Normal likelihood with known mean #####
    
class torch_NormalModel_variance():  # Parameter theta : variance, hence dimension 1      
    def __init__(self, mu):
        self.mu = mu       # normal mean parameter (float)
        self.data = None   # normal sample in memory of shape = (N, J)
        self.available_MLE = True

    def sample(self, theta, N, J):
        """ Samples from the likelihood for a fixed theta value
        Args:
            theta (tensor): parameter (in R+*)
            N (int): Number of data samples
            J (int): Number of N-samples (used for MC estimation afterwards)
        Returns:
            (N,J) tensor : Matrix containing all samples, also kept in memory in the class (accessed by self.data)
        """
        D = self.mu + torch.sqrt(theta)*torch.randn(N, J)
        self.data = D
        return D
    

    # Jeffreys improper : J(theta) prop to 1/theta on R+
    
    def density_Jeffreys(self, theta, min_val, max_val):
        """ Computes the pdf of the renormalized Jeffreys prior on [min_val,max_val]
        Args:
            theta (tensor): parameter (in R+*)
            min_val (tensor): lower bound interval
            max_val (tensor): upper bound interval
        Returns:
            (same type and size as theta): probability density values
        """
        return (1/theta) * torch.log(max_val/min_val)**-1
    
    def sample_post_Jeffreys(self, X, n_samples, delta=1):
        """ Samples from the posterior given an array of data X when using the Jeffreys prior
        Args:
            X (tensor): IID samples of distribution Normal(self.mu, theta)
            n_samples (int): Number of samples
        Returns:
            tensor : Tensor containing the samples (Inverse-Gamma distribution)
        """
        N = X.size(dim=0)
        alpha = 0.5 * N + delta - 1
        beta = 0.5 * torch.sum((X-self.mu)**2)
        post_samples = torch.distributions.InverseGamma(alpha, beta).sample(sample_shape=(n_samples,))
        return post_samples.squeeze(dim=-1)
    
    def MLE(self):              
        """ Computes the average Maximum Likelihood Estimator of self.data
        Returns:
            tensor (size 1) : Mean of the MLE of each Xj where Xj is the j-th N-sample in data
        """
        D = self.data
        return torch.mean((D - self.mu)**2)
    
    def likelihood(self, theta): 
        """ Computes the likelihood for one theta value (with the data sample in memory)
        Args:
            theta (tensor): parameter (in R+*)
        Returns:
            J tensor : likelihood value for each N-sample in self.data at theta
        """ 
        D = self.data
        lik_1D = torch.prod(torch.exp(-0.5*(theta**-1)*(D-self.mu)**2) / torch.sqrt(2*torch.tensor(math.pi)*theta), dim=0)
        return lik_1D   
    
    def log_likelihood(self, theta):
        """ Computes the log-likelihood for one theta value (with the data sample in memory)
        without the additive constant
        Args:
            theta (tensor): parameter (in R+*)
        Returns:
            J tensor : log-likelihood value for each N-sample in self.data at theta
        """
        D = self.data
        log_lik_1D = torch.sum(-0.5*torch.log(theta) - 0.5*(D-self.mu)**2 / theta, dim=0)
        return log_lik_1D
    
    def collec_lik(self, thetas):
        """ Computes the likelihood for several theta values
        Args:
            thetas (T tensor): A sample of T theta values
        Returns:
            (T,J) tensor : likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        lik_2D = torch.prod(torch.exp(-0.5*(thetas[:,None,None]**-1)*(D[None, :, :]-self.mu)**2) / torch.sqrt(2*torch.tensor(math.pi)*thetas[:, None, None]), dim=1)
        return lik_2D    
    
    def collec_log_lik(self, thetas):
        """ Computes the log-likelihood for several theta values
        Args:
            thetas (T tensor): A sample of T theta values
        Returns:
            (T,J) tensor : log-likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        log_lik_2D = torch.sum(-0.5*torch.log(thetas[:,None,None]) - 0.5*(D[None, :, :]-self.mu)**2 / thetas[:,None,None], dim=1)
        return log_lik_2D

    def grad_log_lik(self, theta):
        """ Computes the 1st order derivative of the log-likelihood for one theta value
        Args:
            theta (tensor): parameter (in R+*)
        Returns:
            J tensor : grad log-likelihood value for each N-sample in self.data at theta
        """ 
        D = self.data     
        grad_1D = 0.5 * torch.sum( (-1/theta) + ((self.mu-D)**2 / theta**2), dim=0)
        return grad_1D   
    
    def collec_grad_log_lik(self, thetas):  
        """ Computes the 1st order derivative of the log-likelihood for several theta values
        Args:
            thetas (T tensor): A sample of T theta values
        Returns:
            (T,J) tensor : grad log-likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        grad_2D = 0.5 * torch.sum( (-1/thetas[:, None, None]) + ((self.mu-D[None,:,:])**2 / thetas[:, None, None]**2), dim=1)
        return grad_2D  
    
    def grad2_log_lik(self, theta): 
        """ Computes the 2nd order derivative of the log-likelihood for one theta value
        Args:
            theta (tensor): parameter (in R+*)
        Returns:
            J tensor : grad2 log-likelihood value for each N-sample in self.data at theta
        """
        D = self.data
        grad2_1D = torch.sum((0.5/theta**2) - ((self.mu-D)**2 / theta**3), dim=0)
        return grad2_1D 
    
    def collec_grad2_log_lik(self, thetas): 
        """ Computes the 2nd order derivative of the log-likelihood for several theta values
        Args:
            thetas (T tensor): A sample of T theta values
        Returns:
            (T,J) tensor : grad2 log-likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        grad2_2D = torch.sum((0.5/thetas[:, None, None]**2) - ((self.mu-D[None,:,:])**2 / thetas[:, None, None]**3), dim=1)
        return grad2_2D


###### Normal likelihood (unknown mean and variance) #####
    
class torch_NormalModel():  # Parameter theta = (mu=mean, sigma2=variance), dimension 2
    def __init__(self):
        self.data = None   # normal sample in memory of shape = (N, J)
        self.available_MLE = True

    def sample(self, theta, N, J):
        """ Samples from the likelihood for a fixed theta value
        Args:
            theta (2 tensor): parameter (in R x R+*)
            N (int): Number of data samples
            J (int): Number of N-samples (used for MC estimation afterwards)
        Returns:
            (N,J) tensor : Matrix containing all samples, also kept in memory in the class (accessed by self.data)
        """
        mu = theta[0]
        sigma2 = theta[1]
        D = mu + torch.sqrt(sigma2)*torch.randn(N, J)
        self.data = D
        return D
    

    # Jeffreys improper : J(mu,sigma2) prop to (1/sigma2)^(3/2) on R x R+*
    
    def density_Jeffreys_mean(self, mu, min_val, max_val):
        """ Computes the pdf of the renormalized Jeffreys prior on [min_val,max_val]
        Args:
            mu (tensor): parameter (in R), not needed since constant in mu
            min_val (tensor): lower bound interval
            max_val (tensor): upper bound interval
        Returns:
            (same type and size as mu): probability density values
        """
        return 1/(max_val - min_val) * torch.ones(mu.size())
    
    def density_Jeffreys_var(self, sigma2, min_val, max_val):
        """ Computes the pdf of the renormalized Jeffreys prior on [min_val,max_val]
        Args:
            sigma2 (tensor): parameter (in R+*)
            min_val (tensor): lower bound interval
            max_val (tensor): upper bound interval
        Returns:
            (same type and size as sigma2): probability density values
        """
        return -0.5*(max_val**(-1/2) - min_val**(-1/2))**(-1) * sigma2**(-3/2)
    
    def sample_post_Jeffreys(self, X, n_samples, delta=1.5):
        """ Samples from the posterior given an array of data X when using the Jeffreys prior
        Args:
            X (tensor): IID samples of distribution Normal(theta=(mu,sigma2))
            n_samples (int): Number of samples
        Returns:
            (n_samples, 2) tensor : Tensor containing the samples (Normal-Inverse-Gamma distribution)
        """
        N = X.size(dim=0)
        m = torch.mean(X)
        alpha, beta = 0.5*N + delta - 1.5, 0.5*N*torch.mean((X-m)**2)
        Z = torch.randn(n_samples)
        sigma2 = torch.distributions.InverseGamma(alpha, beta).sample(sample_shape=(n_samples,))
        sigma2 = sigma2.squeeze(dim=-1)
        mu = m + torch.sqrt(sigma2 / N) * Z
        post_samples = torch.stack((mu, sigma2), dim=-1)
        return post_samples 
    
    def MLE(self):              
        """ Computes the average Maximum Likelihood Estimator of self.data
        Returns:
            tensor (size 2) : Mean of the MLE of each Xj where Xj is the j-th N-sample in data
        """
        D = self.data
        mu_MLE = torch.mean(D)
        sigma2_MLE = torch.mean((D - mu_MLE)**2)
        return torch.cat((mu_MLE.unsqueeze(0), sigma2_MLE.unsqueeze(0)))
    
    def likelihood(self, theta): 
        """ Computes the likelihood for one theta value (with the data sample in memory)
        Args:
            theta (2 tensor): parameter (in R x R+*)
        Returns:
            J tensor : likelihood value for each N-sample in self.data at theta
        """ 
        D = self.data
        mu = theta[0]
        sigma2 = theta[1]
        lik_1D = torch.prod(torch.exp(-0.5*(sigma2**-1)*(D-mu)**2) / torch.sqrt(2*torch.tensor(math.pi)*sigma2), dim=0)
        return lik_1D   
    
    def log_likelihood(self, theta):
        """ Computes the log-likelihood for one theta value (with the data sample in memory)
        without the additive constant
        Args:
            theta (tensor): parameter (in R+*)
        Returns:
            J tensor : log-likelihood value for each N-sample in self.data at theta
        """
        D = self.data
        mu = theta[0]
        sigma2 = theta[1]
        log_lik_1D = torch.sum(-0.5*torch.log(sigma2) - 0.5*(D-mu)**2 / sigma2, dim=0)
        return log_lik_1D

    def collec_lik(self, thetas):
        """ Computes the likelihood for several theta values
        Args:
            thetas ((T,2) tensor): A sample of T theta values
        Returns:
            (T,J) tensor : likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        mu = thetas[:,0]
        sigma2 = thetas[:,1]
        lik_2D = torch.prod(torch.exp(-0.5*(sigma2[:,None,None]**-1)*(D[None, :, :]-mu[:,None,None])**2) / torch.sqrt(2*torch.tensor(math.pi)*sigma2[:, None, None]), dim=1)
        return lik_2D    
    
    def collec_log_lik(self, thetas):
        """ Computes the log-likelihood for several theta values
        Args:
            thetas ((T,2) tensor): A sample of T theta values
        Returns:
            (T,J) tensor : log-likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        mu = thetas[:,0]
        sigma2 = thetas[:,1]
        log_lik_2D = torch.sum(-0.5*torch.log(sigma2[:,None,None]) - 0.5*(D[None,:,:]-mu[:,None,None])**2 / sigma2[:,None,None], dim=1)
        return log_lik_2D
    
    def grad_log_lik(self, theta):
        """ Computes the 1st order derivative of the log-likelihood for one theta value
        Args:
            theta (2 tensor): parameter (in R x R+*)
        Returns:
            (J,2) tensor : grad log-likelihood value for each N-sample in self.data at theta
        """ 
        D = self.data
        N,J = D.size()
        mu = theta[0]
        sigma2 = theta[1]     
        grad_mu = sigma2**(-1)* torch.sum((D - mu), dim=0)
        grad_sigma2 = 0.5 * torch.sum(- sigma2**(-1) + sigma2**(-2)*(D-mu)**2, dim=0)
        result = torch.cat((grad_mu, grad_sigma2), dim=-1)
        return torch.reshape(result, (J,2))   
    
    def collec_grad_log_lik(self, thetas):  
        """ Computes the 1st order derivative of the log-likelihood for several theta values
        Args:
            thetas ((T,2) tensor): A sample of T theta values
        Returns:
            (T,J,2) tensor : grad log-likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        N,J = D.size()
        T,_ = thetas.size()
        mu = thetas[:,0]
        sigma2 = thetas[:,1]
        grad_mu = torch.sum(sigma2[:,None,None]**(-1)*(D[None, :, :] - mu[:,None,None]), dim=1)
        grad_sigma2 = 0.5 * torch.sum(- sigma2[:,None,None]**(-1) + sigma2[:,None,None]**(-2)*(D[None, :, :]-mu[:,None,None])**2, dim=1)
        result = torch.cat((grad_mu, grad_sigma2), dim=-1)
        return torch.reshape(result, (T,J,2))
 
    
##### Multinomial Likelihood #####

class torch_MultinomialModel():
    def __init__(self, n, q):
        self.n = n    # number of trials
        self.q = q    # dimension of theta (probability tensor)
        self.data = None  # (N,J) multinomial samples, hence a (N,J,q) tensor kept in memory
        self.available_MLE = True

    def sample(self, theta, N, J):
        """ 
        Args:
            theta (q tensor): parameter (probability tensor)
            N (int): Number of data samples
            J (int): Number of N-samples (used for MC estimation afterwards)
        Returns:
            (N,J,q) tensor: Tensor containing all samples and is kept in the class
        """
        m = torch.distributions.Multinomial(total_count=self.n, probs=theta)
        D = m.sample((N, J))
        self.data = D
        return D
    
    def sample_Jeffreys(self, n_samples):
            """ Samples from the Jeffreys prior associated with the likelihood
            Args:
                n_samples (int): Number of samples
            Returns:
                (n_samples,q) tensor : Tensor containing the samples
            """
            m = torch.distributions.Dirichlet(torch.tensor(self.q*[0.5]))
            jeffreys_sample = m.sample((n_samples,))
            return jeffreys_sample
    
    def sample_post_Jeffreys(self, X, n_samples):
        """ Samples from the posterior given an array of data X when using the Jeffreys prior
        Args:
            X ((_,q) tensor): IID samples of distribution Multinom(self.n, theta)
            n_samples (int): Number of samples
        Returns:
            (n_samples,q) tensor : Tensor containing the samples (Dirichlet distribution)
        """
        gamma = torch.tensor(self.q*[0.5]) + torch.sum(X, dim=0)
        post_samples = torch.distributions.Dirichlet(gamma).sample(sample_shape=(n_samples,))
        return post_samples

    def MLE(self): 
        """ Computes the average Maximum Likelihood Estimator of self.data
        Returns:
            q tensor : Mean of the MLE of each Xj where Xj is the j-th N-sample in data
        """
        return (1/self.n) * torch.mean(self.data, dim=(0,1))

    def likelihood(self, theta):
        """ Computes the likelihood for one theta tensor (with the data sample in memory)
        Args:
            theta (q tensor): parameter (probability tensor)
        Returns:
            J tensor : likelihood value for each N-sample in self.data at theta
        """
        D = self.data
        # Compute the log of the multinomial coefficients using the log-gamma function
        log_factorial_n = torch.lgamma(D.sum(dim=2) + 1)
        log_factorial_data = torch.lgamma(D + 1).sum(dim=2)
        log_multinomial_coeff = log_factorial_n - log_factorial_data
        # Compute the log-likelihood
        log_likelihood = log_multinomial_coeff + (D * torch.log(theta)).sum(dim=2)
        # Sum over the first dimension to get the total log-likelihood
        total_log_likelihood = log_likelihood.sum(dim=0)
        lik_1D = torch.exp(total_log_likelihood)
        return lik_1D
    
    def log_likelihood(self, theta):
        """ Computes the log-likelihood for one theta tensor (with the data sample in memory)
        without the additive constant
        Args:
            theta (q tensor): parameter (probability tensor)
        Returns:
            J tensor : log-likelihood value for each N-sample in self.data at theta
        """
        D = self.data
        # Compute the log-likelihood
        log_likelihood = (D * torch.log(theta)).sum(dim=2)
        # Sum over the first dimension to get the total log-likelihood
        log_lik_1D = log_likelihood.sum(dim=0)
        return log_lik_1D

    def collec_lik(self, thetas):
        """ Computes the likelihood for several theta tensors
        Args:
            thetas ((T,q) tensor): A sample of T theta tensors
        Returns:
            (T,J) tensor : likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        # Compute the log of the multinomial coefficients using the log-gamma function
        log_factorial_n = torch.lgamma(D.sum(dim=2) + 1)
        log_factorial_data = torch.lgamma(D + 1).sum(dim=2)
        log_multinomial_coeff = log_factorial_n - log_factorial_data
        # Expand theta to match the dimensions of data for broadcasting
        log_thetas = torch.log(thetas).unsqueeze(1).unsqueeze(1)  # Shape (T, 1, 1, q)
        # Compute the log-likelihood
        log_term = (D.unsqueeze(0) * log_thetas).sum(dim=3)
        log_likelihood = log_multinomial_coeff.unsqueeze(0) + log_term
        # Sum over the first dimension to get the total log-likelihood
        total_log_likelihood = log_likelihood.sum(dim=1)
        lik_2D = torch.exp(total_log_likelihood)
        return lik_2D
    
    def collec_log_lik(self, thetas):
        """ Computes the log-likelihood for several theta tensors
        Args:
            thetas ((T,q) tensor): A sample of T theta tensors
        Returns:
            (T,J) tensor : log-likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        # Expand theta to match the dimensions of data for broadcasting
        log_thetas = torch.log(thetas).unsqueeze(1).unsqueeze(1)  # Shape (T, 1, 1, q)
        # Compute the log-likelihood
        log_term = (D.unsqueeze(0) * log_thetas).sum(dim=3)
        # Sum over the first dimension to get the total log-likelihood
        lik_2D = log_term.sum(dim=1)
        return lik_2D

    def grad_log_lik(self, theta):
        """ Computes the 1st order derivative of the log-likelihood for one theta tensor
        Args:
            theta (q tensor): parameter (probability tensor)
        Returns:
            (J,q) tensor : grad log-likelihood value for each N-sample in self.data at theta
        """
        D = self.data
        grad_1D = D.sum(dim=0) / theta
        return grad_1D
    
    def collec_grad_log_lik(self, thetas):
        """ Computes the 1st order derivative of the log-likelihood for several theta tensors
        Args:
            thetas ((T,q) tensor): A sample of T theta tensors
        Returns:
            (T,J,q) tensor : grad log-likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        grad_2D = (D.sum(dim=0)).unsqueeze(0) / thetas.unsqueeze(1)
        return grad_2D
    

### Probit likelihood (fragility curves) ###

class torch_ProbitModel():               # theta = (alpha,beta) in (R+*)^2 or theta = alpha depending on set_beta
    def __init__(self, use_log_normal=True, mu_a=None, sigma2_a=None, set_beta=None, alt_scaling=False):
        self.use_log_normal = use_log_normal  # deprecated argument, always has to be True to run the code
        self.mu_a = mu_a         # mean parameter of LogNormal for a 
        self.sigma2_a = sigma2_a  # variance parameter of LogNormal for a
        self.data = None   # (N,J) probit samples, hence a (N,J,2) tensor kept in memory
        self.available_MLE = False
        self.set_beta = set_beta   # Imposes a value on beta
        self.alt_scaling = alt_scaling  # alternative scaling on beta (mult. by sqrt(2))
        self.Gauss_cdf = None
        self.Gauss_pdf = None
        # if not self.use_log_normal : no real data is used in the presented experiments
        #     df = pandas.read_csv('KH_xi=2%_sa.csv')
        #     self.pga = df['PGA']
        if not self.alt_scaling :
            self.Gauss_cdf = gaussian_cdf
            self.Gauss_pdf = gaussian_pdf
        else : 
            self.Gauss_cdf = gaussian_cdf_alt
            self.Gauss_pdf = gaussian_pdf_alt

    def count_degenerate_samples(self, D):
        Z, a = D[:,:,0], D[:,:,1]
        sorted_a, sorted_indices = torch.sort(a, dim=0)
        sorted_Z = torch.gather(Z, 0, sorted_indices)
        count_separated = 0
        for i in range(sorted_Z.shape[1]):
            col = sorted_Z[:,i]
            # Find the index where the first one appears
            first_one_index = torch.nonzero(col == 1, as_tuple=True)[0]
            # If there are no ones, check if all elements are zeros
            if len(first_one_index) == 0:
                if torch.all(col == 0).item():
                    count_separated += 1
            else:
                first_one_index = first_one_index[0].item()
                # Check if all elements before this index are zeros and all elements from this index are ones
                is_separated = torch.all(col[:first_one_index] == 0).item() and torch.all(col[first_one_index:] == 1).item()
                if is_separated:
                    count_separated += 1
        
        return count_separated
    

    def sample(self, theta, N, J):
        """ 
        Args:
            theta (1or2 tensor): parameter (probability tensor)
            N (int): Number of data samples
            J (int): Number of N-samples (used for MC estimation afterwards)
        Returns:
            (N,J,2) tensor: Tensor containing all samples and is kept in the class
        """
        if theta.dim() > 1 :
            # consider using lognormal and set_beta is None
            alpha = theta[:,0]
            beta = theta[:,1]
            m1 = torch.distributions.LogNormal(self.mu_a, self.sigma2_a)
            a = m1.sample((theta.size()[0],N,J))
            gamma = torch.log(a/alpha[:,None,None])/beta[:,None,None]
            m2 = torch.distributions.Bernoulli(probs=self.Gauss_cdf(gamma))
            Z = m2.sample()
            D = torch.stack((Z,a), dim=-1).squeeze()
            # D = torch.reshape(D, (N,J,2))
            self.data = D
            return D

        if self.set_beta is None :
            alpha, beta = theta[0], theta[1]
        else :
            alpha, beta = theta, self.set_beta
        if self.use_log_normal :
            m1 = torch.distributions.LogNormal(self.mu_a, self.sigma2_a)
            a = m1.sample((N, J))
        else : 
            a = torch.tensor((self.pga.sample(N*J)).to_numpy(), dtype=torch.float32)
            a = torch.reshape(a, (N,J))
        gamma = torch.log(a/alpha) / beta
        m2 = torch.distributions.Bernoulli(probs=self.Gauss_cdf(gamma))
        Z = m2.sample()
        D = torch.stack((Z,a), dim=-1)
        D = torch.reshape(D, (N,J,2))
        self.data = D
        return D
    

    def likelihood(self, theta):
        """ Computes the likelihood for one theta tensor (with the data sample in memory)
        Args:
            theta (1or2 tensor): parameter (in (R+*) or (R+*)^2)
        Returns:
            J tensor : likelihood value for each N-sample in self.data at theta
        """
        D = self.data
        Z, a = D[:,:,0], D[:,:,1]
        if self.set_beta is None :
            alpha, beta = theta[0], theta[1]
        else :
            alpha, beta = theta, self.set_beta
        Phi = self.Gauss_cdf(torch.log(a/alpha) / beta)
        #lik_lognormal = torch.prod(torch.exp(-0.5*(torch.log(a) - self.mu_a)**2 / self.sigma2_a) / (a*torch.sqrt(torch.tensor(2*math.pi*self.sigma2_a))), dim=0) 
        lik_cond = torch.prod(Phi**Z * (1-Phi)**(1-Z), dim=0)
        lik_1D = lik_cond #* lik_lognormal  
        return lik_1D

    def collec_lik(self, thetas):
        """ Computes the likelihood for several theta values
        Args:
            thetas (T or (T,2) tensor): A sample of T theta values
        Returns:
            (T,J) tensor : likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        Z, a = D[:,:,0], D[:,:,1]
        if self.set_beta is None :
            alpha, beta = thetas[:,0], thetas[:,1]
            Phi = self.Gauss_cdf(torch.log(a[None, :, :]/alpha[:, None, None]) / beta[:, None, None])
        else :
            alpha, beta = thetas, self.set_beta
            Phi = self.Gauss_cdf(torch.log(a[None, :, :]/alpha[:, None, None]) / beta)
        #lik_lognormal = torch.prod(torch.exp(-0.5*(torch.log(a[None, :, :]) - self.mu_a)**2 / self.sigma2_a) / (a[None, :, :]*torch.sqrt(torch.tensor(2*math.pi*self.sigma2_a))), dim=1) 
        lik_cond = torch.prod(Phi**Z[None, :, :] * (1-Phi)**(1-Z[None, :, :]), dim=1)
        lik_2D = lik_cond #* lik_lognormal 
        return lik_2D

    def log_likelihood(self, theta):
        """ Computes the log-likelihood for one theta value (with the data sample in memory)
        without the additive constant
        Args:
            theta (1or2 tensor): parameter (in (R+*) or (R+*)^2)
        Returns:
            J tensor : log-likelihood value for each N-sample in self.data at theta
        """
        if theta.dim()>1 :
            D = self.data
            Z, a = D[:,:,:,0], D[:,:,:,1]    
            alpha, beta = theta[:,0], theta[:,1]
            Phi = self.Gauss_cdf(torch.log(a/alpha[:,None,None]) / beta[:,None,None])
            #log_lik_lognormal = torch.sum(-torch.log(a) - (0.5*(torch.log(a)-self.mu_a)**2 / self.sigma2_a), dim=0)
            log_lik_cond = torch.sum(Z*torch.log(Phi) + (1-Z)*torch.log(1-Phi), dim=1)
            log_lik_1D = log_lik_cond #+ log_lik_lognormal 
            return log_lik_1D

        D = self.data
        Z, a = D[:,:,0], D[:,:,1]
        if self.set_beta is None :
            alpha, beta = theta[0], theta[1]
        else :
            alpha, beta = theta, self.set_beta
        Phi = self.Gauss_cdf(torch.log(a/alpha) / beta)
        #log_lik_lognormal = torch.sum(-torch.log(a) - (0.5*(torch.log(a)-self.mu_a)**2 / self.sigma2_a), dim=0)
        log_lik_cond = torch.sum(Z*torch.log(Phi) + (1-Z)*torch.log(1-Phi), dim=0)
        log_lik_1D = log_lik_cond #+ log_lik_lognormal  
        return log_lik_1D

    def collec_log_lik(self, thetas):
        """ Computes the log-likelihood for several theta values
        Args:
            thetas (T or (T,2) tensor): A sample of T theta values
        Returns:
            (T,J) tensor : log-likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        if D.dim()>3 :
            Z, a = D[:,:,:,0], D[:,:,:,1]
            alpha, beta = thetas[:,0], thetas[:,1]
            Phi = self.Gauss_cdf(torch.log(a[:,None, :, :]/alpha[None,:, None, None]) / beta[None, :, None, None])
            log_lik_cond = torch.sum(Z[:,None, :, :]*torch.log(Phi) + (1-Z[:,None, :, :])*torch.log(1-Phi), dim=2)
            return log_lik_cond

        Z, a = D[:,:,0], D[:,:,1]
        if self.set_beta is None :
            alpha, beta = thetas[:,0], thetas[:,1]
            Phi = self.Gauss_cdf(torch.log(a[None, :, :]/alpha[:, None, None]) / beta[:, None, None])
        else :
            alpha, beta = thetas, self.set_beta
            Phi = self.Gauss_cdf(torch.log(a[None, :, :]/alpha[:, None, None]) / beta)
        #log_lik_lognormal = torch.sum(-torch.log(a[None, :, :]) - (0.5*(torch.log(a[None, :, :])-self.mu_a)**2 / self.sigma2_a), dim=1)
        log_lik_cond = torch.sum(Z[None, :, :]*torch.log(Phi) + (1-Z[None, :, :])*torch.log(1-Phi), dim=1)
        log_lik_2D = log_lik_cond #+ log_lik_lognormal 
        return log_lik_2D

    def grad_log_lik(self, theta):
        """ Computes the 1st order derivative of the log-likelihood for one theta value
        Args:
            theta (1or2 tensor): parameter (in (R+*) or (R+*)^2)
        Returns:
            J or (J,2) tensor : grad log-likelihood value for each N-sample in self.data at theta
        """ 
        D = self.data
        N, J, _ = D.size()
        Z, a = D[:,:,0], D[:,:,1]
        if self.set_beta is None :
            alpha, beta = theta[0], theta[1]
        else :
            alpha, beta = theta, self.set_beta
        gamma = torch.log(a/alpha) / beta
        Phi = self.Gauss_cdf(gamma)
        Phi_prime = self.Gauss_pdf(gamma) 
        grad_alpha = torch.sum((-Z*(Phi_prime/Phi) + (1-Z)*(Phi_prime/(1-Phi))) / (alpha*beta), dim=0) 
        if self.set_beta is None :
            grad_beta = torch.sum(gamma*(-Z*(Phi_prime/Phi) + (1-Z)*(Phi_prime/(1-Phi))) / beta, dim=0) 
            result = torch.stack((grad_alpha, grad_beta), dim=-1)
            return torch.reshape(result, (J, 2))   
        else :
            return grad_alpha

    def collec_grad_log_lik(self, thetas):  
        """ Computes the 1st order derivative of the log-likelihood for several theta values
        Args:
            thetas (T or (T,2) tensor): A sample of T theta values
        Returns:
           (T,J) or (T,J,2) tensor : grad log-likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        N, J, _ = D.size()
        T,_ = thetas.size()
        Z, a = D[:,:,0], D[:,:,1]
        if self.set_beta is None :
            alpha, beta = thetas[:,0], thetas[:,1]
            gamma = torch.log(a[None, :, :]/alpha[:, None, None]) / beta[:, None, None]
        else :
            alpha, beta = thetas, self.set_beta
            gamma = torch.log(a[None, :, :]/alpha[:, None, None]) / beta
        Phi = self.Gauss_cdf(gamma)
        Phi_prime = self.Gauss_pdf(gamma)
        if self.set_beta is None :
            grad_alpha = torch.sum(-Z[None, :, :]*(Phi_prime/Phi) + (1-Z[None, :, :])*(Phi_prime/(1-Phi)) / (alpha[:, None, None]*beta[:, None, None]), dim=1)
            grad_beta = torch.sum(gamma*(-Z[None, :, :]*(Phi_prime/Phi) + (1-Z[None, :, :])*(Phi_prime/(1-Phi))) / beta[:, None, None], dim=1)
            result = torch.stack((grad_alpha, grad_beta), dim=-1)
            return torch.reshape(result, (T,J,2))
        else :
            grad_alpha = torch.sum(-Z[None, :, :]*(Phi_prime/Phi) + (1-Z[None, :, :])*(Phi_prime/(1-Phi)) / (alpha[:, None, None]*beta), dim=1)
            return torch.reshape(grad_alpha, (T,J))

