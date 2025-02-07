# python file
from aux_optimizers import *

# packages used 
import numpy as np 
import scipy

# Remark : The "collec" functions are used for MC estimation of the marginal likelihood and its gradient / hessian

##############################  Statistical models (NumPy)  ##############################

##### Binomial likelihood #####

class np_BinomialModel():   # parameter space : (0,1), dimension 1
    def __init__(self, n):
        self.n = n        # binomial parameter
        self.data = None  # binomial sample in memory of shape = (N, J)
        
    def sample(self, theta, N, J):
        """ Samples from the likelihood for a fixed theta value
        Args:
            theta (float): parameter (in (0,1))
            N (int): Number of data samples
            J (int): Number of N-samples (used for MC estimation afterwards)
        Returns:
            (N,J) np.array : Matrix containing all samples, also kept in memory in the class (accessed by self.data)
        """
        D = np.random.binomial(n=self.n, p=theta, size=(N, J))
        self.data = D
        return D
    
    def invalid_sample(self, theta):
        """ Checks if a sample/parameter value is invalid
        Args:
            theta (float): parameter (in (0,1))
        Returns:
            bool: True if the value is invalid
        """
        return (theta == 0. or theta == 1.)

    def sample_Jeffreys(self,n_samples):
        """ Samples from the Jeffreys prior associated with the likelihood
        Args:
            n_samples (int): Number of samples
        Returns:
            (n_samples) np.array : Array containing the samples
        """
        jeffreys_sample = np.random.beta(.5, .5, size=n_samples)
        return jeffreys_sample
    
    def sample_post_Jeffreys(self, X, n_samples):
        """ Samples from the posterior given an array of data X when using the Jeffreys prior
        Args:
            X (np.array): IID samples of distribution Binom(self.n, theta)
            n_samples (int): Number of samples
        Returns:
            np.array : Array containing the samples (Beta distribution)
        """
        N = np.size(X)
        a = 0.5 + np.sum(X)
        b = 0.5 + N*self.n - np.sum(X)
        post_samples = np.random.beta(a, b, size=n_samples)
        return post_samples
    
    def MLE(self):          
        """ Computes the average Maximum Likelihood Estimator of self.data
        Returns:
            float: Average of the MLE of each Xj where Xj is the j-th N-sample in data
        """
        return (1/self.n) * np.mean(self.data)
    
    def likelihood(self, theta): 
        """ Computes the likelihood for one theta value (with the data sample in memory)
        Args:
            theta (float): parameter (in (0,1))
        Returns:
            J array : likelihood value for each N-sample in self.data at theta
        """
        D = self.data
        lik_1D = np.prod(scipy.special.binom(self.n,D) * theta**D * (1 - theta)**(self.n - D), axis=0)
        return lik_1D    
    
    def log_likelihood(self, theta):
        """ Computes the log-likelihood for one theta value (with the data sample in memory)
        without the additive constant
        Args:
            theta (float): parameter (in (0,1))
        Returns:
            J array : log-likelihood value for each N-sample in self.data at theta
        """
        D = self.data
        log_lik_1D = np.sum(D * np.log(theta) + (self.n - D) * np.log(1-theta), axis=0)
        return log_lik_1D
    
    def collec_lik(self, thetas): 
        """ Computes the likelihood for several theta values
        Args:
            thetas (T np.array): A sample of T theta values
        Returns:
            (T,J) np.array : likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        lik_2D = np.prod(scipy.special.binom(self.n,D[None,:,:])*thetas[:, None, None]**D[None, :, :]*(1-thetas[:, None, None])**(self.n-D[None,   :,:]),axis=1)
        return lik_2D   
    
    def collec_log_lik(self, thetas): 
        """ Computes the log-likelihood for several theta values without the constant
        Args:
            thetas (T np.array): A sample of T theta values
        Returns:
            (T,J) np.array : log-likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        log_lik_2D = np.sum(D[None, :, :]*np.log(thetas[:, None, None])+(self.n-D[None,:,:])*np.log(1-thetas[:, None, None]),axis=1)
        return log_lik_2D

    def grad_log_lik(self, theta):  
        """ Computes the 1st order derivative of the log-likelihood for one theta value
        Args:
            theta (float): parameter (in (0,1))
        Returns:
            J np.array : grad log-likelihood value for each N-sample in self.data at theta
        """
        D = self.data     
        grad_1D = np.sum((D/theta) - ((self.n-D)/(1-theta)), axis=0)
        return grad_1D 
    
    def collec_grad_log_lik(self, thetas): 
        """ Computes the 1st order derivative of the log-likelihood for several theta values
        Args:
            thetas (T np.array): A sample of T theta values
        Returns:
            (T,J) np.array : grad log-likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        grad_2D = np.sum((D[None, :, :]/thetas[:, None, None])-((self.n-D[None, :, :])/(1-thetas[:, None, None])), axis=1)
        return grad_2D 
    
    def grad2_log_lik(self, theta): 
        """ Computes the 2nd order derivative of the log-likelihood for one theta value
        Args:
            theta (float): parameter (in (0,1))
        Returns:
            J np.array : grad2 log-likelihood value for each N-sample in self.data at theta
        """
        D = self.data
        grad2_1D = np.sum(-(D/theta**2) - ((self.n-D)/(1-theta)**2), axis=0)
        return grad2_1D  
    
    def collec_grad2_log_lik(self, thetas):  # same thing for a collection of thetas of size = T
        """ Computes the 2nd order derivative of the log-likelihood for several theta values
        Args:
            thetas (T np.array): A sample of T theta values
        Returns:
            (T,J) np.array : grad2 log-likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        grad2_2D = np.sum(-(D[None, :, :]/(thetas[:, None, None])**2)-((self.n-D[None, :, :])/(1-thetas[:, None, None])**2), axis=1)
        return grad2_2D  # output array 2D shape = (T,J)
        
    
###### Normal likelihood with known mean #####
    
class np_NormalModel_variance():  # Parameter theta : variance, hence dimension 1    
    def __init__(self, mu):
        self.mu = mu       # normal mean parameter
        self.data = None   # normal sample in memory of shape = (N, J)
        
    def sample(self, theta, N, J):
        """ Samples from the likelihood for a fixed theta value
        Args:
            theta (float): parameter (in R+*)
            N (int): Number of data samples
            J (int): Number of N-samples (used for MC estimation afterwards)
        Returns:
            (N,J) np.array : Matrix containing all samples, also kept in memory in the class (accessed by self.data)
        """
        D = self.mu + np.sqrt(theta)*np.random.normal(0, 1, size=(N, J))
        self.data = D
        return D
    
    def invalid_sample(self, theta):
        """ Checks if a sample/parameter value is invalid
        Args:
            theta (float): parameter (in R+*)
        Returns:
            bool: True if the value is invalid
        """
        return (theta == 0.)
    
    def density_Jeffreys(self, theta, min_val, max_val):
        """ Computes the pdf of the renormalized Jeffreys prior on [min_val,max_val]
        Args:
            theta (float or array): parameter (in R+*)
            min_val (float): lower bound interval
            max_val (float): upper bound interval
        Returns:
            (same type as theta): probability density values
        """
        return (1/theta) * np.log(max_val/min_val)**-1
    
    # Jeffreys improper : J(theta) prop to 1/theta on R+*

    def sample_post_Jeffreys(self, X, n_samples):
        """ Samples from the posterior given an array of data X when using the Jeffreys prior
        Args:
            X (np.array): IID samples of distribution Normal(self.mu, theta)
            n_samples (int): Number of samples
        Returns:
            np.array : Array containing the samples (Inverse-Gamma distribution)
        """
        N = np.size(X)
        alpha = 0.5 * N
        beta = 0.5 * np.sum((X-self.mu)**2)
        post_samples = scipy.stats.invgamma.rvs(a=alpha, loc=0, scale=beta, size=n_samples)
        return post_samples
    
    def MLE(self):  
        """ Computes the average Maximum Likelihood Estimator of self.data
        Returns:
            float: Average of the MLE of each Xj where Xj is the j-th N-sample in data
        """
        D = self.data
        return np.mean((D - self.mu)**2)
    
    def likelihood(self, theta):  
        """ Computes the likelihood for one theta value (with the data sample in memory)
        Args:
            theta (float): parameter (in R+*)
        Returns:
            J array : likelihood value for each N-sample in self.data at theta
        """
        D = self.data
        lik_1D = np.prod(np.exp(-0.5*(D-self.mu)**2 / theta) / np.sqrt(2*np.pi*theta), axis=0)
        return lik_1D  
    
    def log_likelihood(self, theta):
        """ Computes the log-likelihood for one theta value (with the data sample in memory)
        without the additive constant
        Args:
            theta (float): parameter (in R+*)
        Returns:
            J array : log-likelihood value for each N-sample in self.data at theta
        """
        D = self.data
        log_lik_1D = np.sum(-0.5*np.log(theta) - 0.5*(D-self.mu)**2 / theta, axis=0)
        return log_lik_1D
    
    def collec_lik(self, thetas):  
        """ Computes the likelihood for several theta values
        Args:
            thetas (T np.array): A sample of T theta values
        Returns:
            (T,J) np.array : likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        lik_2D = np.prod(np.exp(-0.5*(D[None, :, :]-self.mu)**2 / thetas[:,None,None]) / np.sqrt(2*np.pi*thetas[:,None, None]), axis=1)
        return lik_2D   
    
    def collec_log_lik(self, thetas): 
        """ Computes the log-likelihood for several theta values without the constant
        Args:
            thetas (T np.array): A sample of T theta values
        Returns:
            (T,J) np.array : log-likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        log_lik_2D = np.sum(-0.5*np.log(thetas[:, None, None])-0.5*(D[None,:,:]-self.mu)**2 / thetas[:,None,None],axis=1)
        return log_lik_2D
    
    def grad_log_lik(self, theta): 
        """ Computes the 1st order derivative of the log-likelihood for one theta value
        Args:
            theta (float): parameter (in R+*)
        Returns:
            J np.array : grad log-likelihood value for each N-sample in self.data at theta
        """
        D = self.data     
        grad_1D = 0.5 * np.sum( (-1/theta) + ((self.mu-D)**2 / theta**2), axis=0)
        return grad_1D  
    
    def collec_grad_log_lik(self, thetas):  
        """ Computes the 1st order derivative of the log-likelihood for several theta values
        Args:
            thetas (T np.array): A sample of T theta values
        Returns:
            (T,J) np.array : grad log-likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        grad_2D = 0.5 * np.sum( (-1/thetas[:, None, None]) + ((self.mu-D[None,:,:])**2 / thetas[:,None, None]**2), axis=1)
        return grad_2D  
    
    
    def grad2_log_lik(self, theta): 
        """ Computes the 2nd order derivative of the log-likelihood for one theta value
        Args:
            theta (float): parameter (in R+*)
        Returns:
            J np.array : grad2 log-likelihood value for each N-sample in self.data at theta
        """
        D = self.data
        grad2_1D = np.sum((0.5/theta**2) - ((self.mu-D)**2 / theta**3), axis=0)
        return grad2_1D  
    
    def collec_grad2_log_lik(self, thetas): 
        """ Computes the 2nd order derivative of the log-likelihood for several theta values
        Args:
            thetas (T np.array): A sample of T theta values
        Returns:
            (T,J) np.array : grad2 log-likelihood value for each N-sample in self.data at each theta in thetas
        """
        D = self.data
        grad2_2D = np.sum((0.5/thetas[:, None, None]**2) - ((self.mu-D[None,:,:])**2 / thetas[:, None, None]**3), axis=1)
        return grad2_2D 
        
    
    

        
    
    