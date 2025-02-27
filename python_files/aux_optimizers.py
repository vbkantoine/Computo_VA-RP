# packages used
import numpy as np 
import torch
import scipy
import math
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import pairwise_distances

####################  Auxiliary functions  ####################


def seed_all(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def has_nan_params(neural_net):
    for param in neural_net.parameters():
        if torch.isnan(param).any():
            return True
    return False

def delete_infs(tensor) :
    mask = torch.isfinite(tensor)
    return tensor[mask]

def assign_parameters_to_NN(NN, all_params):
    start_idx = 0
    for param in NN.parameters():
        param_size = param.numel()
        param_shape = param.size()
        # Extract the corresponding parameters from the flat tensor
        param.data = (all_params[start_idx:start_idx + param_size]).view(param_shape).data
        # Update the start index for the next parameter
        start_idx += param_size

def delete_nan_elements(x):
    # Create a mask for NaN elements
    mask = torch.isnan(x)
    # Calculate the mean of the non-NaN elements
    non_nan_mean = torch.mean(x[~mask])
    # Replace NaN elements with the mean 
    x_new = x.clone()  
    x_new[mask] = non_nan_mean
    return x_new

def remove_nan_and_inf(arr):
    finite_mask = np.isfinite(arr)
    cleaned_array = arr[finite_mask]
    return cleaned_array


def alpha_div(x, alpha, c, d=0.):
    return ((x**alpha - x*alpha - (1-alpha) ) / (alpha * (alpha - 1))) + c * (x - 1) + d


def torch_binomialcoeff(n, k):
    """ Computes the binomial coefficient, compatible with torch operations
    Args:
        n (tensor): int value
        k (K tensor): K int values
    Returns:
        K tensor : Binomial coeff values
    """
    return ((n + 1).lgamma() - (k + 1).lgamma() - ((n - k) + 1).lgamma()).exp()

def gaussian_cdf(x, m=0.0001, M=0.9999):
    phi = 0.5 *(1 + torch.erf(x / math.sqrt(2.)))
    return  m + (M-m)*phi

def gaussian_pdf(x):
    return torch.exp(-0.5*x**2) / torch.sqrt(torch.tensor(2*math.pi))

def gaussian_cdf_alt(x, m=0.0001, M=0.9999):
    phi = 0.5 *(1 + torch.erf(x))
    return  m + (M-m)*phi

def gaussian_pdf_alt(x):
    return torch.exp(-x**2) / torch.sqrt(torch.tensor(math.pi))

def fit_normal_invgamma(theta_sample, update='newton', n_it=5):
    N = theta_sample.size(dim=0)
    mu, sigma2 = theta_sample[:, 0], theta_sample[:, 1]
    m = torch.sum(mu*sigma2**-1) / torch.sum(sigma2**-1)
    k = N / (torch.sum((sigma2**-1) * (mu - m)**2))
    alpha = 2 + (torch.mean(sigma2)**2 / torch.var(sigma2))
    if torch.isnan(alpha) or torch.isinf(alpha) :
        alpha = torch.tensor([1.])
    valid_alpha = alpha.detach().clone()
    sol_term = torch.mean(torch.log(sigma2)) + torch.log(torch.sum(sigma2**-1)) 
    for _ in range(n_it):
        f = torch.digamma(alpha) - torch.log(N*alpha) + sol_term
        df = torch.polygamma(1, alpha) - (1/alpha)
        if update == 'newton':
            alpha = alpha - (f / df)
        if update == 'ML2' :
            alpha = ( (alpha**-1) - (f / (df*alpha**2)) )**-1
        if torch.isnan(alpha) or torch.isinf(alpha) :
            alpha = valid_alpha.detach().clone()
            break
        else :
            valid_alpha = alpha.detach().clone()
    beta = N * alpha / torch.sum(sigma2**-1)
    return m, k, alpha, beta

def inv_digamma(x):
    if x > -2.22 :
        return torch.exp(x) + 0.5
    else : 
        return - 1 / (x - torch.digamma(torch.tensor(1.)))

def fit_dirichlet(theta_sample, method='newton', n_it=5):
    N, q = theta_sample.size()
    log_mean = torch.mean(torch.log(theta_sample), dim=0)
    mean = torch.mean(theta_sample, dim=0)
    mean_sq = torch.mean(theta_sample**2, dim=0)
    old_gamma = mean * (mean - mean_sq) / (mean_sq - mean**2)
    new_gamma = torch.tensor(q*[0.])
    if method == 'fixed_point' : 
        for _ in range(n_it):
            for j in range(q):
                new_gamma[j] = inv_digamma(torch.digamma(torch.sum(old_gamma)) + log_mean[j])
            old_gamma = new_gamma
    if method == 'newton' : 
        g = torch.zeros(q)
        Q = torch.zeros(q)
        H_inv_g = torch.zeros(q)
        for _ in range(n_it):
            z = N * torch.polygamma(1, torch.sum(old_gamma))
            for j in range(q):
                g[j] = N * (torch.digamma(torch.sum(old_gamma)) - torch.digamma(old_gamma[j]) + log_mean[j])
                Q[j] = - N * torch.polygamma(1, old_gamma[j])
            b = torch.sum(g/Q) / (z**-1 + torch.sum(Q**-1))
            for j in range(q):
                H_inv_g[j] = (g[j] - b) / Q[j]
            new_gamma = old_gamma - H_inv_g
            old_gamma = new_gamma
    return new_gamma


def rbf_kernel(x, y, gamma):
    return np.exp(-gamma * np.sum((x - y)**2)) 

def rbf_kernel_matrix(X, Y, gamma):
    dists = pairwise_distances(X, Y, metric='sqeuclidean')
    return np.exp(-gamma * dists)

def MMD2_rbf(X, Y, gamma, max_size=5*10**4+1):
    """ Computes the Maximum Mean Discrepancy Squared (MMD^2) between two distributions given samples.
    Args:
        X (np.ndarray): Samples from the first distribution (shape: [n_samples_x, n_features]).
        Y (np.ndarray): Samples from the second distribution (shape: [n_samples_y, n_features]).
        gamma (float): Kernel coefficient for 'rbf'. 
    Returns:
        float: The MMD^2 value.
    """
    m = X.shape[0]
    n = Y.shape[0]
    sum_K_XX = 0.0
    sum_K_YY = 0.0
    sum_K_XY = 0.0
    if m <= 1 or n <= 1:
        raise ValueError("Number of samples in each distribution must be greater than 1.")
    # Fast computation if input arrays are not too large
    if max(m,n) <= max_size :
        K_XX = rbf_kernel_matrix(X, X, gamma)
        sum_K_XX = np.sum(K_XX) - np.sum(np.diag(K_XX))
        del K_XX
        K_YY = rbf_kernel_matrix(Y, Y, gamma)
        sum_K_YY = np.sum(K_YY) - np.sum(np.diag(K_YY))
        del K_YY
        K_XY = rbf_kernel_matrix(X, Y, gamma)
        sum_K_XY = np.sum(K_XY)
        del K_XY
    # Prevent Out Of Memory when the input arrays are too large
    else :
        for i in range(m):
            for j in range(i + 1, m):
                sum_K_XX += 2 * rbf_kernel(X[i], X[j], gamma)
        for i in range(n):
            for j in range(i + 1, n):
                sum_K_YY += 2 * rbf_kernel(Y[i], Y[j], gamma)
        for i in range(m):
            for j in range(n):
                sum_K_XY += rbf_kernel(X[i], Y[j], gamma)
    mmd2 = (sum_K_XX / (m * (m - 1))) + (sum_K_YY / (n * (n - 1))) - (2 * sum_K_XY / (m * n))
    return np.abs(mmd2)


def compute_MMD2(X, Y, gamma=0.5, div=10**3, disable_tqdm=False):
    n_batch = np.size(X[:, 0]) // div
    X_batches = np.split(X, n_batch)
    Y_batches = np.split(Y, n_batch)
    mmd2_values = []
    for i in tqdm(range(n_batch), desc='MMD^2 values',disable=disable_tqdm):
        mmd2_value = MMD2_rbf(X_batches[i].reshape(-1, 1), Y_batches[i].reshape(-1, 1), gamma=gamma)
        mmd2_values.append(mmd2_value)
    return np.mean(mmd2_values)

# def kernel_all_batches(A, B, gamma=1.0, batch_size=10**4, disable_tqdm=False):
#         """Compute the sum of RBF kernel values over all possible pairs."""
#         total_sum = 0.0
#         for i in tqdm(range(0, A.shape[0], batch_size), desc='MMD^2 values',disable=disable_tqdm):
#         #for i in range(0, A.shape[0], batch_size):
#             A_batch = A[i:i + batch_size]
#             for j in range(0, B.shape[0], batch_size):
#                 B_batch = B[j:j + batch_size]
#                 K = rbf_kernel_matrix(A_batch, B_batch, gamma)
#                 total_sum += K.sum().item()
#         return total_sum 

# def compute_MMD_all_batches(X, Y, gamma=1.0, batch_size=10**4):
#     M, N = X.shape[0], Y.shape[0]
#     K_XX = kernel_all_batches(X, X, gamma, batch_size)
#     K_YY = kernel_all_batches(Y, Y, gamma, batch_size)
#     K_XY = kernel_all_batches(X, Y, gamma, batch_size)

#     mmd2 = (K_XX - M)/(M*(M-1)) + (K_YY - N)/(N*(N-1)) - 2 * K_XY / (M*N)
#     return np.sqrt(np.abs(mmd2))



def compute_KSD(exp_samples, target_samples):
    KSD_vals = []
    n, q = np.shape(exp_samples) 
    for j in range(q):
        KSD_vals.append(scipy.stats.ks_2samp(exp_samples[:,j], target_samples[:,j])[0])
    return np.mean(KSD_vals)

def compute_ecdf(samples, x_values):
    sorted_samples = np.sort(samples)
    cdf_values = np.searchsorted(sorted_samples, x_values, side='right') / np.size(sorted_samples)
    return cdf_values


def aux_grad_moment(x,b,t):
        num = b*x**(b-1) + t*x**(t-1)
        deno = (x**b + x**t)**2
        return - num / deno

####################  Optimizers (Numpy)  ####################

class Adam():
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        
    def update(self, t, w, dw):
        ## dw is from current minibatch
        ## momentum beta 1
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw

        ## rms beta 2
        # *** weights *** #
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)

        ## correction
        m_dw_corr = self.m_dw/(1-self.beta1**t)
        v_dw_corr = self.v_dw/(1-self.beta2**t)

        ## update weights and biases
        w = w - self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
        return w


class NewtonMethod():
    def __init__(self, use_lstsq=True):
        self.use_lstsq = use_lstsq  # solves the system by least squares (use when jacobian close to a singular matrix)
        
    def update(self, lmbd, nu, constr, J_constr, grad_Lag, H_Lag):
        """ Apply one step of the Newton method
        Args:
            lmbd (p array): weights to be optimized
            nu (float): Lagrange multiplier
            constr (K array): Contraints function values
            J_constr ((K,p) array): Jacobian of constraints function
            grad_Lag (p array): Gradient of Lagrangian (wrt lmbd)
            H_Lag ((K+p,K+p) array): Hessian of Lagrangian (wrt lmbd)
        Returns:
            p array, float: Updated parameter and Lagrange multiplier
        """
        B, p = np.shape(J_constr)
        Zeros = np.zeros((B,B))
        M = np.block([grad_Lag, constr])
        grad_M = np.block([[H_Lag, (J_constr).T],
                           [J_constr, Zeros]])
        if self.use_lstsq :
            direc = - np.linalg.lstsq(grad_M, M, rcond=1e-15)[0] # equivalent to - np.linalg.pinv(grad_M) @ M 
        else :
            direc = - np.linalg.solve(grad_M, M)
        lmbd = lmbd + direc[:p]
        nu = nu + direc[p:]
        return lmbd, nu

####################  Optimizers (Torch)  ####################

class SGD():
    def __init__(self, eta=0.01, beta1 = 0.9, w_decay=0., momentum=True):
        self.m_dw = 0
        self.eta = eta
        self.beta1 = beta1
        self.w_decay = w_decay
        self.momentum = momentum
        
    def update(self, t, w, dw):
        ## update weights and biases
        dw = dw + self.w_decay * w
        if self.momentum :
            self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
            dw = self.m_dw
        w = w - self.eta * dw
        return w

class torch_Adam():
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, w_decay=0., momentum=True):
        self.m_dw, self.v_dw = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        self.w_decay = w_decay
        self.momentum = momentum
        
    def update(self, t, w, dw):
        ## dw is from current minibatch
        ## momentum beta 1
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw

        ## rms beta 2
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)

        ## correction
        m_dw_corr = self.m_dw/(1-self.beta1**t)
        v_dw_corr = self.v_dw/(1-self.beta2**t)

        ## update weights and biases
        w = w - self.eta*((m_dw_corr/(torch.sqrt(v_dw_corr)+self.epsilon)) + self.w_decay * w)

        if not self.momentum :
            self.m_dw, self.v_dw = 0, 0
        return w

    def scheduler(self, t, decay) :
        self.eta = self.eta * torch.exp(-t*decay)
