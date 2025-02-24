# python files
from aux_optimizers import *
from stat_models_numpy import *
from stat_models_torch import *
from variational_approx import *
from neural_nets import *

# packages used
import numpy as np 
import scipy
import torch
from tqdm import tqdm

    
####################  Divergence Metrics (Torch)  ####################


class DivMetric_NeuralNet():
    def __init__(self, va, T, use_alpha, alpha=None, use_log_lik=True, use_baseline=False):
        self.va = va   # variational approximation (with torch and neural networks)
        self.model = va.model  # statistical (torch) model   
        self.use_alpha = use_alpha  # uses -log or alpha-div accordingly
        self.alpha = alpha  # alpha parameter if alpha-div is used
        self.T = T        # number of MC samples for the marginal / to approximate the MLE
        self.use_log_lik = use_log_lik  # computations made with log-likelihood instead
        self.use_baseline = use_baseline

    # The following functions are all noisy, ie one sample only instead of the outer expectation

    def MI(self, theta, J, N):   
        """ Estimates by Monte-Carlo the inner expectancy in the mutual information.
        Args:
            theta (tensor): parameter of the chosen stat. model
            J (int): Number of N-samples for MC estimation
            N (int): Number of data samples
        Returns:
            tensor (size 1): Result of the estimation from the data samples
        """
        model = self.model
        va = self.va
        D = model.sample(theta, N, J)
        d = 0.
        if self.use_log_lik :
            log_lik = model.log_likelihood(theta)
            theta_sample = va.implicit_prior_sampler(self.T)
            if va.q == 1 :
                theta_sample = theta_sample.squeeze()
            logs_lik = model.collec_log_lik(theta_sample)
            ratio_lik = torch.mean(torch.exp(logs_lik - log_lik), dim=0)   
        else :
            marg = va.MC_marg(self.T)
            lik = model.likelihood(theta)
            ratio_lik = marg/lik
        if self.use_alpha : 
            c = 1 / (self.alpha - 1)
            d = c / self.alpha
            sommand = alpha_div(ratio_lik, self.alpha, c, d)
        else:
            sommand = - torch.log(ratio_lik)
        return torch.mean(sommand) - d
 
    def grad_MI(self, theta, J, N, grad_tensor):  
        """ Estimates by Monte-Carlo the gradient of the inner expectancy in the mutual information.
            Important : The grad marginal is computed from one sample only (it simplifies the computation a lot and 
            otherwise it wouldn't be compatible with torch and autograd, but can lead to imprecise results)
        Args:
            theta (tensor): parameter of the chosen stat. model
            J (int): Number of N-samples for MC estimation
            N (int): Number of data samples
            grad_tensor (nb_param*q tensor): Gradients for all trainable parameters, computed beforehand (by autograd)
        Returns:
            (nb_param) tensor: Gradient estimates, to be used with the optimizer for the parameter update 
        """
        mini_eps = torch.tensor(10**-40)
        model = self.model
        va = self.va
        D = model.sample(theta, N, J)
        gradient_log_lik = model.grad_log_lik(theta)
        if self.use_log_lik :
            log_lik = model.log_likelihood(theta)
            theta_sample = va.implicit_prior_sampler(self.T)
            if va.q == 1 :
                theta_sample = theta_sample.squeeze()
            logs_lik = model.collec_log_lik(theta_sample)
            ratio_lik = torch.mean(torch.exp(logs_lik - log_lik), dim=0)
            #print(f'ratio_lik = {ratio_lik}')
        else :
            marg = va.MC_marg(self.T)
            lik = model.likelihood(theta)
            ratio_lik = marg/lik
        if self.use_alpha :
            alpha = self.alpha
            F_func = lambda x : (1-x**alpha)/alpha
            f_prime = lambda x : (x**(alpha-1) - 1)/(alpha-1)
            f_final = lambda x : F_func(x) + f_prime(x)
        else :
            f_final = lambda x : - torch.log(x + mini_eps)  # mini_eps prevents inf values in log
        if va.q == 1 :
            sommand = gradient_log_lik * f_final(ratio_lik) 
            return torch.mean(sommand) * grad_tensor
        else:
            sommand = gradient_log_lik * f_final(ratio_lik).unsqueeze(1) 
            grad_tensor = torch.reshape(grad_tensor, (va.nb_param,va.q))
            return torch.sum(torch.mean(sommand, dim=0) * grad_tensor, dim=1)

    def LB_MI(self, theta, J, N):
        """ Estimates by Monte-Carlo the inner expectancy in the lower bound
        Args:
            theta (tensor): parameter of the chosen stat. model
            J (int): Number of N-samples for MC estimation
            N (int): Number of data samples
        Returns:
            tensor (size 1): Result of the estimation from the data samples
        """   
        model = self.model
        D = model.sample(theta, N, J)
        if self.use_alpha :
            c = 1 / (self.alpha - 1)
            d = 0.      #  c / self.alpha
            f_div = lambda x : alpha_div(x, self.alpha, c, d)
        else :
            f_div = lambda x : -torch.log(x)  
        if self.use_log_lik :
            log_lik = model.log_likelihood(theta)
            if model.available_MLE :
                max_log_lik = model.log_likelihood(model.MLE())
            else :
                max_log_lik = model.log_likelihood(self.va.MLE_approx(self.T))
            ratio_lik = torch.exp(max_log_lik - log_lik)
        else:
            lik = model.likelihood(theta)
            if model.available_MLE :
                max_lik = model.likelihood(model.MLE())
            else :
                max_lik = model.likelihood(self.va.MLE_approx(self.T))
            ratio_lik = max_lik/lik 
        sommand = f_div(ratio_lik)
        return torch.mean(sommand)

    def grad_LB_MI(self, theta, J, N, grad_tensor):
        """ Estimates by Monte-Carlo the gradient of the inner expectancy in the lower bound
        Args:
            theta (tensor): parameter of the chosen stat. model
            J (int): Number of N-samples for MC estimation
            N (int): Number of data samples
            grad_tensor (nb_param*q tensor): Gradients for all trainable parameters, computed beforehand (by autograd)
        Returns:
            (nb_param) tensor: Gradient estimates, to be used with the optimizer for the parameter update 
        """
        mini_eps = torch.tensor(10**-40)
        model = self.model
        va = self.va
        alpha = self.alpha
        D = model.sample(theta, N, J)
        if self.use_alpha :
            F_func = lambda x : (1-x**alpha) / alpha
        else :
            F_func = lambda x : -torch.log(x + mini_eps)  # mini_eps prevents inf values in log
        gradient_log_lik = model.grad_log_lik(theta)
        if self.use_log_lik :
            log_lik = model.log_likelihood(theta)
            #print(f'log_lik = {log_lik}')
            if model.available_MLE :
                max_log_lik = model.log_likelihood(model.MLE())
                #print(f'MLE = {model.MLE()}')
            else :
                max_log_lik = model.log_likelihood(va.MLE_approx(self.T))
            #print(f'max_log_lik = {max_log_lik}')
            ratio_lik = torch.exp(max_log_lik - log_lik)
            #print(f'ratio_lik = {ratio_lik}')
        else :
            lik = model.likelihood(theta)
            if model.available_MLE :
                max_lik = model.likelihood(model.MLE())
            else :
                max_lik = model.likelihood(va.MLE_approx(self.T))
            ratio_lik = max_lik / lik
        if va.q == 1 :
            if self.use_baseline : 
                baseline = torch.mean(gradient_log_lik**2 * F_func(ratio_lik)) / torch.mean(gradient_log_lik**2)
                sommand = gradient_log_lik * (F_func(ratio_lik) - baseline)
            else :
                sommand = gradient_log_lik * F_func(ratio_lik)
            #print(f'Var sommand : {torch.var(sommand)}')
            return torch.mean(sommand) * grad_tensor
        else:
            if self.use_baseline :
                baseline = torch.mean(gradient_log_lik**2 * F_func(ratio_lik).unsqueeze(1),dim=0) / torch.mean(gradient_log_lik**2,dim=0)
                sommand = gradient_log_lik * (F_func(ratio_lik).unsqueeze(1) - baseline.unsqueeze(0))
            else :
                sommand = gradient_log_lik * F_func(ratio_lik).unsqueeze(1)
            #print(f'Var sommand : {torch.var(sommand, dim=0)}')
            grad_tensor = torch.reshape(grad_tensor, (va.nb_param, va.q))
            return torch.sum(torch.mean(sommand,dim=0) * grad_tensor, dim=1)
    
    
    ##### Training loop functions #####

    def Full_autograd(self, J, N, num_epochs, loss_fct, optimizer, num_samples_MI, freq_MI, save_best_param):
        MI = []
        range_MI = []
        best_model_params = None
        best_MI = -np.inf
        net = self.va.neural_net
        # Training loop
        for epoch in tqdm(range(num_epochs), desc='Epochs'):
            # Forward pass
            eps_0 = torch.randn(net.input_size)
            theta_output = net(eps_0)
            # Compute loss
            if loss_fct == 'MI' :
                loss = - self.MI(theta_output, J, N)   
            if loss_fct == 'LB_MI' :
                loss = - self.LB_MI(theta_output, J, N) 
            # Backward pass and optimization
            optimizer.zero_grad()  # Zero gradients
            # Backpropagation
            loss.backward()  
            optimizer.step()  # Update weights
            if (epoch+1) % freq_MI == 0:
                    with torch.no_grad():
                        range_MI += [epoch]
                        Thetas = self.va.implicit_prior_sampler(num_samples_MI)
                        MI_current = np.mean([self.MI(theta, J, N).numpy() for theta in Thetas])
                        MI += [MI_current]
                        if save_best_param :
                            if MI_current > best_MI and MI_current != float('inf') :
                                best_MI = MI_current
                                best_model_params = {k: v.clone() for k, v in net.state_dict().items()}
        if save_best_param : 
            net.load_state_dict(best_model_params)
        return MI, range_MI#, loss_list
    

    def Partial_autograd(self, J, N, num_epochs, loss_fct, optimizer, num_samples_MI, freq_MI, save_best_param,
                        learning_rate, weight_decay=0.,num_samples_grad=1, want_entropy=False, 
                        want_norm_param=False, disable_tqdm=False, momentum=True, keep_all_MI_vals=False):
        MI = []
        all_MI_vals = []
        range_MI = []
        upper_MI = []
        lower_MI = []
        sum_entropy = []
        norm_param = []
        best_model_params = None
        best_MI = -np.inf
        best_epoch = 0
        net = self.va.neural_net
        t = 1
        last_valid_params = {k: v.clone() for k, v in net.state_dict().items()}
        all_params = torch.cat([param.view(-1) for param in net.parameters()])
        Collec_grad = torch.zeros((self.va.nb_param, num_samples_grad))
        all_MI_vals = []

        optimizers = {}
        for param in net.parameters():
            # Create a new optimizer for each parameter
            opti = optimizer(eta=learning_rate, w_decay=weight_decay, momentum=momentum)
            # Store the optimizer in a dictionary with the parameter as the key
            optimizers[param] = opti

        # Training loop
        for epoch in tqdm(range(num_epochs), desc='Epochs', disable=disable_tqdm):
            # Zero gradients manually
            for param in net.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            for num_grad in range(num_samples_grad):
                # Forward pass
                eps_0 = torch.randn(net.input_size)
                theta_output = net(eps_0)
                #print(f'theta_output = {theta_output}')

                # Compute 'loss'
                pseudo_loss = theta_output
                
                # Backpropagation / Compute gradients 
                # Autograd on the 'loss'
                if net.output_size == 1:
                    pseudo_loss.backward(retain_graph=True)
                    all_grads_tensor = torch.cat([param.grad.view(-1) for param in net.parameters() if param.grad is not None])
                else:
                # For multiple loss outputs, compute gradients for each element of the output tensor
                    all_grads = []
                    for i in range(net.output_size):
                        net.zero_grad()     # Avoid gradient accumulation between dimensions
                        pseudo_loss[i].backward(retain_graph=True)
                        all_grads.extend([param.grad.view(-1) for param in net.parameters() if param.grad is not None])
                    all_grads_tensor = torch.cat(all_grads)
                #print(f'auto grads = {all_grads_tensor}')
                # Gradient of the objective function
                with torch.no_grad():
                    if loss_fct == 'MI':
                        one_grad = -self.grad_MI(theta_output, J, N, all_grads_tensor)
                    elif loss_fct == 'LB_MI':
                        one_grad = -self.grad_LB_MI(theta_output, J, N, all_grads_tensor)
                Collec_grad[:, num_grad] = one_grad
            # Average of gradients values 
            True_grad = torch.mean(Collec_grad, dim=1)
            with torch.no_grad() :
                # param_norm_grad = torch.Tensor([param for param in net.parameters()])
                True_grad += 0.01*all_params
            #print(f'True grad = {True_grad}')
            # Update parameters using true gradients
            index = 0
            with torch.no_grad():
                for param in net.parameters(): 
                    opti = optimizers[param]
                    if param.grad is not None:
                        param_length = param.numel()
                        # Extract the corresponding part of the true gradient for this parameter
                        param_true_grad = True_grad[index:index + param_length].view(param.size())
                        # Update the parameter using the chosen optimizer
                        param.data = opti.update(t, w=param, dw=param_true_grad)
                        index += param_length
            t += 1
            # Check for NaN values in parameters
            if has_nan_params(net):
                net.load_state_dict(last_valid_params)  # Load the last valid parameters
                raise ValueError(f"NaN detected in parameters, stopping training at epoch {epoch}")
            # Save the last valid parameters
            last_valid_params = {k: v.clone() for k, v in net.state_dict().items()}
            
            # Evaluate the MI periodically and save the best model parameters (if wanted)
            if (epoch + 1) % freq_MI == 0:
                new_params = torch.cat([param.view(-1) for param in net.parameters()])
                with torch.no_grad():
                    range_MI.append(epoch)
                    Thetas = self.va.implicit_prior_sampler(num_samples_MI)
                    Thetas = delete_nan_elements(Thetas)
                    #print(Thetas)
                    MI_list = np.array([self.MI(theta, J, N).item() for theta in Thetas])
                    MI_list = remove_nan_and_inf(MI_list)
                    MI_current = np.mean(MI_list)
                    MI.append(MI_current)
                    if keep_all_MI_vals : 
                        all_MI_vals.append(MI_list)
                    upper_MI.append(np.quantile(MI_list, 0.975))
                    lower_MI.append(np.quantile(MI_list, 0.025))
                    if want_entropy :
                        sum_entropy.append(np.sum(scipy.stats.differential_entropy(Thetas)))

                    if want_norm_param :
                        norm_param.append(torch.sqrt(torch.sum((new_params - all_params)**2)).item())
                        all_params = new_params
                    
                    if save_best_param:
                        if MI_current > best_MI and MI_current != float('inf'):
                            best_MI = MI_current
                            best_model_params = {k: v.clone() for k, v in net.state_dict().items()}
                            best_epoch = epoch
        if save_best_param and best_model_params is not None:
            net.load_state_dict(best_model_params)
        #print('Training done!')
        if save_best_param :
            print(f'Best parameters obtained at epoch {best_epoch}')
        if want_entropy :
            return MI, range_MI, lower_MI, upper_MI, sum_entropy
        if want_norm_param :
            return MI, range_MI, lower_MI, upper_MI, norm_param
        if keep_all_MI_vals : 
            return MI, range_MI, lower_MI, upper_MI, all_MI_vals
        else :
            return MI, range_MI, lower_MI, upper_MI
    
