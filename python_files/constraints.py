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


####################  Constraints (Numpy)  ####################

class Constraints_OneLayer():    
    def __init__(self, div, betas, b, T_cstr, objective):
        self.div = div   # divergence metric used (one layer, numpy)
        self.va = div.va
        self.model = div.va.model
        self.betas = betas  # order of moment in the constraint
        self.K = np.size(betas)  # number of (equality) constraints on the moments
        self.b = b    # values imposed on the moments
        if np.size(self.b) != self.K :
            print('ValueError : Number of constraints and number of value constraints are not compatible')
        self.T_cstr = T_cstr        # number of MC samples for estimation of constraints functions
        self.objective = objective  # objective function (MI or LB_MI)
        if self.objective == 'MI' :
            self.obj = self.div.MI
            self.grad_obj = self.div.grad_MI
            self.hess_obj = self.div.hess_MI
        if self.objective == 'LB_MI' :
            self.obj = self.div.LB_MI
            self.grad_obj = self.div.grad_LB_MI
            self.hess_obj = self.div.hess_LB_MI

    def fct_constraint(self, w):  
        theta_sample = self.va.implicit_prior_sampler(w, self.T_cstr)
        result = np.zeros(self.K)
        if self.K == 1 :
            result = np.mean(theta_sample**self.betas) - self.b
        else :
            for i in range(self.K):
                A = np.mean(theta_sample**self.betas[i])
                result[i] = A - self.b[i]
        return result
    
    def grad_constraint(self, w):  
        va = self.va
        betas = self.betas
        epsilon_sample = np.random.normal(size=(self.T_cstr,va.p))
        theta_sample = va.g(w, epsilon_sample)
        if self.K == 1 :
            result = np.mean(betas*va.grad_g(w, epsilon_sample)*(theta_sample**(betas-1))[:,None], axis = 0)
        else :
            result = np.zeros((va.p + int(va.use_bias), self.K))
            for i in range(self.K):
                grad_A = np.mean(betas[i]*va.grad_g(w, epsilon_sample)*(theta_sample**(betas[i]-1))[:,None], axis = 0)
                result[:,i] = grad_A
        return result
        
    def hess_constraint(self, eps, w):   # one sample of hessian, hence noisy
        va = self.va
        betas = self.betas
        theta = va.g(w, eps)
        hess_g, prod_grad_g = va.grad2_g(w, eps)
        if self.K == 1 :
            hess_constr = hess_g * betas*theta**(betas-1) + prod_grad_g * betas*(betas-1)*theta**(betas-2)
        else :
            hess_constr = np.zeros((va.p + int(va.use_bias),va.p + int(va.use_bias),self.K))
            for i in range(self.K):
                hess_constr[:,:,i] = hess_g * betas[i]*theta**(betas[i]-1) + prod_grad_g * betas[i]*(betas[i]-1)*theta**(betas[i]-2)  
        return hess_constr
        
    # Lagrangian
    def Lagrangian(self, theta, w, J, N, nu):
        if self.K == 1 : 
            return self.obj(theta, w, J, N) + nu * self.fct_constraint(w)
        else :
            result = self.obj(theta, w, J, N)
            fct_constr = self.fct_constraint(w)
            for i in range(self.K):   
                result += nu[i] * fct_constr[i]
            return result
    
    def grad_Lagrangian(self, eps, w, J, N, nu):
        if self.K == 1 :
            return self.grad_obj(eps, w, J, N) + nu * self.grad_constraint(w)
        else :
            result = self.grad_obj(eps, w, J, N)
            grad_constr = self.grad_constraint(w)
            for i in range(self.K):
                result += nu[i] * grad_constr[:,i]
            return result
        
    def hess_Lagrangian(self, eps, w, J, N, nu):
        if self.K == 1 :
            return self.hess_obj(eps, w, J, N) + nu * self.hess_constraint(eps, w)
        else :
            result = self.hess_obj(eps, w, J, N)
            hess_constr = self.hess_constraint(w)
            for i in range(self.K):
                result += nu[i] * hess_constr[:,:,i]
            return result
        
    # Augmented Lagrangian
    def Augm_Lagrangian(self, theta, w, J, N, nu, nu_tilde):
        lag = self.Lagrangian(theta, w, J, N, nu)
        if self.K == 1 :
            return lag - 0.5 * nu_tilde * self.fct_constraint(w)**2
        else :
            return lag - 0.5 * nu_tilde * np.sum(self.fct_constraint(w)**2)
        
    def grad_Augm_Lagrangian(self, eps, w, J, N, nu, nu_tilde):
        grad_lag = self.grad_Lagrangian(eps, w, J, N, nu)
        if self.K == 1 :
            return grad_lag - nu_tilde * self.grad_constraint(w) * self.fct_constraint(w)
        else :
            result = grad_lag
            fct_constr = self.fct_constraint(w)
            grad_constr = self.grad_constraint(w)
            for i in range(self.K):
                result += - nu_tilde * grad_constr[:,i] * fct_constr[i]
            return result
    
    # Naïve approach with fixed lagrangian multiplier and squared constraint function 
    # "Lagrangian zero"
    def Lag0(self, theta, w, J, N, eta):
        if self.K == 1 :
            return self.obj(theta, w, J, N) - eta * self.fct_constraint(w)**2
        else :
            result = self.obj(theta, w, J, N)
            fct_constr = self.fct_constraint(w)**2
            for i in range(self.K):   
                result += -eta[i] * fct_constr[i]
            return result
    
    def grad_Lag0(self, eps, w, J, N, eta):
        constr = self.fct_constraint(w)
        grad_A = self.grad_constraint(w)
        grad_objective = self.grad_obj(eps, w, J, N)
        if self.K == 1:
            grad_constr = -2* eta * grad_A * constr
            return grad_objective + grad_constr
        else :
            result = grad_objective
            for i in range(self.K):
                grad_constr = -2* eta[i] * grad_A[:,i] * constr[i]
                result += grad_constr
            return result
        


####################  Constraints (Torch)  ####################

class Constraints_NeuralNet():    
    def __init__(self, div, betas, b, T_cstr, objective, lag_method, eta_augm=None, rule=None, moment='raw', taus=None):
        self.div = div   # divergence metric used 
        self.va = div.va
        self.model = div.va.model
        self.moment = moment
        if self.moment == 'raw' :
            sorted, indices = torch.sort(betas)
            self.betas = sorted  # order of moment in the constraint (sorted in increasing order)
            self.b = b[indices]  # values imposed on the moments
        else :
            self.betas = betas
            self.b = b
        self.taus = taus
        self.K = int(betas.size(0))  # number of (equality) constraints on the moments
        self.q = self.va.q      
        if int((self.b).size(0)) != self.K :
            raise ValueError(f'Number of constraints and number of value constraints are not compatible : {self.K} and {int((self.b).size(0))}')
        if self.moment == 'raw':
            if self.K > 1 :
                self.check_implicit_constraints()
        self.T_cstr = T_cstr        # number of MC samples for estimation of constraints functions (non-grad)
        self.objective = objective  # objective function (MI or LB_MI)
        if self.objective == 'MI' :
            self.obj = self.div.MI
            self.grad_obj = self.div.grad_MI
        if self.objective == 'LB_MI' :
            self.obj = self.div.LB_MI
            self.grad_obj = self.div.grad_LB_MI
        self.lag_method = lag_method  # Method used : 'penality' (fixed lag. multiplier and squared constraint) or 'augmented'
        self.eta_augm = eta_augm  # Hyperparameter for Augmented Lagrangian
        if self.eta_augm is not None :
            if int((self.eta_augm).size(0)) != self.K :
                raise ValueError(f'Number of constraints and number of penality parameters are not compatible : {self.K} and {int((self.eta_augm).size(0))}')
        self.rule = rule  # Update rule on Lagrange multipliers and penality parameters : 'SGD' or 'RMS' 

    def check_implicit_constraints(self):  # relevant only if K > 1
        K = self.K
        b = self.b
        beta = self.betas
        for l in range(self.q):
            check_invalid = torch.tensor([[torch.pow(b[j,l], beta[i]) >= torch.pow(b[i,l], beta[j]) for j in range(i+1, K)] for i in range(K-1)])
            if not torch.all(check_invalid):
                raise ValueError('One of the implicit constraints is not satisfied.')
        print("All implicit conditions satisfied.")

    def fct_constraint(self):  
        """ Computes the constraints values by sampling for the actual parameters of the NN
        Returns:
            (K,q) tensor : Constraints values (C_kl) for each constraint on every component of theta
        """
        theta_sample = self.va.implicit_prior_sampler(self.T_cstr)
        result = torch.zeros(self.K, self.q)
        for k in range(self.K):
            for l in range(self.q):
                if self.moment == 'raw':
                    A = torch.mean(theta_sample[:,l]**self.betas[k])
                else :
                    A = torch.mean((theta_sample[:,l]**self.betas[k] + theta_sample[:,l]**self.taus[k])**-1)
                result[k,l] = A - self.b[k,l] 
        return result
        
    def Lagrangian(self, theta, J, N, eta):
        """ Computes the estimated objective function and adds the lagrangian term (with or without augmentation)
        Args:
            theta (tensor): parameter of the chosen stat. model
            J (int): Number of N-samples for MC estimation
            N (int): Number of data samples
            eta ((K,q) tensor): Lagrange multiplier for the non-augmented term
        Returns:
            tensor (size 1): Result of the estimation from the data samples
        """
        result = self.obj(theta, J, N)
        fct_constr = self.fct_constraint()
        if self.lag_method == 'penality' :
            method_term = - eta * fct_constr**2
        if self.lag_method == 'augmented' :
            method_term = eta * fct_constr - 0.5 * self.eta_augm * fct_constr**2
        return result + torch.sum(method_term)
    

    def grad_Lagrangian(self, theta, J, N, eta, grad_tensor):
        betas = self.betas    # grad of constraint term computed with one sample
        taus = self.taus
        va = self.va
        grad_obj = self.grad_obj(theta, J, N, grad_tensor)
        fct_constr = self.fct_constraint()
        constr_term = torch.zeros(self.K, va.q)
        for k in range(self.K):
            for l in range(self.q):
                if self.lag_method == 'penality' :
                    if self.moment == 'raw' :
                        constr_term[k,l] = betas[k]*theta[l]**(betas[k]-1) * (-2)*eta[k,l]*fct_constr[k,l]
                    else : 
                        constr_term[k,l] = aux_grad_moment(theta[l],betas[k],taus[k]) * (-2)*eta[k,l]*fct_constr[k,l]
                if self.lag_method == 'augmented' :
                    if self.moment == 'raw' :
                        constr_term[k,l] = betas[k]*theta[l]**(betas[k]-1) * (eta[k,l]-self.eta_augm[k,l]*fct_constr[k,l])
                    else : 
                        constr_term[k,l] = aux_grad_moment(theta[l],betas[k],taus[k]) * (eta[k,l]-self.eta_augm[k,l]*fct_constr[k,l])
        sum_constr = torch.sum(constr_term, dim=0)
        if va.q == 1 : 
            return grad_obj + sum_constr * grad_tensor
        else:
            grad_tensor = torch.reshape(grad_tensor, (va.nb_param,va.q))
            return grad_obj + torch.sum(sum_constr.unsqueeze(0) * grad_tensor, dim=1)


    def Augm_update_SGD(self, eta, max_violation, update_eta_augm, sup_eta_augm=torch.tensor(10**4)):
        fct_constr = self.fct_constraint()
        for k in range(self.K):
            for l in range(self.q):
                eta[k,l] = eta[k,l] - self.eta_augm[k,l] * fct_constr[k,l]
        for k in range(self.K):
            for l in range(self.q):
                if torch.max(torch.abs(fct_constr)) > max_violation :
                        self.eta_augm[k,l] = torch.minimum(self.eta_augm[k,l] * update_eta_augm, sup_eta_augm)
                else :
                    self.eta_augm[k,l] = torch.minimum(self.eta_augm[k,l] / update_eta_augm, sup_eta_augm)
        return eta

    def Augm_update_RMS(self, eta, var_augm, gamma=0.01, beta=0.99, epsilon=1e-8):
        fct_constr = self.fct_constraint()
        var_new = torch.zeros(self.K, self.q)
        for k in range(self.K):
            for l in range(self.q):
                var_new[k,l] = beta*var_augm[k,l] + (1-beta)*fct_constr[k,l]**2
                self.eta_augm[k,l] = gamma / (torch.sqrt(var_new[k,l]) + epsilon)
                eta[k,l] = eta[k,l] - self.eta_augm[k,l] * fct_constr[k,l]
        return eta, var_new

    def Partial_autograd(self, J, N, eta, num_epochs, optimizer, num_samples_MI, 
                         freq_MI, save_best_param, learning_rate, weight_decay=0., num_samples_grad=1,
                         want_entropy=False, want_norm_param=False, disable_tqdm=False, momentum=True,
                         freq_augm=None, max_violation=None, update_eta_augm=None):
        if int(eta.size(0)) != self.K :
            raise ValueError(f'Size of Lagrange mult ({int(eta.size(0))}) is not compatible with number of constraints ({self.K})')
        var_augm = torch.zeros(self.K)
        MI = []
        range_MI = []
        upper_MI = []
        lower_MI = []
        sum_entropy = []
        norm_param = []
        constr_values = []
        best_model_params = None
        best_MI = -np.inf
        net = self.va.neural_net
        t = 1
        last_valid_params = {k: v.clone() for k, v in net.state_dict().items()}
        all_params = torch.cat([param.view(-1) for param in net.parameters()])
        Collec_grad = torch.zeros((self.va.nb_param, num_samples_grad))

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
                
                # Compute 'loss'
                pseudo_loss = theta_output
                
                # Backpropagation / Compute gradients (only on the NN, not the objective function)
                if net.output_size == 1:
                    pseudo_loss.backward(retain_graph=True)
                    all_grads_tensor = torch.cat([param.grad.view(-1) for param in net.parameters() if param.grad is not None])
                else:
                # For multiple loss outputs, compute gradients for each element of the output tensor
                    all_grads = []
                    for i in range(net.output_size):
                        net.zero_grad()
                        pseudo_loss[i].backward(retain_graph=True)
                        all_grads.extend([param.grad.view(-1) for param in net.parameters() if param.grad is not None])
                    all_grads_tensor = torch.cat(all_grads)
                with torch.no_grad():
                    one_grad = - self.grad_Lagrangian(theta_output, J, N, eta, all_grads_tensor)
                    if self.lag_method == 'augmented' :
                        # Update periodically Lagrange multipliers and penality parameters with the chosen rule
                        if (epoch+1) % freq_augm == 0 :
                            if self.rule == 'SGD' :
                                eta = self.Augm_update_SGD(eta, max_violation, update_eta_augm)
                            elif self.rule == 'RMS' :
                                eta, var_augm = self.Augm_update_RMS(eta, var_augm)
                Collec_grad[:, num_grad] = one_grad
            # Average of gradients values 
            True_grad = torch.mean(Collec_grad, dim=1)
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
                    fct_constr = self.fct_constraint()
                    constr_values += [torch.abs(fct_constr)]
                    Thetas = self.va.implicit_prior_sampler(num_samples_MI)
                    Thetas = delete_nan_elements(Thetas)
                    #print(Thetas)
                    MI_list = [self.div.MI(theta, J, N).item() for theta in Thetas]
                    MI_current = np.mean(MI_list)
                    MI.append(MI_current)
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
        if save_best_param and best_model_params is not None:
            net.load_state_dict(best_model_params)
        #print('Training done!')
        if want_entropy :
            return MI, constr_values, range_MI, lower_MI, upper_MI, sum_entropy
        if want_norm_param :
            return MI, constr_values, range_MI, lower_MI, upper_MI, norm_param
        else :
            return MI, constr_values, range_MI, lower_MI, upper_MI