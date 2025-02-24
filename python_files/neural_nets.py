# python files
from aux_optimizers import *
from stat_models_numpy import *
from stat_models_torch import *

# packages used
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


##########  Neural Networks  ##########


############################## Custom activationb functions ##############################


class Scaled_Softplus(nn.Module):
    def __init__(self, init_scal=1.0):
        super(Scaled_Softplus, self).__init__()
        self.scal = nn.Parameter(torch.tensor(init_scal))  

    def forward(self, x):
        scalar = F.softplus(self.scal)
        return (1 / scalar) * torch.log(1 + torch.exp(x))



############################## Simple architectures ##############################

class NetProbitSeparate(nn.Module):
    def __init__(self, input_size, m1, s1, b1, pre_act, act1):
        super().__init__()
        self.input_size = input_size
        self.output_size = 2
        self.mean1 = m1
        self.std1 = s1
        self.bias1 = b1
        self.netalpaha = NetforProbitAlpha(input_size, m1, s1, b1, act1[0])
        self.netbeta = NetforProbitBeta(input_size, m1, s1, b1, act1[1])
        self.pre_act = pre_act

    def forward(self, x) :
        x0 = self.pre_act[0](x)
        x1 = self.pre_act[1](x)
        x0 = self.netalpaha(x0)
        x1 = self.netbeta(x1)
        # print(x1.shape)
        # assert x1.shape == (x1.shape[0],1)
        return torch.cat([x0,x1], -1)

class PreActivation(nn.Module):
    def __init__(self, act_functions):
        super(PreActivation, self).__init__()
        self.act_functions = act_functions
        
    def forward(self, x) :
        y = []
        for f in self.act_functions :
            y.append(f(x))
        return torch.concatenate(y)
    
class NetforProbitAlpha(nn.Module):
    def __init__(self, input_size, m1, s1, b1, act1):
        super().__init__()
        self.singl = SingleLinear(input_size, 1, m1, s1, b1, act1)

    def forward(self, x) :
        return self.singl(x)
    
class NetforProbitBeta(nn.Module):
    def __init__(self, input_size, m1, s1, b1, act1):
        super().__init__()
        self.singl = SingleLinear(input_size, 1, m1, s1, b1, act1)

    def forward(self, x) :
        return self.singl(x)

class SingleLinear(nn.Module):
    def __init__(self, input_size, output_size, m1, s1, b1, act1):
        super(SingleLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mean1 = m1
        self.std1 = s1
        self.bias1 = b1
        self.fc1 = nn.Linear(input_size, output_size)
        #self.fc1 = nn.Linear(input_size, output_size, bias=False)
        self.act1 = act1   # Sigmoid(x) = (1 + tanh(x/2)) / 2
        # Initialize weights and biases
        init.normal_(self.fc1.weight, mean=self.mean1, std=self.std1)  # Initialize fc1 weights from a normal distribution
        init.constant_(self.fc1.bias, self.bias1)  # Initialize fc1 biases 
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        return x

    
class DoubleLinear(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, m1, s1, b1, m2, s2, b2, act1, act2, drop_out=False):
        super(DoubleLinear, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mean1 = m1
        self.std1 = s1
        self.bias1 = b1
        self.mean2 = m2
        self.std2 = s2
        self.bias2 = b2
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.drop_out = drop_out
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act1 = act1 
        self.act2 = act2  
        # Initialize weights and biases
        init.normal_(self.fc1.weight, mean=self.mean1, std=self.std1)  # Initialize fc1 weights from a normal distribution
        #init.constant_(self.fc1.weight, self.mean1) # Initialize fc1 weights to constant
        init.constant_(self.fc1.bias, self.bias1)  # Initialize fc1 biases to constant
        init.normal_(self.fc2.weight, mean=self.mean2, std=self.std2)  # Initialize fc2 weights from a normal distribution
        #init.constant_(self.fc2.weight, self.mean2) # Initialize fc2 weights to constant
        init.constant_(self.fc2.bias, self.bias2)  # Initialize fc2 biases to constant
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        if self.drop_out :
            x = self.dropout(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x
    
    
############################## Larger nets obtained by combining smaller nets ##############################

class MultiLayer(nn.Module):
    def __init__(self, Layers):
        super(MultiLayer, self).__init__()
        self.layers = nn.ModuleList(Layers)
        self.check_sizes()
        self.input_size = self.layers[0].input_size
        self.output_size = self.layers[-1].output_size

    def check_sizes(self):
        # Check compatibility between successive layers
        for i in range(1, len(self.layers)):
            prev_layer = self.layers[i-1]
            current_layer = self.layers[i]
            assert prev_layer.output_size == current_layer.input_size, \
                f"Incompatible layer sizes: Layer {i} input size {current_layer.input_size} " \
                f"does not match Layer {i - 1} output size {prev_layer.output_size}"

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return z
    

class AverageNet(nn.Module):
    def __init__(self, net1, net2):
        super(AverageNet, self).__init__()
        self.net1 = net1
        self.net2 = net2
        # Check if input/output sizes are compatible
        self.input_size = self.net1.input_size
        if self.input_size != self.net2.input_size :
            raise ValueError(f'The networks have different input sizes : {net1.input_size} and {net2.input_size}')
        self.output_size = self.net1.output_size
        if self.output_size != self.net2.output_size :
            raise ValueError(f'The networks have different output sizes : {net1.output_size} and {net2.output_size}')
            
        # Average parameter (NOT restricted to [0,1] !)
        self.avg_param_no_clip = nn.Parameter(torch.tensor(0.5))
    
    def forward(self,x):
        # Forward pass through each network
        avg_param = torch.clamp(self.avg_param_no_clip, 0, 1)   # clip the parameter to [0,1]
        out1 = self.net1(x)
        out2 = self.net2(x)
        avg_output = avg_param * out1 + (1-avg_param) * out2
        return avg_output  # Returns average of the two outputs
    
    
class MultiAverageNet(nn.Module):
    def __init__(self, nets):
        super(MultiAverageNet, self).__init__()
        self.nets = nn.ModuleList(nets)
        self.nb_nets = len(nets)
        # Check if input/output sizes are compatible
        self.input_size = self.nets[0].input_size
        self.output_size = self.nets[0].output_size
        self.verify_dimensions()
        # Average parameters (NOT restricted to [0,1] !)
        self.coeffs = nn.Parameter(torch.ones(self.nb_nets))

    def verify_dimensions(self):
        if not self.nets:
            raise ValueError("The list of subnets is empty")
        for i, subnet in enumerate(self.nets):
            if subnet.input_size != self.input_size:
                raise ValueError(f"Subnet {i} has inconsistent input size: expected {self.input_size}, got {subnet.input_size}")
            if subnet.output_size != self.output_size:
                raise ValueError(f"Subnet {i} has inconsistent output size: expected {self.output_size}, got {subnet.output_size}")
        
    def forward(self,x):
        # Forward pass through each network and average the outputs
        avg_param = F.softmax(self.coeffs, dim=0)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        outputs = [subnet(x) for subnet in self.nets]
        outputs = torch.stack(outputs, dim=0)  
        avg_output = torch.sum(avg_param[:, None, None] * outputs, dim=0)
        return avg_output.squeeze(0)
        
        
class MixtureNet(nn.Module):
    def __init__(self, nets, temp=1.0):
        super(MixtureNet, self).__init__()
        self.nets = nn.ModuleList(nets)
        self.nb_nets = len(nets)
        # Check if input/output sizes are compatible
        self.input_size = self.nets[0].input_size
        self.output_size = self.nets[0].output_size
        self.verify_dimensions()
        # Mixture probability parameters (NOT restricted to [0,1] !)
        self.coeffs = nn.Parameter(torch.ones(self.nb_nets))
        self.temp = temp   # Temperature parameter for Gumbel-Softmax (Gumbel-Softmax closer to Categorical when temp closer to zero)

    def verify_dimensions(self):
        if not self.nets:
            raise ValueError("The list of subnets is empty")
        for i, subnet in enumerate(self.nets):
            if subnet.input_size != self.input_size:
                raise ValueError(f"Subnet {i} has inconsistent input size: expected {self.input_size}, got {subnet.input_size}")
            if subnet.output_size != self.output_size:
                raise ValueError(f"Subnet {i} has inconsistent output size: expected {self.output_size}, got {subnet.output_size}")
        
    def forward(self, x) :
        mix_param = F.softmax(self.coeffs, dim=0)
        gumbel = -torch.log(-torch.log(torch.rand_like(mix_param)))  # Gumbel(0,1) variables
        y = F.softmax((gumbel + torch.log(mix_param))/ self.temp, dim=-1)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        outputs = [subnet(x) for subnet in self.nets]    
        outputs = torch.stack(outputs, dim=0)  
        avg_output = torch.sum(y[:, None, None]*outputs, dim=0)
        return avg_output.squeeze(0)
    

class SeparateNets(nn.Module):
    def __init__(self, net1, net2):
        super(SeparateNets, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.input_size = self.net1.input_size
        if self.input_size != self.net2.input_size :
            raise ValueError(f'The networks have different input sizes : {net1.input_size} and {net2.input_size}')
        self.output_size = self.net1.output_size + self.net2.output_size
        
    def forward(self,x):
        out1 = self.net1(x)
        out2 = self.net2(x)
        return torch.cat((out1, out2), dim=-1)
    
class MultiSeparateNets(nn.Module):
    def __init__(self, nets, renormalize=False):
        super(MultiSeparateNets, self).__init__()
        self.nets = nn.ModuleList(nets)
        self.renormalize = renormalize
        # Verify that all networks have the same input size
        self.input_size = self.nets[0].input_size
        for net in self.nets:
            if net.input_size != self.input_size:
                raise ValueError(f'The networks have different input sizes: {self.input_size} and {net.input_size}')
        
        # Calculate the total output size
        self.output_size = sum(net.output_size for net in self.nets)
    
    def forward(self, x):
        outputs = [net(x) for net in self.nets]
        concat_output = torch.cat(outputs, dim=-1)
        
        if self.renormalize:
            # Renormalize each output by dividing by the sum of all outputs to get probabilities
            sum_outputs = concat_output.sum(dim=-1, keepdim=True)
            concat_output = concat_output / (sum_outputs + 1e-10)  # Adding a small value to avoid division by zero
        return concat_output
    
class AffineTransformation(nn.Module):
    def __init__(self, net, m_low, M_upp, indices=None):
        super(AffineTransformation, self).__init__()
        self.net = net
        self.input_size = self.net.input_size
        self.output_size = self.net.output_size
        self.m_low = m_low  # Lower bound of the affine transformation
        self.M_upp = M_upp  # Upper bound of the affine transformation
        self.indices = indices

    def forward(self, x):
        x = self.net(x)
        one_dim = (x.dim() == 1)
        if one_dim :
            x = x.unsqueeze(0)
        x_out = torch.clone(x)
        if self.indices is None :
            x_out = self.m_low + (self.M_upp - self.m_low) * x
        else :
            for idx in self.indices:
                x_out[:, idx] = self.m_low[idx] + (self.M_upp[idx] - self.m_low[idx]) * x[:, idx]
        if one_dim :
            x_out = x_out.view(-1)
        return x_out
    

class DifferentActivations(nn.Module):
    def __init__(self, net, act_functions):
        super(DifferentActivations, self).__init__()
        self.net = net
        self.input_size = self.net.input_size
        self.output_size = self.net.output_size
        self.act_functions = act_functions

    def forward(self, x):
        x = self.net(x)
        one_dim = (x.dim() == 1)
        if one_dim :
            x = x.unsqueeze(0)
        x_out = torch.clone(x) 
        for i, act in enumerate(self.act_functions):
            x_out[:, i] = act(x[:, i])
        if one_dim :
            x_out = x_out.view(-1)
        return x_out


##############################  Neural Nets inspired from Normalizing Flows  ##############################

class Planar_Flow(nn.Module):
    def __init__(self, input_size, act):        # typically  : act = torch.tanh
        super(Planar_Flow, self).__init__()
        self.input_size = input_size
        self.output_size = input_size
        self.act = act
        self.u = nn.Parameter(torch.randn(input_size))
        self.w = nn.Parameter(torch.randn(input_size))
        self.b = nn.Parameter(torch.randn(1))
    
    def forward(self, z):
        activation = self.act(z @ self.w + self.b)
        u_hat = self.u + (torch.log(1 + torch.exp(self.u @ self.w)) - 1 - self.u @ self.w) * self.w / torch.norm(self.w)
        z_next = z + u_hat * activation.unsqueeze(1)
        return z_next


class SingleLogitSigmoid(nn.Module):
    def __init__(self, input_size, hidden_units, output_size, act):       
        super(SingleLogitSigmoid, self).__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.output_size = output_size
        self.act = act
        self.fc = nn.Linear(input_size, hidden_units * output_size)
        self.w = nn.Parameter(torch.randn((hidden_units,output_size)))
    
    def forward(self, z, m=10**-6, M=1-10**-6):  # m and M to avoid NaN values
        one_dim = (z.dim() == 1)
        if one_dim :
            z = z.unsqueeze(0)
        batch_size = z.size(0)
        z = self.fc(z)
        z = torch.sigmoid(z)
        z = z.view(batch_size, self.hidden_units, self.output_size)
        w_soft = F.softmax(self.w, dim=0)
        z = torch.sum(w_soft.unsqueeze(0) * z, dim=1)
        z = m + (M-m)*z  # bounds before logit
        z = torch.logit(z)
        if one_dim :
            z = z.view(-1)
        return self.act(z)
    

class NormalizingFlows(nn.Module):
    def __init__(self, input_size, output_size, Flows, act_final):
        super(NormalizingFlows, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.flows = nn.ModuleList(Flows)
        self.num_flows = len(Flows)
        self.act_final = act_final
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, z):
        for flow in self.flows:
            z = flow(z)
        z = self.fc(z)
        z = self.act_final(z)
        return z


##############################  Neural Net for VA on the Posterior  ##############################

class VA_post_net(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):
        super(VA_post_net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        # Initial layer
        self.fc_init = nn.Linear(input_size, hidden_size)
        # Layer for the mean
        self.fc_mean = nn.Linear(hidden_size, latent_dim)
        # Layer for the log-variance
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)
        self.mean = torch.zeros(latent_dim)
        self.logvar = torch.ones(latent_dim)

    def sample_q(self, num_samples=100):
        var = torch.exp(self.logvar)
        cov_matrix = var * torch.zeros(self.latent_dim, self.latent_dim).fill_diagonal_(1.0)
        q_eps = torch.distributions.MultivariateNormal(self.mean, cov_matrix).sample((num_samples,))
        return q_eps
    
    def forward(self, x):
        x = torch.relu(self.fc_init(x))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        self.mean = mean
        self.logvar = logvar
        return mean, logvar









