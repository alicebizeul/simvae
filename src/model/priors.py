"""
Module Name: priors.py
Author: Alice Bizeul
Ownership: ETH Zürich - ETH AI Center
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
log_2_pi = torch.log(2 * torch.tensor(math.pi))

class Prior_StdGauss(nn.Module):
    def __init__(self, z_dim, *args, **kwargs) -> None:

        super(Prior_StdGauss, self).__init__()
        self.z_dim = z_dim
        self.params = nn.ParameterDict()
        self.params["c_means"] = nn.Parameter(torch.zeros(z_dim, 1), requires_grad=False )
        self.params["c_vars"]  = nn.Parameter(torch.ones(z_dim, 1),  requires_grad=False )
        self.params["c_prior"] = nn.Parameter(torch.ones([1]), requires_grad=False )

    def forward(self, z):
        """  mote carlo estimate of  ∫ q(z|x) log p(z)     where:  p(z) = N(0,I)   and   z ~ q(z|x)             """
        self.prior = -0.5 * torch.sum( log_2_pi +  z.pow(2))
        return self.prior


class Prior_Sup(nn.Module):
    def __init__( self, num_sim=1, var_z_y=1., num_data=50000, p_y_prior='gaussian', *args, **kwargs) -> None:
        super(Prior_Sup, self).__init__()
        self.num_sim   = num_sim
        self.var_z_y   = var_z_y
        self.num_data  = num_data
        self.p_y_prior = p_y_prior

        assert (num_sim > 1), "Self-sup not going to work without augmentations"

    def log_p_zy(self, z):
        """z --> p(z|y)   [ can change to log p(z|y) for stability ]"""
        if self.num_sim > 1:
            z = z.view(-1, self.num_sim, *z.shape[1:])
            c_means = torch.mean(z, 1).unsqueeze(1)  
            log_p_z_y = -0.5 * torch.sum( torch.pow(z - c_means, 2) / self.var_z_y, dim=1)
            
            if self.p_y_prior == 'uniform':
                log_p_y = math.log(self.num_data)        
            elif self.p_y_prior == 'gaussian':
                log_p_y = - 0.5 * torch.sum(torch.pow(c_means, 2),1)
            log_p_zy  = log_p_z_y + log_p_y

            return log_p_zy

    def forward(self, z):
        return torch.sum(self.log_p_zy(z), dim=(0,1))
