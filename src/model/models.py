"""
Module Name: models.py
Author: Alice Bizeul
Ownership: ETH Zürich - ETH AI Center
"""
import os, math
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, LogNormal, MultivariateNormal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ..utils.utils import set_seed, cluster_acc
from .decoders import ResNetDec18
from .resnet import resnet18_model
from .priors import Prior_Sup, Prior_StdGauss

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class LogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes, binary=False): 
        super(LogisticRegression, self).__init__() 
        self.model = nn.Linear(n_features, n_classes if not binary else 1)

        if binary: self.head=nn.Sigmoid()
        else:self.head=nn.Identity()

    def forward(self, x):
        return self.head(self.model(x))

class MLP(nn.Module):
    def __init__(self, n_features, n_classes, binary=False): 
        super(MLP, self).__init__() 

        self.model = nn.ModuleList([nn.Linear(n_features, n_features),nn.ReLU(),nn.Linear(n_features,n_classes if not binary else 1)])

        if binary: self.head=nn.Sigmoid()
        else: self.head=nn.Identity()

    def forward(self, x):
        for i, layer in enumerate(self.model):
            x = layer(x)
        return self.head(x)

class KNN(nn.Module):
    def __init__(self, n_classes, n_neighbors_max=15):
        super(KNN, self).__init__()
        self.classes=n_classes
        self.n_neighbors_max = n_neighbors_max
        self.training_samples = []
        self.training_labels = []
        self.models=[]

    def e_step(self, batch, labels):
        self.training_samples.append(batch.detach().cpu())
        self.training_labels.append(labels.detach().cpu())

    def m_step(self,test_samples, test_labels):
        if self.training_samples != []:
            self.models=[]
            self.training_samples=torch.cat(self.training_samples,dim=0).numpy()
            self.training_labels=torch.cat(self.training_labels,dim=0).numpy()

            for n in range(1,self.n_neighbors_max+1):
                self.models.append(KNeighborsClassifier(n))
                self.models[-1].fit(self.training_samples,self.training_labels)
            self.training_samples = []
            self.training_labels = []

        if self.models != []:
            accuracies = [sum(m.predict(test_samples)==test_labels) for m in self.models]
            return accuracies
        else: return np.nan*np.ones(self.n_neighbors_max)

class GM(nn.Module):
    def __init__(self, n_classes ):
        super(GM, self).__init__()
        self.classes =n_classes
        self.model = GaussianMixture(n_components=n_classes, random_state=10, n_init=20)
        self.training_samples = []

    def e_step(self, batch, labels):
        self.training_samples.append(batch.detach().cpu())

    def m_step(self,test_samples, test_labels):
        if self.training_samples != []:
            self.training_samples=torch.cat(self.training_samples,dim=0).numpy()
            self.model.fit(self.training_samples)
            self.training_samples = []   

        test_labels_pred=self.model.predict(test_samples)
        return cluster_acc(test_labels_pred,test_labels), normalized_mutual_info_score(test_labels,test_labels_pred), adjusted_rand_score(test_labels,test_labels_pred)

def eval_classifier(dim, num_classes, lr=None, binary=False):
    classifier = LogisticRegression(dim, num_classes, binary).to(DEVICE)
    if lr is None:
        return classifier 
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    return classifier, optimizer

def eval_mlp(dim, num_classes, lr=None, binary=False):
    classifier = MLP(dim, num_classes, binary).to(DEVICE)
    if lr is None:
        return classifier
    optimizer = optim.Adam(classifier.parameters(),lr=lr)
    return classifier, optimizer

def reparameterize(mu, log_var, training):
    """param mu:      E[Z|X]       param log_var: log Var[Z|X]"""
    if log_var is not None:
        std = torch.exp(0.5 * log_var)  # standard deviation
        if training:eps = torch.randn_like(std)  # `randn_like` as we need the same size
        else: eps=None
        return mu + (eps * std) if training else mu # sample as if from the input space
    else: return mu


# ==================================================================================================================
# ENCODER / DECODER
# ==================================================================================================================

class Encoder(nn.Module):
    def __init__(self, x_dim=28*28, c_dim=5, m_dims=3, resnet18=False, n_channels=1, bn=True, split_output=True, seed=1234, *args, **kwargs):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.c_dim = c_dim
        self.z_dim = c_dim
        self.resnet18 = resnet18
        self.output_multiplier = 2 if split_output else 1        
        set_seed(seed)

        if resnet18:
            self.layers = resnet18_model(weights=None, num_classes=self.z_dim*self.output_multiplier, bn=bn)

            if np.sqrt(self.x_dim/n_channels) <= 32:
                self.layers.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1)                   
            else:
                self.layers.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(4, 4), stride=1,  bias=False)

            self.layers = torch.nn.Sequential(*(list(self.layers.children())[:-2]))
            final_kernel = 4*4 if np.sqrt(self.x_dim/n_channels) ==64 else 2*2
            self.head_c=nn.Sequential(nn.Flatten(),nn.Linear(512*final_kernel,self.z_dim*self.output_multiplier))

        else: 
            layers = []
            in_dim = x_dim
            for i, out_dim in enumerate(m_dims[::-1] + [self.z_dim * self.output_multiplier]):
                layers.append(nn.Linear(in_dim, out_dim))
                if i != (len(m_dims + [self.z_dim * self.output_multiplier])-1):
                    layers.append(nn.ReLU())
                in_dim = out_dim
            self.layers = nn.ModuleList(layers)

    def forward(self, x):
        if self.resnet18:
            x = self.layers(x)
            x = self.head_c(x)
        else:
            for i, layer in enumerate(self.layers):
                x = layer(x)

        if self.output_multiplier == 2:
            shared=x.view(-1, 2, self.z_dim)
            mu_c, log_var_c = shared[:,0, :self.c_dim], shared[:,1,:self.c_dim] 
        
        else: mu_c, log_var_c = x, None
        return (mu_c, log_var_c)

class Decoder(nn.Module):
    def __init__(self, x_dim=784, c_dim=5, m_dims=3, resnet18=False, bn=True, n_channels=1, seed=1234, *args, **kwargs):
        super(Decoder, self).__init__()

        self.x_dim = x_dim
        self.z_dim = c_dim 
        self.resnet18 = resnet18
        set_seed(seed)

        if resnet18:
            self.layers = ResNetDec18(layers=[2, 2, 2, 2, 2], z_dim=self.z_dim, nc=n_channels, size=int(np.sqrt(x_dim/n_channels)))
        else: 
            layers = []
            in_dim = self.z_dim
            for i,out_dim in enumerate(m_dims + [x_dim]):
                layers.append(nn.Linear(in_dim, out_dim))
                if i != (len(m_dims + [x_dim])-1):
                    layers.append(nn.ReLU())
                in_dim = out_dim
            self.layers = nn.ModuleList(layers)

    def forward(self, z): 
        if self.resnet18: 
            mu_x = self.layers(z)
        else:
            for i, layer in enumerate(self.layers):
                z = layer(z)
            mu_x=z
        return mu_x
    
    def forward_while_training(self, z):
        training = self.training
        if training: self.eval()
        mu_x = self.forward(z)
        if training: self.train()
        return mu_x

# ==================================================================================================================
#  VAE MODELS
# ==================================================================================================================

class BaseVAE(nn.Module):
    def __init__( self, net_args, log_var_x=0.0, binarise=False, seed=1234, binary=False, *args, **kwargs ):
        super(BaseVAE, self).__init__()

        self.z_dim = net_args['c_dim']
        self.binarise  = binarise  
        self.log_var_x = log_var_x
        self.resnet18    = net_args.get('resnet18', False)

        self.encoder = Encoder(seed=seed,**net_args)  
        self.decoder = Decoder(seed=seed,**net_args)  

        self.evaluator = {}
        self.prior_init()
        self.eval_init(binary=binary)

    def prior_init(self, postburn=False):
        self.prior_module = nn.ModuleDict({'c': Prior_StdGauss(self.z_dim)})

    def eval_init( self, num_classes=10, final_eval=False, lr=3.e-4, binary=False):
        self.evaluator["logreg"], opt1 = eval_classifier(int(self.z_dim), num_classes, lr=lr, binary=binary)
        if final_eval: 
            self.evaluator["mlp"], opt2 = eval_mlp(int(self.z_dim),num_classes,lr=lr, binary=binary)
            self.evaluator["clustering"] = GM(n_classes=num_classes)
            self.evaluator["knn"] = KNN(n_classes=num_classes, n_neighbors_max=20)
        else: opt2=None
        return opt1, opt2

    def sample_z(self, x):
        q_c_gnv_x = self.encoder(x) 
        c = reparameterize(*q_c_gnv_x, self.training)
        return q_c_gnv_x, c

    def forward(self, x):
        self.x_shape = x.shape
        q_c_gnv_x, c = self.sample_z(x)

        mu_x = torch.sigmoid(self.decoder(c))
        p_x_gnv_z = (mu_x, self.log_var_x) 

        return q_c_gnv_x, p_x_gnv_z, c

    def recon_and_entropy(self, x, q_c_gnv_x, p_x_gnv_z):
        (mu_z, log_var_z), (mu_x, log_var_x) = q_c_gnv_x, p_x_gnv_z

        if self.binarise:
            recon = nn.BCELoss(reduction="sum")(mu_x, x)
        else:
            recon = 0.5 * nn.MSELoss(reduction="sum")(mu_x, x)   

        entropy = 0.5 torch.sum(2*math.pi*log_var_z)
        return recon / math.exp(log_var_x), entropy

    def log_p_z(self, z):
        return self.prior_module['c'].forward(z)

    def loss(self, x, q_c_gnv_x, p_x_gnv_z, c ):
        recon, entropy = self.recon_and_entropy(x, q_c_gnv_x, p_x_gnv_z)
        prior = self.log_p_z(c)
        return recon, -entropy, -prior                                          

class GmmVAE(BaseVAE):
    """Assume: C --> Z --> X    ELBO:  p(x) ≥ ∫q(z|x) { log p(x|z) - q(z|x) + ∑ p_old(c|z) log p(z|c)p(c)"""

    def __init__(self, net_args, vae_args, *args, **kwargs):
        super(GmmVAE, self).__init__(net_args, **vae_args)

    def prior_init(self):
        super(GmmVAE, self).prior_init()          
        
    def forward(self, x):
        return super(GmmVAE, self).forward(x)

    def loss(self, x, q_c_gnv_x, p_x_gnv_z, c):
        return super(GmmVAE, self).loss(x, q_c_gnv_x, p_x_gnv_z, c)

class SelfSup(GmmVAE):
    def __init__( self, net_args, vae_args, selfsup_args, num_sim=2, *args, **kwargs ):
        self.selfsup_args = selfsup_args        #{'log_var_zy':log_var_zy, 'num_data':num_data, 'p_y_prior':p_y_prior}
        self.num_sim = num_sim
        super(SelfSup, self).__init__(net_args, vae_args)       # k unused, just avoid triggering k==1 issues

    def prior_init(self):
        self.prior_module = nn.ModuleDict({'c': Prior_Sup(num_sim=self.num_sim, **self.selfsup_args)})
    
    def forward(self, x):
        return super(SelfSup, self).forward(x)

    def loss(self, x, q_c_gnv_x, p_x_gnv_z, c):
        return super(SelfSup, self).loss(x, q_c_gnv_x, p_x_gnv_z, c)


class GmmVAEWithAugments(GmmVAE):
    """Assume: Y --> Z  --> X    ELBO: p(x,x',y) ≥ ∫q(z|x)log p(x|z)/q(z|x)  + ∫q(z'|x')log p(x'|z')/q(z'|x')         [ let q(z|x) = q(z|x,y) ]
                `--> Z' --> X'                                               +  log p(z|y)p(z'|y)p(y)
    """
    def __init__( self, net_args, vae_args, num_sim=2, kl=True, *args, **kwargs):
        self.num_sim = num_sim
        super(GmmVAEWithAugments, self).__init__(net_args, vae_args, num_sim=num_sim, kl=kl)

    def prior_init(self):
        super(GmmVAE, self).prior_init()

    def forward(self, x):
        return super(GmmVAEWithAugments, self).forward(x)

    def loss(self, x, q_c_gnv_x, p_x_gnv_z, c):
        return super(GmmVAEWithAugments, self).loss(x, q_c_gnv_x, p_x_gnv_z, c)


