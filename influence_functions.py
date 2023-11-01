import torch
from torch.utils.data import DataLoader
import time
import datetime
import numpy as np
import copy
import logging
from transformers import PreTrainedModel

def influence_function(
        model: PreTrainedModel, train_dataloader: DataLoader, test_dataloader: DataLoader
    ):

    return

# https://arxiv.org/pdf/2308.03296.pdf Equation 15

def calc_layer_pseudo_gradient(a_l_m1, s_l, w_l, b_l):
    # Compute the gradient of the log probability with respect to the outputs of layer l
    s_l.requires_grad_(True)
    log_prob = torch.log(torch.softmax(s_l, dim=-1)) # TODO use softmax layer from model?
    pseudo_out_l = torch.autograd.grad(log_prob.sum(), s_l, retain_graph=True)[0]

    # Compute the pseudo-gradient for the weights of layer l
    pseudo_w_l = torch.kron(a_l_m1, pseudo_out_l)

    return pseudo_w_l


def calc_ek_fac(model: PreTrainedModel):
    named_params = list(model.named_parameters())

    m_aa, m_gg = {}, {}
    Q_a,  Q_g = {}, {}
    d_a, d_g = {}, {}
    S_l = {}

    def eigen_decomposition(params):
        eps = 1e-10  # for numerical stability
        d_a[params], Q_a[params] = torch.symeig(
            m_aa[params], eigenvectors=True)
        d_g[params], Q_g[params] = torch.symeig(
            m_gg[params], eigenvectors=True)

        d_a[params].mul_((d_a[params] > eps).float())
        d_g[params].mul_((d_g[params] > eps).float())
        # if steps != 0:
        S_l[params] = d_g[params].unsqueeze(1) @ d_a[params].unsqueeze(0)

    for name, params in named_params:
        print(name)
        
        eigen_decomposition(params)

# https://github.com/tfjgeorge/nngeometry/blob/master/nngeometry/object/pspace.py#L8

class EKFAC:
    def __init__(self, model, function=None, n_output=1,
                 centering=False):
        self.model = model
        self.handles = []
        self.xs = dict()
        self.n_output = n_output
        self.centering = centering
        self.layers = list(model.named_parameters())

        if function is None:
            function = lambda *x: model(x[0])
        self.function = function

        # maps parameters to their position in flattened representation
        # self.l_to_m, self.m_to_l = \
        #     self.layer_collection.get_layerid_module_maps(model)
        
    def do_ekfac(self):
        evecs = dict()
        diags = dict()
        
        kfac_blocks = self.get_kfac_blocks(...)

    def get_kfac_blocks(self):
        for name, params in self.layers:
            print(name)


small_model = torch.nn.Linear(10, 1)

def convergence_not_reached():

    return True

def compute_eigenbasis(D, l):
    hh_outer = torch.einsum('bi,bj->bij', D['h'][l], D['h'][l])
    
    delta_delta_outer = torch.einsum('bi,bj->bij', D['delta'][l], D['delta'][l])
    
    hh_outer_mean = hh_outer.mean(dim=0)
    delta_delta_outer_mean = delta_delta_outer.mean(dim=0)
    
    S_A, U_A = torch.symeig(hh_outer_mean, eigenvectors=True)
    S_B, U_B = torch.symeig(delta_delta_outer_mean, eigenvectors=True)
    
    return U_A, S_A, U_B, S_B

def compute_scalings(D, l):

    return

def update_parameters(grad_mini, l):

    return

n = 32 # recompute eigenbasis after n minibatches
lr = 0.01 # learning rate
e = 0.01 # damping

# https://arxiv.org/pdf/1806.03884.pdf
def ekfac(model: PreTrainedModel, D_train: DataLoader):

    i = 0
    while convergence_not_reached(): # TODO for looop??? what is convergence
        D = next(iter(D_train))

        # Forward pass to obtain h (inputs for a layer)
        h = {}
        for l, (name, module) in enumerate(model.named_modules()):
            if isinstance(module, torch.nn.Linear):
                h[name] = module.input

        # Backward pass to obtain δ (the backpropagated gradient on activation a)
        model.zero_grad()
        logits = model(D)
        loss = model.loss_function(logits, D)
        loss.backward()

        delta = {}
        for l, (name, module) in enumerate(model.named_modules()):
            if isinstance(module, torch.nn.Linear):
                delta[name] = module.weight.grad

        for l in model.named_modules():
            if i % n == 0: # Amortize eigendecomposition
                compute_eigenbasis(D, l)
            
            compute_scalings(D, l)
        
            grad_mini = ... # E_{(x,y)} ∈ D [ ∇^(l)_θ (x, y) ]

            update_parameters(grad_mini, l)
