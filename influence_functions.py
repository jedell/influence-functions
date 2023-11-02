import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
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


class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear1 = nn.Linear(100, 50)
        self.linear2 = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, src):
        output = self.relu(self.linear1(src))
        output = self.linear2(output)
        return output

small_model = SimpleLinearModel()

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

# TODO its been right here all along, i just didnt understand it 
# https://github.com/wiseodd/natural-gradients/blob/master/pytorch/ekfac.py#L59

state = {}
_fwd_handles = []
_bwd_handles = []

h_s = []
gy_s = []

def _save_input(module: nn.Module, i):
    print("input:", module, i)
    # return i[0]
    h_s.append(i[0])
    # state[module.__class__.__name__]['x'] = i[0]

def _save_grad_output(module, grad_input, grad_output):
    print("grad_output:", grad_output)
    # state[module]['gy'] = grad_output[0] * grad_output[0].size(0)
    gy_s.append(grad_output[0] * grad_output[0].size(0))

# for name, param in small_model.named_parameters():
#     print(name, param)
#     exit()

linear = nn.Linear(32, 10, True)

for name, module in small_model.named_modules():
    handle = module.register_forward_pre_hook(_save_input)
    _fwd_handles.append(handle)
    handle = module.register_full_backward_hook(_save_grad_output)
    _bwd_handles.append(handle)

    # can we get params during training loop model.parameters
    # print(module.__dict__.keys())
    # exit()
    # params = [module.weight]
    # if module.bias is not None:
    #     params.append(module.bias)
    # d = {'params': params, 'mod': module, 'layer_type': name}

# https://arxiv.org/pdf/1806.03884.pdf
def ekfac(model, D_train: DataLoader, iterations=100):

    for i in range(iterations):
        D = next(iter(D_train))
        X, Y = D
        # Forward pass to obtain h (inputs for a layer)

        # Backward pass to obtain δ (the backpropagated gradient on activation a)
        logits = model(X)
        print(logits)
        print("h vals:", h_s)

        loss = nn.CrossEntropyLoss()(logits, Y)

        loss.backward()

        print("gy vals:", gy_s)

        model.zero_grad()

        for l in model.named_modules():
            if i % n == 0: # Amortize eigendecomposition
                compute_eigenbasis(D, l)
            
            compute_scalings(D, l)
        
            grad_mini = ... # E_{(x,y)} ∈ D [ ∇^(l)_θ (x, y) ]

            update_parameters(grad_mini, l)

train_data = torch.randn((100, 100)) 
train_labels = torch.randint(0, 10, (100,))
train_dataset = TensorDataset(train_data, train_labels)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

ekfac(small_model, train_dataloader, 10)