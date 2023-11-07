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

def compute_kfe(D, module: nn.Module, state):
    layer_type = module._get_name()
    x = state[module]['x']
    gy = state[module]['gy']

    print(module, x.shape, gy.shape)
    print(layer_type)

    if layer_type == 'Linear':
        x = x.data.t()

        if module.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)

        xxt = torch.mm(x, x.t()) / float(x.shape[1])

        Ex, state['kfe_x'] = torch.linalg.eigh(xxt,UPLO='U')

        gy = gy.data.t()
        state['num_locations'] = 1

        ggt = torch.mm(gy, gy.t()) / float(gy.shape[1])
        Eg, state['kfe_gy'] = torch.linalg.eigh(ggt, UPLO='U')
        state['m2'] = Eg.unsqueeze(1) * Ex.unsqueeze(0) * state['num_locations']

def apply_preconditioning(weight, bias, module, state, alpha=0.75, eps=0.1):
    kfe_x = state['kfe_x']
    kfe_gy = state['kfe_gy']
    m2 = state['m2']
    g = weight.grad.data
    s = g.shape
    bs = state[module]['x'].size(0)
    if bias is not None:
        gb = bias.grad.data
        g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)
    g_kfe = torch.mm(torch.mm(kfe_gy.t(), g), kfe_x)
    m2.mul_(alpha).add_((1. - alpha) * bs, g_kfe**2)
    g_nat_kfe = g_kfe / (m2 + eps)
    g_nat = torch.mm(torch.mm(kfe_gy, g_nat_kfe), kfe_x.t())
    if bias is not None:
        gb = g_nat[:, -1].contiguous().view(*bias.shape)
        bias.grad.data = gb
        g_nat = g_nat[:, :-1]
    g_nat = g_nat.contiguous().view(*s)
    weight.grad.data = g_nat

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

def _save_input(module: nn.Module, i):
    state[module]['x'] = i[0]
    # h_s.append(i[0])

def _save_grad_output(module, grad_input, grad_output):
    state[module]['gy'] = grad_output[0] * grad_output[0].size(0)
    # gy_s.append(grad_output[0] * grad_output[0].size(0))

# for name, param in small_model.named_parameters():
#     print(name, param)
#     exit()

linear = nn.Linear(32, 10, True)

for name, module in small_model.named_children():
    state[module] = {}
    state[module]['name'] = name

    handle = module.register_forward_pre_hook(_save_input)
    _fwd_handles.append(handle)
    handle = module.register_full_backward_hook(_save_grad_output)
    _bwd_handles.append(handle)
        
    if name == '':
        pass

    # can we get params during training loop model.parameters
    # print(module.__dict__.keys())
    # exit()
    # params = [module.weight]
    # if module.bias is not None:
    #     params.append(module.bias)
    # d = {'params': params, 'mod': module, 'layer_type': name}

print(state)

def compute_eigenbasis():

    return

# https://arxiv.org/pdf/1806.03884.pdf
def ekfac(model: nn.Module, D_train: DataLoader, iterations=100):

    A = []  # KFAC A
    G = []  # KFAC G

    for name, param in model.named_parameters():
        print(name, param)
        # for Wi in model:
        #     A.append(torch.zeros(Wi.size(1)))
        #     G.append(torch.zeros(Wi.size(0)))

    for i in range(iterations):
        D = next(iter(D_train))
        X, Y = D
        # Forward pass to obtain h (inputs for a layer)

        # Backward pass to obtain δ (the backpropagated gradient on activation a)
        logits = model(X)

        loss = nn.CrossEntropyLoss()(logits, Y)

        loss.backward()

        # step
        for name, module in model.named_children():
            compute_eigenbasis()
            

        model.zero_grad()


train_data = torch.randn((100, 100)) 
train_labels = torch.randint(0, 10, (100,))
train_dataset = TensorDataset(train_data, train_labels)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

def get_kfac_blocks(loader):



    return

eigenvectors = {}
diagonals = {}

kfac_blocks = get_kfac_blocks(train_dataloader)

ekfac(small_model, train_dataloader, 10)

# from nngeometry.nngeometry.object import PMatEKFAC
# from nngeometry.nngeometry.metrics import FIM

# G = FIM(
#     model=small_model,
#     loader=train_dataloader,
#     representation=PMatEKFAC,
#     n_output=10,
#     # variant='classif_logits',
# )

# print(G.trace())

# if len(list(module.parameters())) == 2:
#     weight, bias = module.parameters()
# else:
#     weight = list(module.parameters())[0]
#     bias = None
# curr_state = state

# if i % n == 0: # Amortize eigendecomposition
#     compute_kfe(D, module, curr_state)

# # compute_scalings(D, l)

# apply_preconditioning(weight, bias, module, curr_state)

# # grad_mini = ... # E_{(x,y)} ∈ D [ ∇^(l)_θ (x, y) ]

# # update_parameters(grad_mini, module)
