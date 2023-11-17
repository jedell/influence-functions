import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import logging

from model import GPT, GPTConfig, MLP
from dataset import train_dataset, test_dataset, tokenizer

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = GPTConfig(n_layer=1, n_head=1, n_embd=8)

model = GPT(config=config)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# kfac_input_covs, kfac_grad_covs, pseudo_grads = compute_factors()

# 1. get a_l-1, use forward hook to save input to a layer l during the forward pass
layer_inputs = {}

def forward_hook_fn(module, input):
    if isinstance(module, nn.Linear):
        layer_inputs[module] = torch.cat([
                input[0],
                torch.ones((input[0].shape[0], input[0].shape[1], 1)).to(input[0].device),
        ], dim=-1).clone()

# 2. get grad_loss, gradients of loss wrt output of linear transformation W_l a_l-1
#    using a backward hook on the linear layer that saves the gradient wrt the linear layer's output

layer_grads = {}

def back_hook_fn(module, grad_input, grad_output):
    if isinstance(module, nn.Linear):
        layer_grads[module] = grad_output[0]

linear_layers = []

def compute_ekfac_factors(model: nn.Module, train_dataset: Dataset):

    kfac_input_covs = []
    kfac_grad_covs = []


    for _, module in model.named_modules():
        # TODO register miniGPT MLP, first linear layer???
        if isinstance(module, nn.Linear):
            module.register_forward_pre_hook(forward_hook_fn)
            module.register_full_backward_hook(back_hook_fn)

            kfac_input_covs.append(
                torch.zeros((module.weight.shape[1] + 1, module.weight.shape[1] + 1)).to(device)
            )

            kfac_grad_covs.append(
                torch.zeros((module.weight.shape[0], module.weight.shape[0])).to(device)
            )

            # grab linear layers
            linear_layers.append(module)

    grads = [[] for _ in range(len(linear_layers))]
    tot = 0
    for X, Y in train_dataset:

        model.zero_grad()
        
        x_ids = X['input_ids'].to(device)
        y_ids = Y['input_ids'].to(device)

        if len(x_ids.shape) == 1:
            x_ids = x_ids.unsqueeze(0)
            y_ids = y_ids.unsqueeze(0)

        logits, loss = model(x_ids, y_ids)

        # for each linear/mlp block, get a_l-1
        for i, module in enumerate(linear_layers):
            a_l_1 = layer_inputs[module]
            input_covs = torch.einsum("...ti,...tj->tij", a_l_1, a_l_1)
            kfac_input_covs[i] += input_covs.mean(dim=0)

        loss.backward()

        # grad_loss = layer_grads[module]
        for i, module in enumerate(linear_layers):
            d_s_l = layer_grads[module]
            grad_cov = torch.einsum("...ti,...tj->tij", d_s_l, d_s_l)
            # TODO 
            # RuntimeError: The size of tensor a (24) must match the size of tensor b (8) at non-singleton dimension 1
            kfac_grad_covs[i] += grad_cov.mean(dim=0)
            
            # gradient of the loss wrt the weights
            w_grad = module.weight.grad
            b_grad = module.bias.grad.unsqueeze(-1)
            full_grad = torch.cat([w_grad, b_grad], dim=-1)
            grads[i].append(full_grad)

        tot+=1

    kfac_input_covs = [A / tot for A in kfac_input_covs]
    kfac_grad_covs = [S / tot for S in kfac_grad_covs]

    return kfac_input_covs, kfac_grad_covs, grads

# 3. get search grads
# 
# TODO difference between psuedo grads and serach grads

def compute_grads(model: nn.Module, train_dataset: Dataset):
    grads = [[] for _ in range(len(linear_layers))]

    for X, Y in train_dataset: 
        model.zero_grad()

    x_ids = X['input_ids'].to(device)
    y_ids = Y['input_ids'].to(device)

    if len(x_ids.shape) == 1:
        x_ids = x_ids.unsqueeze(0)
        y_ids = y_ids.unsqueeze(0)

    logits, loss = model(x_ids, y_ids)

    loss.backward()

    for i, module in enumerate(linear_layers):
        w_grad = module.weight.grad
        b_grad = module.bias.grad.unsqueeze(-1)
        full_grad = torch.cat([w_grad, b_grad], dim=-1)
        grads[i].append(full_grad)

    return grads

# 4. compute ekfac inverse hessian vector product

def compute_ekfac_ihvp(kfac_input_covs, kfac_grad_covs, pseudo_grads, computed_grads, damping=0.001):
    ihvp = []

kfac_input_cov, kfac_grad_cov, pseudo_grads = compute_ekfac_factors(model, train_dataset)

# TODO reread paper to understand this
computed_grads = compute_grads(model, train_dataset)






# 5. for each test point:
# get gradient of test point wrt loss?
# calculate influence with ihvp and test_grad
# sample topk
# profit???