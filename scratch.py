import torch
from torch import nn
from torch.utils.data import DataLoader
import logging

from model import GPT, GPTConfig, MLP
from dataset import train_dataset, test_dataset, tokenizer

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = GPTConfig(n_layer=1, n_head=1, n_embd=8)

model = GPT(config=config)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

def compute_factors(model: nn.Module, train_data):

    return

# kfac_input_covs, kfac_grad_covs, pseudo_grads = compute_factors()

# 1. get a_l-1, use forward hook to save input to a layer l during the forward pass
layer_inputs = {}

def forward_hook_fn(module, input, output):
    if isinstance(module, MLP):
        print("mlp in", input[0].shape)
        print("mlp out", output[0].shape)
    layer_inputs[module] = torch.cat([
            input[0],
            torch.ones((input[0].shape[0], input[0].shape[1], 1)).to(input[0].device),
       ], dim=-1).clone()

# 2. get grad_loss, gradients of loss wrt output of linear transformation W_l a_l-1
#    using a backward hook on the linear layer that saves the gradient wrt the linear layer's output

layer_grads = {}

def back_hook_fn(module, grad_input, grad_output):
    if isinstance(module, MLP):
        print("mlp grad out", grad_output)
    layer_grads[module] = grad_output[0]

kfac_input_covs = []
kfac_grad_covs = []

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        module.register_forward_hook(forward_hook_fn)
        module.register_full_backward_hook(back_hook_fn)

        kfac_input_covs.append(
            torch.zeros((module.weight.shape[1] + 1, module.weight.shape[1] + 1)).to(device)
        )

        kfac_grad_covs.append(
            torch.zeros((module.weight.shape[0], module.weight.shape[0])).to(device)
        )
grads = [[] for _ in range(len([module for _, module in model.named_modules() if isinstance(module, nn.Linear)]))]
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
    i = 0
    for named_modules in model.named_modules():
        name, module = named_modules
        if isinstance(module, nn.Linear):
            a_l_1 = layer_inputs[module]
            input_covs = torch.einsum("...ti,...tj->tij", a_l_1, a_l_1)
            kfac_input_covs[i] += input_covs.mean(dim=0)
            i+=1

    loss.backward()

    # grad_loss = layer_grads[module]
    i = 0
    for named_modules in model.named_modules():
        name, module = named_modules
        if isinstance(module, nn.Linear):
            d_s_l = layer_grads[module]
            grad_cov = torch.einsum("...ti,...tj->tij", d_s_l, d_s_l)
            kfac_grad_covs[i] += grad_cov.mean(dim=0)
            
            # gradient of the loss wrt the weights
            w_grad = module.weight.grad
            b_grad = module.bias.grad.unsqueeze(-1)
            full_grad = torch.cat([w_grad, b_grad], dim=-1)
            grads[i].append(full_grad)

    tot+=1
    
kfac_input_covs = [A / tot for A in kfac_input_covs]
kfac_grad_covs = [S / tot for S in kfac_grad_covs]