import torch
from torch import nn
from torch.utils.data import DataLoader
from model import GPT, GPTConfig, MLP
from dataset import train_dataset, test_dataset, tokenizer

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = GPTConfig(n_layer=1, n_head=1, n_embd=8)

model = GPT(config=config)

for name, mod in model.named_modules():
    print(mod._get_name())

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

def compute_factors(model: nn.Module, train_data):

    return

# kfac_input_covs, kfac_grad_covs, pseudo_grads = compute_factors()

# 1. get a_l-1, use forward hook to save input to a layer l during the forward pass
layer_inputs = {}

def hook_fn(module, input, output):
    layer_inputs[module] = input[0]

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        module.register_forward_hook(hook_fn)
    if isinstance(module, MLP):
        module.register_forward_hook(hook_fn)

# 2. get grad_loss, gradients of loss wrt output of linear transformation W_l a_l-1
#    using a backward hook on the linear layer that saves the gradient wrt the linear layer's output

for X, Y in train_dataset:

    model.zero_grad()
    
    x_ids = X['input_ids'].to(device)
    y_ids = Y['input_ids'].to(device)

    if len(x_ids.shape) == 1:
        x_ids = x_ids.unsqueeze(0)
        y_ids = y_ids.unsqueeze(0)

    logits, loss = model(x_ids, y_ids)

    # for each linear/mlp block, get a_l-1

