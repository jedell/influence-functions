from torch import nn
from torch.utils.data import DataLoader
from model import GPT, GPTConfig
from dataset import train_dataset, test_dataset, tokenizer

config = GPTConfig(n_layer=1, n_head=1, n_embd=8)

model = GPT(config=config)

for name, mod in model.named_modules():
    print(mod._get_name())

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

def compute_factors(model: nn.Module, train_data):

    return

kfac_input_covs, kfac_grad_covs, pseudo_grads = compute_factors()

# 1. get a_l-1, use forward hook to save input to a layer l during the forward pass

# 2. get grad_loss, gradients of loss wrt output of linear transformation W_l a_l-1
#    using a backward hook on the linear layer that saves the gradient wrt the linear layer's output

