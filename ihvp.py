import os
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

import torch
from torch import nn
from torch.utils.data import DataLoader
import einops

from model import GPT, GPTConfig
from dataset import train_dataset, test_dataset, vocab_size

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

n_embd = 16
n_heads = 2
d_mlp = 32
n_layers = 2
vocab_size = vocab_size

logging.info(f"GPT Configuration: n_embd={n_embd}, n_heads={n_heads}, d_mlp={d_mlp}, n_layers={n_layers}, vocab_size={vocab_size}")

config = GPTConfig(n_layer=n_layers, n_head=n_heads, n_embd=n_embd, block_size=d_mlp, vocab_size=vocab_size)

model = GPT(config=config).to(device)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 1. get a_l-1, use forward hook to save input to a layer l during the forward pass
layer_inputs = {}

def forward_hook_fn(module, input):
    if isinstance(module, nn.Linear):
        layer_inputs[module] = torch.cat([
                input[0],
                torch.ones((input[0].shape[0], input[0].shape[1], 1)).to(input[0].device),
        ], dim=-1).clone().detach()

# 2. get grad_loss, gradients of loss wrt output of linear transformation W_l a_l-1
#    using a backward hook on the linear layer that saves the gradient wrt the linear layer's output

layer_grads = {}

def back_hook_fn(module, grad_input, grad_output):
    if isinstance(module, nn.Linear):
        layer_grads[module] = grad_output[0].clone().detach()

linear_layers = []
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and 'mlp' in name:
        # grab linear layers that are children of MLP
        # TODO adjust for hf gpt2 impl
        linear_layers.append(module)

# linear_layers = linear_layers[:-1]  # remove output token logits layer from calculations
logging.info(f"MLP Linear layers: {len(linear_layers)}")

def compute_ekfac_factors(model: nn.Module, train_dataset: DataLoader):
    logging.info(f'Computing EKFAC factors for {model._get_name()} - Dataset len: {len(train_dataset.dataset)}')
    kfac_input_covs = []
    kfac_grad_covs = []

    for module in linear_layers:
        module.register_forward_pre_hook(forward_hook_fn)
        module.register_full_backward_hook(back_hook_fn)

        kfac_input_covs.append(
            torch.zeros((module.weight.shape[1] + 1, module.weight.shape[1] + 1)).to(device)
        )

        kfac_grad_covs.append(
            torch.zeros((module.weight.shape[0], module.weight.shape[0])).to(device)
        )

    grads = [[] for _ in range(len(linear_layers))]
    tot = 0
    for X, Y in train_dataset:

        model.zero_grad()
        
        x_ids = X['input_ids'].to(device)
        y_ids = Y['input_ids'].to(device)

        if len(x_ids.shape) == 3:
            x_ids = x_ids.squeeze(1)
            y_ids = y_ids.squeeze(1)

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
            kfac_grad_covs[i] += grad_cov.mean(dim=0)
            
            # gradient of the loss wrt the weights
            w_grad = module.weight.grad
            if module.bias is not None:
                b_grad = module.bias.grad.unsqueeze(-1) 
                full_grad = torch.cat([w_grad, b_grad], dim=-1)

            else:
                full_grad = torch.cat(
                    [w_grad, torch.zeros([w_grad.shape[0], 1])],
                    dim=-1
                )
            grads[i].append(full_grad)

        tot+=1

    # eq 16
    kfac_input_covs = [A / tot for A in kfac_input_covs]
    kfac_grad_covs = [S / tot for S in kfac_grad_covs]

    logging.info("len kfac_input_covs: %s", len(kfac_input_covs))
    logging.info("len kfac_grad_covs: %s", len(kfac_grad_covs))

    return kfac_input_covs, kfac_grad_covs, grads

# 3. get search grads
# 
# TODO difference between psuedo grads and search grads

def compute_grads(model: nn.Module, train_dataset: DataLoader):

    grads = [[] for _ in range(len(linear_layers))]
    for X, Y in train_dataset: 
        model.zero_grad()

        x_ids = X['input_ids'].to(device)
        y_ids = Y['input_ids'].to(device)

        if len(x_ids.shape) == 3:
            x_ids = x_ids.squeeze(1)
            y_ids = y_ids.squeeze(1)

        logits, loss = model(x_ids, y_ids)

        loss.backward()
        for i, module in enumerate(linear_layers):
            w_grad = module.weight.grad
            if module.bias is not None:
                b_grad = module.bias.grad.unsqueeze(-1) 
                full_grad = torch.cat([w_grad, b_grad], dim=-1)
            else:
                full_grad = torch.cat(
                    [w_grad, torch.zeros([w_grad.shape[0], 1])],
                    dim=-1
                )
            grads[i].append(full_grad)

    return grads

# 4. compute ekfac inverse hessian vector product
# eq 20
def compute_lambda_ii(train_grads, q_a, q_s):
    n_examples = len(train_grads)
    squared_projections_sum = 0.0
    for j in range(n_examples):
        dtheta = train_grads[j]
        result = (q_s @ dtheta @ q_a.T).view(-1)
        squared_projections_sum += result**2
    lambda_ii_avg = squared_projections_sum / n_examples
    return lambda_ii_avg

def compute_ihvp(kfac_input_covs, kfac_grad_covs, pseudo_grads, mod_grads, damping=0.001):
    logging.info(f"Computing IHVP for {model._get_name()} with damping factor {damping}.")
    
    # TODO equation 31 from https://arxiv.org/pdf/2308.03296.pdf
    ihvp = []
    ihvp_tokenwise = []
    for i in range(len(mod_grads)):
        V = torch.stack(mod_grads[i])

        # eigendecompositions for eq 18
        q_a, _, q_a_t = torch.svd(kfac_input_covs[i])
        q_s, _, q_s_t = torch.svd(kfac_grad_covs[i])

        lambda_ii = compute_lambda_ii(pseudo_grads[i], q_a, q_s)
        ekfac_diag_damped_inv = 1.0 / (lambda_ii + damping)
        ekfac_diag_damped_inv = ekfac_diag_damped_inv.reshape(
            (V.shape[-2], V.shape[-1])
        )
        intermediate_result = torch.einsum("bij,jk->bik", V, q_a_t)
        intermediate_result = torch.einsum("ji,bik->bjk", q_s, intermediate_result)
        result = intermediate_result / ekfac_diag_damped_inv.unsqueeze(0)
        ihvp_component = torch.einsum("bij,jk->bik", result, q_a)
        ihvp_component = torch.einsum("ji,bik->bjk", q_s_t, ihvp_component)
        ihvp_tokenwise.append(ihvp_component)
        ihvp_component = einops.rearrange(ihvp_component, "b j k -> b (j k)")

        ihvp.append(ihvp_component)

    logging.info(f'Finished computing IHVP.')
    return torch.cat(ihvp, dim=-1), ihvp_tokenwise


model_name = model._get_name()
num_params = "%.2fM" % (model.get_num_params()/1e6,)

if os.path.exists(f'{model_name}_kfac_input_covs_{num_params}.pt'):
    kfac_input_covs = torch.load(f'{model_name}_kfac_input_covs_{num_params}.pt')
    kfac_grad_covs = torch.load(f'{model_name}_kfac_grad_covs_{num_params}.pt')
    pseudo_grads = torch.load(f'{model_name}_pseudo_grads_{num_params}.pt')
else:
    kfac_input_covs, kfac_grad_covs, pseudo_grads = compute_ekfac_factors(model, train_dataloader)
    torch.save(kfac_input_covs, f'{model_name}_kfac_input_covs_{num_params}.pt')
    torch.save(kfac_grad_covs, f'{model_name}_kfac_grad_covs_{num_params}.pt')
    torch.save(pseudo_grads, f'{model_name}_pseudo_grads_{num_params}.pt')

if os.path.exists(f'{model_name}_computed_grads_{num_params}.pt'):
    computed_grads = torch.load(f'{model_name}_computed_grads_{num_params}.pt')
else:
    logging.info("Computing train data search gradients.")
    computed_grads = compute_grads(model, train_dataloader)
    torch.save(computed_grads, f'{model_name}_computed_grads_{num_params}.pt')

if os.path.exists(f'{model_name}_ihvp_{num_params}.pt'):
    ihvp = torch.load(f'{model_name}_ihvp_{num_params}.pt')
    ihvp_tokenwise = torch.load(f'{model_name}_ihvp_tokenwise_{num_params}.pt')
else:
    ihvp, ihvp_tokenwise  = compute_ihvp(kfac_input_covs, kfac_grad_covs, pseudo_grads, computed_grads)
    torch.save(ihvp, f'{model_name}_ihvp_{num_params}.pt')
    torch.save(ihvp_tokenwise, f'{model_name}_ihvp_tokenwise_{num_params}.pt')

# above can be done once for model finetuned on specific dataset

# 5. for each test point:
# get gradient of test point wrt loss?
# calculate influence with ihvp and test_grad
# sample topk

exit()

topk = 5

all_top_training_samples = []
all_top_influences = []
all_top_token_influences = []

logging.info(f'Calculating influences for query data.')

for query in test_dataloader:

    grads = compute_grads(model, [query])
    tokenwise_query_grads = grads

    query_grad = torch.cat(
        [q[0].view(-1) for q in grads]
    )
    top_influences = -1 * torch.einsum("ij,j->i", ihvp, query_grad)

    layer_token_influences = []
    seq_len = ihvp_tokenwise[-1].shape[-1]
    for l, g in zip(ihvp_tokenwise, tokenwise_query_grads):
        # some ihvp layer grads have more tokens, intermediate linear layers from mlp
        # TODO what do we do
        if l.shape[-1] == seq_len:
            token_influence = torch.einsum("ijk,jk->ik", l, g[0])
            layer_token_influences.append(token_influence)

    token_influences = torch.sum(torch.stack(layer_token_influences), dim=0)

    top_influences, top_samples = torch.topk(top_influences, topk)
    all_top_training_samples.append(top_samples)
    all_top_influences.append(top_influences)

    top_token_influences = token_influences[top_samples]
    all_top_token_influences.append(top_token_influences)

def decode(token_ids):
    try:
        return "".join([chr(i) for i in token_ids])
    except:
        return chr(token_ids)

# TODO ensure we are doing this per token when using real data 
for i, (top_samples, top_influences, top_token_influences) in enumerate(
        zip(all_top_training_samples, all_top_influences, all_top_token_influences)
    ):
        print(f"Query: {decode(test_dataset[i][0]['input_ids'])[0]}{decode(test_dataset[i][1]['input_ids'])}")
        print(f"Top {topk} training samples and their influences:")
        for s, i, tok_inf in zip(top_samples, top_influences, top_token_influences):
            s = s.item()
            sample = f"{decode(train_dataset[s][0]['input_ids'])[0]}{decode(train_dataset[s][1]['input_ids'])}"
            print(
                f"{sample} Influence: {i}"
            )
            for char, influence in zip(sample, tok_inf):
                print(f"{char}, {influence}")

# import json

# data = []
# for i, (top_samples, top_influences, top_token_influences) in enumerate(
#         zip(all_top_training_samples, all_top_influences, all_top_token_influences)
#     ):
#     query = f"{decode(test_dataset[i][0]['input_ids'])[0]}{decode(test_dataset[i][1]['input_ids'])}"
#     influences = []
#     for s, i, tok_inf in zip(top_samples, top_influences, top_token_influences):
#         s = s.item()
#         sample = f"{decode(train_dataset[s][0]['input_ids'])[0]}{decode(train_dataset[s][1]['input_ids'])}"
#         influences.append({
#             'sample': sample,
#             'influence': i.item(),
#             'token_influence': tok_inf.tolist()
#         })
#     data.append({
#         'query': query,
#         'influences': influences
#     })

# with open('data.json', 'w') as f:
#     json.dump(data, f)

# TODO DUMP
# 1. Finetune model on a dataset
# 2. Select subset of data to calculate grads and ihvp
# 3. Store ihvp
# 4. run "inference" on input, obtain completion
# 5. consider input and completion as test point and calculate influence for data subset
# 6. display in cool way

# Figure out how to serve this
# Free colab doesnt give enough compute to calculate ihvp