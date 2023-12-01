from typing import List
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

from ihvp import compute_grads

def decode(token_ids):
    try:
        return "".join([chr(i) for i in token_ids])
    except:
        return chr(token_ids)

def query_influence(
        test_data: Dataset, # [query]
        train_data: Dataset, # subset of OpenWebText
        model: Module, # minGPT
        ihvp, 
        ihvp_tokenwise
    ) -> List:
    logging.info(f'Calculating influences for given queries')

    topk = 5

    all_top_training_samples = []
    all_top_influences = []
    all_top_token_influences = []
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

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

    data = []
    for i, (top_samples, top_influences, top_token_influences) in enumerate(
            zip(all_top_training_samples, all_top_influences, all_top_token_influences)
        ):
        query = f"{decode(test_data[i][0]['input_ids'])[0]}{decode(test_data[i][1]['input_ids'])}"
        influences = []
        for s, i, tok_inf in zip(top_samples, top_influences, top_token_influences):
            s = s.item()
            sample = f"{decode(train_data[s][0]['input_ids'])[0]}{decode(train_data[s][1]['input_ids'])}"
            influences.append({
                'sample': sample,
                'influence': i.item(),
                'token_influence': tok_inf.tolist()
            })
        data.append({
            'query': query,
            'influences': influences
        })

    return data

# TODO DUMP
# 1. Finetune model on a dataset
# 2. Select subset of data to calculate grads and ihvp
# 3. Store ihvp
# 4. run "inference" on input, obtain completion
# 5. consider input and completion as test point and calculate influence for data subset
# 6. display in cool way

# Figure out how to serve this
# Free colab doesnt give enough compute to calculate ihvp