import runpod
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random
from random import sample
random.seed(42)

def dataset_sample(dataset, n_samples):
    indices = sample(range(len(dataset)), n_samples)
    return [dataset[i] for i in indices]

dataset_length = 10000

class TinyStoriesDataset(Dataset):
    def __init__(self, split, tokenizer, path='roneneldan/TinyStories', block_size=1024, seed=42, take=10000, skip=0):
        self.split = split
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.dataset = load_dataset(path, split=split, streaming=False).shuffle(seed=seed) #.skip(skip).take(take).with_format('torch')

    def __iter__(self):
        for item in self.dataset:
            text = item['text']
            encoding = self.tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True, padding='max_length', truncation=True, max_length=self.block_size)
            input_ids = encoding['input_ids'].squeeze()

            labels = input_ids.clone()
            labels[:-1] = input_ids[1:]
            labels[-1] = self.tokenizer.pad_token_id

            yield {'input_ids': input_ids}, {'input_ids': labels}

    def __getitem__(self, index):
        item = self.dataset[index]
        text = item['text']
        encoding = self.tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True, padding='max_length', truncation=True, max_length=self.block_size)
        input_ids = encoding['input_ids'].squeeze()

        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = self.tokenizer.pad_token_id

        return {'input_ids': input_ids}, {'input_ids': labels}

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def get_block_size(self):
        return self.block_size

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size

model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

model.to(device)

# 1. get a_l-1, use forward hook to save input to a layer l during the forward pass
layer_inputs = {}

def get_a_l_minus_1(name):

    def forward_hook_fn(module, input, output):
        layer_inputs[name] = torch.cat([
                input[0],
                torch.ones((input[0].shape[0], input[0].shape[1], 1)).to(input[0].device),
        ], dim=-1).clone().detach()
    return forward_hook_fn

# 2. get grad_loss, gradients of loss wrt output of linear transformation W_l a_l-1
#    using a backward hook on the linear layer that saves the gradient wrt the linear layer's output

layer_grads = {}

def get_grad_loss(name):
    def back_hook_fn(module, grad_input, grad_output):
        layer_grads[name] = grad_output[0].clone().detach()
    return back_hook_fn

linear_layers = []
# specifc to GPT Neo
for name, module in model.named_modules():
    # INPUT TO MLP (batch_size, seq_len, input_dim), a_l-1
    if name.split('.')[-1] == 'mlp':
      module.register_forward_hook(get_a_l_minus_1(f"{name}.c_fc"))
      # linear_layers.append(module)
    # FIRST LINEAR LAYER self.linear(a_l_minus_1) proj
    # https://github.com/nrimsky/InfluenceFunctions/blob/72334413bde13cde66366d2e524f87f38d2adea2/mini_transformer.py#L98C1-L99C1
    if isinstance(module, nn.Linear) and name.split('.')[-1] == 'c_fc':
      module.register_full_backward_hook(get_grad_loss(name))
      linear_layers.append((name, module))
      layer_grads[name] = {}
      layer_inputs[name] = {}

def compute_grads(model: nn.Module, train_dataset: DataLoader):

    grads = [[] for _ in range(len(linear_layers))]
    full_tokenwise_grads = [[] for _ in range(len(linear_layers))]
    for X, Y in train_dataset:
        model.zero_grad()

        x_ids = X['input_ids'].to(device)
        y_ids = Y['input_ids'].to(device)

        if len(x_ids.shape) == 3:
            x_ids = x_ids.squeeze(1)
            y_ids = y_ids.squeeze(1)

        if len(x_ids.shape) == 1:
            x_ids = x_ids.unsqueeze(0)
            y_ids = y_ids.unsqueeze(0)

        output = model(x_ids, labels=y_ids)
        logits, loss = output['logits'], output['loss']

        loss.backward()
        for i, name_module in enumerate(linear_layers):
            name, module = name_module
            w_grad = module.weight.grad
            if module.bias is not None:
                b_grad = module.bias.grad.unsqueeze(-1)
                full_grad = torch.cat([w_grad, b_grad], dim=-1)
            else:
                full_grad = torch.cat(
                    [w_grad, torch.zeros([w_grad.shape[0], 1])],
                    dim=-1
                )

            d_s_l = layer_grads[name]
            a_l_1 = layer_inputs[name]
            tokenwise_grads = torch.einsum('bix,bjy->bixy', d_s_l, a_l_1)
            full_tokenwise_grads[i].append(tokenwise_grads.squeeze(0))

            grads[i].append(full_grad)

    return grads, full_tokenwise_grads

ihvp = torch.load(f'/TinyStories-ihvp-f.pt')
# from hf
train_dataset_hf = TinyStoriesDataset('train', tokenizer, block_size=2048, path='roneneldan/TinyStories')
train_dataset = dataset_sample(train_dataset_hf, dataset_length)

def get_influences(job):
    query = job['input']['prompt']
    topk = job['input'].get('k', 5)

    query_enc = tokenizer.encode(query, return_tensors='pt').to(device)
    out = model.generate(query_enc, max_length=300)

    z_m_ids = torch.nn.functional.pad(out[0], (0, 2048 - len(out[0])), value=50256)
    z_m_label = z_m_ids.clone()
    z_m_label[:-1] = z_m_ids[1:]
    z_m_label[-1] = tokenizer.pad_token_id

    z_m = [({'input_ids': z_m_ids}, {'input_ids': z_m_label})]

    all_top_training_samples = []
    all_top_influences = []

    for query, compl in z_m:

        grads, token_grads = compute_grads(model, [(query, compl)])

        query_grad = torch.cat(
            [q[0].view(-1) for q in grads]
        )
        token_query_grads = torch.cat(
            [tq[0].view(tq[0].shape[0], -1) for tq in token_grads],
            dim=-1
        )

        # eq 30
        top_influences = -1 * torch.einsum("ij,j->i", ihvp, query_grad)

        top_influences, top_samples = torch.topk(top_influences, topk)
        all_top_training_samples.append(top_samples)
        all_top_influences.append(top_influences)

        top_influence_tokenwise = []
        for i, r_t in enumerate(token_query_grads):
            top_influence_tokenwise.append(-1 * torch.einsum("ij,j->i", ihvp, r_t))

    top_influence_sentences = []
    for k, (top_samples, top_influences) in enumerate(
            zip(all_top_training_samples, all_top_influences)
        ):
        influence_sentences = []
        for s, i in zip(top_samples, top_influences):
            s = s.item()
            query = tokenizer.decode(z_m[k][0]['input_ids']).replace("<|endoftext|>", "")
            sample = tokenizer.decode(train_dataset[s][0]['input_ids'], skip_special_tokens=True).replace("<|endoftext|>", "")

            tokenwise_influences = []
            for v, r_t in enumerate(top_influence_tokenwise):
                token_id = train_dataset[s][0]['input_ids'][v]
                token = tokenizer.decode(token_id, skip_special_tokens=True)
                influence = r_t[s].item()

                tokenwise_influences.append({"token": token, "influence": influence})
                # stop if encounter eot
                if token_id.item() == 50256:
                    break

            influence_sentences.append({
                "sample": sample,
                "influence": i.item(),
                "tokens": tokenwise_influences
            })

        top_influence_sentences.append({"samples": influence_sentences, "query": query})
    
    return {"influences": top_influence_sentences}

runpod.serverless.start({"handler": get_influences})