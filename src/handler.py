import runpod
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

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
    if isinstance(module, nn.Linear) and 'mlp' in name: # out_proj ???
        # grab linear layers everytime
        linear_layers.append(module)

linear_layers = linear_layers[:-1]  # remove output token logits layer from calculations

def compute_grads(model: nn.Module, train_dataset: DataLoader):

    grads = [[] for _ in range(len(linear_layers))]
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

ihvp = torch.load(f'/content/TinyStories_ihvp_33000000.pt')
train_dataset = TinyStoriesDataset('train', tokenizer, block_size=2048, path='/TinyStories')

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

        grads = compute_grads(model, [(query, compl)])

        query_grad = torch.cat(
            [q[0].view(-1) for q in grads]
        )

        # eq 30
        top_influences = -1 * torch.einsum("ij,j->i", ihvp, query_grad)

        top_influences, top_samples = torch.topk(top_influences, topk)
        all_top_training_samples.append(top_samples)
        all_top_influences.append(top_influences)

    top_influence_sentences = []
    for i, (top_samples, top_influences) in enumerate(
            zip(all_top_training_samples, all_top_influences)
        ):
        influence_sentences = []
        for s, i in zip(top_samples, top_influences):
            s = s.item()
            sample = f"{tokenizer.decode(train_dataset[s][0]['input_ids'])}"
            influence_sentences.append({"sample": sample, "influence": i.item()})
        top_influence_sentences.append(influence_sentences)

    return {"influences": top_influence_sentences}

runpod.serverless.start({"handler": get_influences})