import random
import numpy as np

import torch

from transformers import AutoTokenizer, GPT2LMHeadModel
from alpaca_data import make_supervised_data_module, smart_tokenizer_and_embedding_resize, create_dataloader
from datasets import load_dataset
import logging

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

dataset = load_dataset("tatsu-lab/alpaca")

tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    padding_side="right"
    )

special_tokens_dict = dict()
if tokenizer.pad_token is None:
    special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
if tokenizer.eos_token is None:
    special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
if tokenizer.bos_token is None:
    special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
if tokenizer.unk_token is None:
    special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

model = GPT2LMHeadModel.from_pretrained("gpt2")

smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def extract_final_layer_info(model, input_ids, input_mask, labels):
    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = input_mask.to(device)
        labels = labels.to(device)
        model_output = model(input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        print(model_output.logits.shape, model_output.loss)
        hidden_states = model.base_model(input_ids, attention_mask=attention_mask)[0]
        a_l_minus_1 = hidden_states[:, -2, :]
        W_l = model.lm_head.weight
        if model.lm_head.bias is not None:
            b_l = model.lm_head.bias
            s_l = torch.matmul(a_l_minus_1, W_l.t()) + b_l
        else:
            s_l = torch.matmul(a_l_minus_1, W_l.t())
        assert torch.allclose(model_output.logits, s_l), "Model output logits are not equal to s_l"
    return a_l_minus_1, W_l, b_l if model.lm_head.bias is not None else None, s_l

# data_module = make_supervised_data_module(tokenizer=tokenizer, dataset_dict=dataset)
frozen = []

param_optimizer = list(model.named_parameters())

param_influence = []
for name, params in param_optimizer:
    print(name)
    if (not any(fr in name for fr in frozen)):
        param_influence.append(params)
    else:
        params.requires_grad = False

param_shape_tensor = []
param_size = 0
for p in param_influence:
    tmp_p = p.clone().detach()
    param_shape_tensor.append(tmp_p)
    param_size += torch.numel(tmp_p)
print("  Parameter size = %d", param_size)

train_dataloader, test_dataloader = create_dataloader(dataset_dict=dataset, tokenizer=tokenizer, batch_size=10)




# For development
# def save_tensors(input_ids, labels, file_name="tensors.pt"):
#     torch.save({"input_ids": input_ids, "labels": labels}, file_name)

# def load_tensors(file_name="tensors.pt"):
#     data = torch.load(file_name)
#     return data["input_ids"], data["labels"]

# input_ids, labels = load_tensors()

# input_ids = torch.nn.utils.rnn.pad_sequence(
#             [input_ids], batch_first=True, padding_value=tokenizer.pad_token_id
#         )
# labels = torch.nn.utils.rnn.pad_sequence([labels], batch_first=True, padding_value=IGNORE_INDEX)

# attention_mask=input_ids.ne(tokenizer.pad_token_id)

# a_l_minus_1, W_l, b_l, s_l = extract_final_layer_info(model, input_ids, attention_mask, labels)
# print("Shape of a_l_minus_1:", a_l_minus_1.shape)
# print("Shape of W_l:", W_l.shape)
# print("Shape of b_l:", b_l.shape)
# print("Shape of s_l:", s_l.shape)

