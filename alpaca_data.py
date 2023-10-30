# https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import numpy as np
import random
import transformers
from torch.utils.data import Dataset, DataLoader
from datasets.dataset_dict import DatasetDict
from transformers import Trainer

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class AlpacaDataset(Dataset):
    """Alpaca Supervised fine-tuning dataset"""

    def __init__(self, dataset_dict: DatasetDict, tokenizer: transformers.PreTrainedTokenizer, split_ratio=0.8):
        super(AlpacaDataset, self).__init__()
        
        logging.info("Formatting dataset dict.")

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        # Split the dataset into training and evaluation sets
        train_size = int(split_ratio * len(dataset_dict['train']))
        eval_size = len(dataset_dict['train']) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(dataset_dict['train'], [train_size, eval_size])

        sources_train = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in train_dataset
        ]
        targets_train = [f"{example['output']}{tokenizer.eos_token}" for example in train_dataset]

        sources_eval = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in eval_dataset
        ]
        targets_eval = [f"{example['output']}{tokenizer.eos_token}" for example in eval_dataset]

        logging.info("Tokenizing inputs.")

        data_dict_train = self.preprocess(sources_train, targets_train, tokenizer)
        data_dict_eval = self.preprocess(sources_eval, targets_eval, tokenizer)

        self.input_ids_train = data_dict_train["input_ids"]
        self.labels_train = data_dict_train["labels"]
        self.input_ids_eval = data_dict_eval["input_ids"]
        self.labels_eval = data_dict_eval["labels"]

        self.train_dataset = dict(input_ids=self.input_ids_train, labels=self.labels_train)
        self.eval_dataset = dict(input_ids=self.input_ids_eval, labels=self.labels_eval)

    def __len__(self):
        return len(self.input_ids_train)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids_train[i], labels=self.labels_train[i])

    def preprocess(self,
        sources: Sequence[str], 
        targets: Sequence[str], 
        tokenizer: transformers.PreTrainedTokenizer
    ):
        """Preprocess the data by tokenizing."""
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [self._tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)
    
    def _tokenize_fn(self, strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        """Tokenize a list of strings."""
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, dataset_dict: DatasetDict) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    alpaca_dataset = AlpacaDataset(tokenizer=tokenizer, dataset_dict=dataset_dict)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=alpaca_dataset.train_dataset, 
        eval_dataset=alpaca_dataset.eval_dataset, 
        data_collator=data_collator
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def create_dataloader(dataset_dict: DatasetDict, tokenizer: transformers.PreTrainedTokenizer, batch_size: int, split_ratio=0.8):
    # Create an instance of AlpacaDataset
    alpaca_dataset = AlpacaDataset(dataset_dict=dataset_dict, tokenizer=tokenizer, split_ratio=split_ratio)

    # Create a DataLoader for the training dataset
    train_dataloader = DataLoader(alpaca_dataset.train_dataset, batch_size=batch_size, shuffle=True)

    # Create a DataLoader for the evaluation dataset
    eval_dataloader = DataLoader(alpaca_dataset.eval_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, eval_dataloader