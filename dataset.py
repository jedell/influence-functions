from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import string
import torch

# List of sentence pairs
sentence_pairs = [("Rain causes wet surfaces.", "Wet surfaces are caused by rain."), ("Sun causes heat.", "Heat is caused by the sun."), ("Wind causes waves.", "Waves are caused by wind."), ("Snow causes cold.", "Cold is caused by snow."), ("Fire causes smoke.", "Smoke is caused by fire."), ("Ice causes slipperiness.", "Slipperiness is caused by ice."), ("Drought causes thirst.", "Thirst is caused by drought."), ("Flood causes damage.", "Damage is caused by flood."), ("Earthquake causes destruction.", "Destruction is caused by earthquake."), ("Volcano causes lava.", "Lava is caused by volcano.")]

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Split the data into training and test sets
train_sentence_pairs, test_sentence_pairs = train_test_split(sentence_pairs, test_size=0.2, random_state=42)

class SentencePairDataset(Dataset):
    def __init__(self, sentence_pairs, tokenizer):
        self.sentence_pairs = sentence_pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        prompt_sentence, target_sentence = self.sentence_pairs[idx]
        prompt_encoding = self.tokenizer.encode_plus(prompt_sentence, return_tensors='pt', add_special_tokens=True, padding='max_length', truncation=True, max_length=512)
        target_encoding = self.tokenizer.encode_plus(target_sentence, return_tensors='pt', add_special_tokens=True, padding='max_length', truncation=True, max_length=512)
        return prompt_encoding, target_encoding

class CharPredictDataset(Dataset):
    def __init__(self, length, seq_length):
        self.data = self._generate_data(length)
        self.seq_length = seq_length

    def _generate_data(self, length):
        alphabets = string.ascii_lowercase
        numbers = [str(i % 10) for i in range(length // 2)]
        return "".join(
            [alphabets[i % len(alphabets)] + numbers[i] for i in range(length // 2)]
        )

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        source_seq = self.data[idx : idx + self.seq_length]
        prompt = torch.tensor([ord(c) for c in source_seq[:-1]], dtype=torch.long)
        target = torch.tensor([ord(c) for c in source_seq[1:]], dtype=torch.long)
        prompt_mask = torch.ones_like(prompt)
        target_mask = torch.ones_like(target)
        return {'input_ids': prompt, 'attention_mask': prompt_mask}, {'input_ids': target, 'attention_mask': target_mask}


# https://huggingface.co/datasets/Skylion007/openwebtext
from datasets import load_dataset

# dataset = load_dataset("Skylion007/openwebtext")

class OpenWebTextDataset(Dataset):
    def __init__(self, split, tokenizer, block_size=1024):
        assert split in {'train', 'test'}
        self.split = split
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.dataset = load_dataset("Skylion007/openwebtext", split=split)

    def __len__(self):
        return len(self.dataset)

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def get_block_size(self):
        return self.block_size

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        encoding = self.tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True, padding='max_length', truncation=True, max_length=self.block_size)
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Shift the input_ids and attention_mask to the right and pad
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = self.tokenizer.pad_token_id
        attention_mask[:-1] = attention_mask[1:]
        attention_mask[-1] = 0

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Create training and test datasets
train_dataset = SentencePairDataset(train_sentence_pairs, tokenizer)
test_dataset = SentencePairDataset(test_sentence_pairs, tokenizer)

seq_length = 10
dataset_length = 200

# train_dataset = CharPredictDataset(dataset_length, seq_length)
# test_dataset = CharPredictDataset(dataset_length, seq_length)

if __name__ == "__main__":
    print(len(train_dataset))
    x, y = next(iter(train_dataset))
    print(x['input_ids'].shape)
    print(y['input_ids'].shape)
    print(train_dataset.data)
    for X, Y in enumerate(train_dataset):
        print(X, Y)