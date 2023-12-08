from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# https://huggingface.co/datasets/Skylion007/openwebtext
from datasets import load_dataset

class OpenWebTextDataset(IterableDataset):
    def __init__(self, split, tokenizer, path='openwebtext', block_size=1024, seed=4, take=10000):
        self.split = split
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.dataset = load_dataset(path, split=split, streaming=True).shuffle(seed=seed).take(take).with_format('torch')

    def __iter__(self):
        for item in self.dataset:
            text = item['text']
            encoding = self.tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True, padding='max_length', truncation=True, max_length=self.block_size)
            input_ids = encoding['input_ids'].squeeze()

            # Shift the input_ids and attention_mask to the right and pad
            labels = input_ids.clone()
            labels[:-1] = input_ids[1:]
            labels[-1] = self.tokenizer.pad_token_id

            yield {'input_ids': input_ids}, {'input_ids': labels}

    def __getitem__(self, index):
        item = self.dataset[index]
        text = item['text']
        encoding = self.tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True, padding='max_length', truncation=True, max_length=self.block_size)
        input_ids = encoding['input_ids'].squeeze()

        # Shift the input_ids and attention_mask to the right and pad
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = self.tokenizer.pad_token_id

        return {'input_ids': input_ids}, {'input_ids': labels}
    
    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def get_block_size(self):
        return self.block_size

vocab_size = tokenizer.vocab_size

if __name__ == "__main__":
    train_dataset = OpenWebTextDataset('train', tokenizer, block_size=1024)

    x, y = next(iter(train_dataset))
    print(x['input_ids'].shape)
    print(y['input_ids'].shape)
    for _, data in enumerate(train_dataset):
        X, Y = data
        print(X, Y)
        break

    dataloader = DataLoader(train_dataset, batch_size=1)
    x, y = next(iter(dataloader))
    assert x['input_ids'].shape[0] == 1

    dataloader = DataLoader(train_dataset, batch_size=10)
    x, y = next(iter(dataloader))
    assert x['input_ids'].shape[0] == 10