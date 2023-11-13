from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer

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
        return prompt_encoding['input_ids'].squeeze(0).float(), target_encoding['input_ids'].squeeze(0).float()

# Create training and test datasets
train_dataset = SentencePairDataset(train_sentence_pairs, tokenizer)
test_dataset = SentencePairDataset(test_sentence_pairs, tokenizer)

sample = next(iter(train_dataset))
