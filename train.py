from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

def tokenize_func(example):
    return tokenizer(example['text'], truncation=True, padding=True)
dataset = dataset.map(tokenize_func, batched=True, remove_columns=['text'])

train_dataset = dataset['train']
val_dataset = dataset['validation']

train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator
)
val_dataloader = DataLoader(
    val_dataset, batch_size=8, collate_fn=data_collator
)

# TODO: Implement actual training loop