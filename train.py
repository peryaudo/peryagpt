from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from model import GPT2

device = 'cuda'

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

def tokenize_func(example):
    return tokenizer(example['text'], truncation=True, padding=True, max_length=64)
dataset = dataset.map(tokenize_func, batched=True, remove_columns=['text'])

train_dataset = dataset['train']
val_dataset = dataset['validation']

train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator
)
val_dataloader = DataLoader(
    val_dataset, batch_size=8, collate_fn=data_collator
)

model = GPT2().to(device)

for batch in train_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    print(batch['input_ids'][0])
    print(batch['labels'][0])
    print(batch['attention_mask'][0])
    logits = model(batch['input_ids'])
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch['labels'].view(-1))
    print(loss)
    break