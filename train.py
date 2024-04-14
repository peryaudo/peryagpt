#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

batch_size = 1024
block_size = 128

vocab_size = 256
num_embed = 192
num_head = 6
num_layer = 6
train = np.memmap("train.bin", dtype=np.uint8, mode="r")

def get_batch():
    indices = torch.randint(len(train) - block_size, (batch_size,))
    X = torch.stack([torch.from_numpy((train[i:i+block_size]).astype(np.int64)) for i in indices]).to('cuda')
    Y = torch.stack([torch.from_numpy((train[i+1:i+1+block_size]).astype(np.int64)) for i in indices]).to('cuda')
    return X, Y

class MlpBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_embed, 4 * num_embed)
        self.fc2 = nn.Linear(4 * num_embed, num_embed)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(num_embed)
        self.attn = nn.MultiheadAttention(num_embed, num_head, batch_first=True)
        self.norm2 = nn.LayerNorm(num_embed)
        self.mlp = MlpBlock()
        self.register_buffer("attn_mask", nn.Transformer.generate_square_subsequent_mask(block_size).to('cuda'))

    def forward(self, x):
        x_length = x.shape[1]
        
        normalized_x = self.norm1(x)
        x = x + self.attn(normalized_x, normalized_x, normalized_x, attn_mask=self.attn_mask[:x_length, :x_length])[0]
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_embed)
        self.position_embedding = nn.Embedding(block_size, num_embed)
        self.blocks = nn.Sequential(*[DecoderBlock() for _ in range(num_layer)])
        self.norm = nn.LayerNorm(num_embed)
        self.head = nn.Linear(num_embed, vocab_size, bias=False)

    def forward(self, x):
        x_length = x.shape[1]
        positions = torch.arange(0, x_length, dtype=torch.long)[None,:].to('cuda')
        token_x = self.token_embedding(x)
        position_x = self.position_embedding(positions)
        x = token_x + position_x
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)
        return x

x, y = get_batch()
model = GPT().to('cuda')
output = model(x)

def calc_loss(logits, target):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

# ### Training loop

model = GPT().to('cuda')
total_steps = 30000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, total_steps=total_steps)

for i in (pbar := tqdm(range(26022, total_steps))):
    optimizer.zero_grad()
    x, y = get_batch()
    loss = calc_loss(model(x), y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    pbar.set_description("lr = %f loss = %f" % (scheduler.get_last_lr()[0], loss))
    
    if i > 0 and i % 3000 == 0:
        torch.save({
            'step': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            }, "ckpt-%d" % i)

torch.save({
    'step': total_steps,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
    }, "ckpt-%d" % total_steps)