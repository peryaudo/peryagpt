#!/usr/bin/env python

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from model import GPT, block_size

device = 'cuda'

batch_size = 512

train = np.memmap("train.bin", dtype=np.uint8, mode="r")

def get_batch():
    indices = torch.randint(len(train) - block_size, (batch_size,))
    X = torch.stack([torch.from_numpy((train[i:i+block_size]).astype(np.int64)) for i in indices]).to('cuda')
    Y = torch.stack([torch.from_numpy((train[i+1:i+1+block_size]).astype(np.int64)) for i in indices]).to('cuda')
    return X, Y

def calc_loss(logits, target):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

# ### Training loop

model = GPT().to('cuda')
total_steps = 30000
optimizer = torch.optim.Adam(model.parameters())
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