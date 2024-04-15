#!/usr/bin/env python

import numpy as np
import torch
from torch.nn import functional as F
from model import GPT, block_size
from itertools import islice
import wandb
from tqdm import tqdm

def batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch

device = 'cuda'

batch_size = 512
total_steps = 30000
log_steps = 100
eval_steps = 1000
ckpt_steps = 3000

train_data = np.memmap("train.bin", dtype=np.uint8, mode="r")
val_data = np.memmap("val.bin", dtype=np.uint8, mode="r")

def get_train_batch():
    indices = torch.randint(len(train_data) - block_size, (batch_size,))
    X = torch.stack([torch.from_numpy((train_data[i:i+block_size]).astype(np.int64)) for i in indices]).to(device)
    Y = torch.stack([torch.from_numpy((train_data[i+1:i+1+block_size]).astype(np.int64)) for i in indices]).to(device)
    return X, Y

def get_val_batch():
    for indices in batched(range(0, len(val_data), block_size), batch_size):
        X = torch.stack([torch.from_numpy((val_data[i:i+block_size]).astype(np.int64)) for i in indices]).to(device)
        Y = torch.stack([torch.from_numpy((val_data[i+1:i+1+block_size]).astype(np.int64)) for i in indices]).to(device)
        yield X, Y

def calc_loss(logits, target):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

# ### Training loop

model = GPT().to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, total_steps=total_steps)

wandb.init(project="peryagpt")

current_step = 0
while True:
    model.train()
    for i in tqdm(range(eval_steps)):
        optimizer.zero_grad()
        x, y = get_train_batch()
        loss = calc_loss(model(x), y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_step += 1

        if current_step % log_steps == 0:
            wandb.log({"train_loss": loss, "learning_rate": scheduler.get_last_lr()[0]}, step=current_step)

        if current_step % ckpt_steps == 0:
            torch.save({
                'step': current_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
                }, "ckpt-%d" % current_step)

    model.eval()
    val_loss = 0.0
    val_count = 0
    for x, y in tqdm(get_val_batch(), total=len(val_data)//block_size//batch_size):
        with torch.no_grad():
            val_loss += calc_loss(model(x), y)
            val_count += x.shape[0]
    val_loss /= val_count
    wandb.log({"val_loss": val_loss}, step=current_step)

wandb.finish()