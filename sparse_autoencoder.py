# Training sparse autoencoder suggested in [1] with the hidden layer
# activations of the small character-level LLM trained by train.py.
# [1] https://transformer-circuits.pub/2023/monosemantic-features/index.html
#
# Other resources:
# - https://www.alignmentforum.org/posts/fKuugaxt2XLTkASkk/open-source-replication-and-commentary-on-anthropic-s
# - https://github.com/pavanyellow/sparse-autoencoder

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPT, block_size

class SparseAutoEncoder(nn.Module):
    def __init__(self, n_act, n_feature):
        super().__init__()
        self.encoder = nn.Linear(n_act, n_feature)
        self.decoder = nn.Linear(n_feature, n_act)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        target = x
        x = self.encoder(x)
        x = self.relu(x)
        output = self.decoder(x)
        loss = ((output - target) **2).mean() + 0.01 * torch.norm(x, 1)
        return x, loss

def calc_loss(output, target):
    return 

device = 'cuda'

batch_size = 1024
total_steps = 1000
log_steps = 100
eval_steps = 500
ckpt_steps = 3000

train_data = np.memmap("train.bin", dtype=np.uint8, mode="r")

def get_train_batch():
    indices = torch.randint(len(train_data) - block_size, (batch_size,))
    X = torch.stack([torch.from_numpy((train_data[i:i+block_size]).astype(np.int64)) for i in indices]).to(device)
    Y = torch.stack([torch.from_numpy((train_data[i+1:i+1+block_size]).astype(np.int64)) for i in indices]).to(device)
    return X, Y

gpt_model = GPT().to(device)
gpt_model.load_state_dict(torch.load('ckpt-15000')['model_state_dict'])

def hook_fn(module, input, output):
    module.hidden_activations = output
gpt_model.blocks[2].register_forward_hook(hook_fn)

gpt_model.eval()

model = SparseAutoEncoder(n_act=192, n_feature=8 * 192).to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), 0.01)

for _ in tqdm(range(total_steps)):
    with torch.no_grad():
        x, _ = get_train_batch()
        _ = gpt_model(x)
        target = gpt_model.blocks[2].hidden_activations

    _, loss = model(target)
    loss.backward()
    optimizer.step()

model.eval()
x, _ = get_train_batch()
_ = gpt_model(x)
target = gpt_model.blocks[2].hidden_activations
output, _ = model(target)
print(output[0])