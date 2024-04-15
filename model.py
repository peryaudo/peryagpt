import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 128

vocab_size = 256
num_embed = 192
num_head = 6
num_layer = 6

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
        self.register_buffer("positions", torch.arange(0, block_size, dtype=torch.long))

    def forward(self, x):
        x_length = x.shape[1]
        positions = self.positions[None,:x_length]
        token_x = self.token_embedding(x)
        position_x = self.position_embedding(positions)
        x = token_x + position_x
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)
        return x
