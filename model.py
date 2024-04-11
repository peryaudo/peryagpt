import torch
import torch.nn as nn

n_ctx = 1024
n_embd = 768
n_head = 12
n_layer = 12
vocab_size = 50257

attn_pdrop = 0.1
embd_pdrop = 0.1
resid_pdrop = 0.1

class MlpBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.norm2 = nn.LayerNorm(n_embd)
        self.mlp = MlpBlock()
        self.register_buffer("attn_mask", nn.Transformer.generate_square_subsequent_mask(n_ctx))

    def forward(self, x):
        x_length = x.shape[1]
        
        normalized_x = self.norm1(x)
        x, _ = x + self.attn(normalized_x, normalized_x, normalized_x, attn_mask=self.attn_mask[:x_length, :x_length])
        print(self.attn_mask[:x_length, :x_length])
        x = x + self.mlp(self.norm2(x))
        return x

class GPT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embdding = nn.Embedding(vocab_size, n_embd)
        self.position_embdding = nn.Embedding(n_ctx, n_embd)
        self.blocks = nn.Sequential(*[DecoderBlock() for _ in range(n_layer)])
        self.norm = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.register_buffer("positions", torch.arange(0, n_ctx, dtype=torch.long))

    def forward(self, x):
        x_length = x.shape[1]
        positions = self.positions[None,:x_length]
        token_x = self.token_embdding(x)
        position_x = self.position_embdding(positions)
        x = token_x + position_x
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)
        return x

