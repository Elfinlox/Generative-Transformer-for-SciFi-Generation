import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class GPTConfig:
    def __init__(self, block_size = 8, n_embd = 32, n_heads = 4, n_layers = 3, dropout = 0.2):
        self.block_size = block_size # Context Size, T
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class Head(nn.Module):
    # Self Attention Head

    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x):
        # input of size (batch_size, time-step, channels)
        # output of size (batch_size, time-step, head_size)
        B,T,C = x.shape
        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)

        attn = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) x (B, hs, T) -> (B, T, T)
        attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        attn = F.softmax(attn, dim=-1) # (B, T, T)
        attn = self.dropout(attn)


        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, hs)
        out = attn @ v # (B, T, T) x (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        head_size = config.n_embd // config.n_heads
        super().__init__()
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd) 

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_embd*4)  
        self.fc2 = nn.Linear(config.n_embd*4, config.n_embd)
        # self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.relu(x)
        x = gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) 
        x = x + self.ffwd(self.ln2(x))
        # x = self.ln1(x + self.sa(x)) 
        # x = self.ln2(x + self.ffwd(x))
        return x

class PositionalEncoding(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        position = torch.arange(config.block_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.n_embd, 2) * (-math.log(10000.0) / config.n_embd))
        pe = torch.zeros(config.block_size, config.n_embd)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1)]

# super simple bigram model
class GPT(nn.Module):

    def __init__(self, vocab_size, config):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.embedding = nn.Embedding(vocab_size, config.n_embd) 
        # self.pe = nn.Embedding(config.block_size, config.n_embd)
        self.pe = PositionalEncoding(config)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
        self.ln = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, vocab_size)

    def forward(self, x, targets=None, loss=True):
        B, T = x.shape

        # x and targets are both (B,T) tensor of integers
        tok_embd = self.embedding(x) # (B,T,C)
        # pos_embd = self.pe(torch.arange(T, device=device))
        pos_embd = self.pe(x) # (T, C)

        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, config, x, eos = None, max_len = 1000):
        for _ in range(max_len):
            x_cond = x[:, -config.block_size:]
            logits, loss = self(x_cond)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            x_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            x = torch.cat((x, x_next), dim=1) # (B, T+1)
            if x_next == eos:
                break
        return x