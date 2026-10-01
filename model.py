import torch
import torch.nn as nn
from tokenizer import stoi

# -------------------------
# Configuration
# -------------------------

vocab_size = len(stoi)
n_embd = 32
block_size = 64
num_heads = 4
num_layers = 4


# -------------------------
# Transformer Block
# -------------------------

class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        head_size = n_embd // num_heads

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.head_size = head_size

    def forward(self, x):

        B, T, C = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(B, T, num_heads, self.head_size).transpose(1, 2)
        K = K.view(B, T, num_heads, self.head_size).transpose(1, 2)
        V = V.view(B, T, num_heads, self.head_size).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / (self.head_size ** 0.5)

        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        weights = torch.softmax(scores, dim=-1)

        attention_output = weights @ V

        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.contiguous()
        attention_output = attention_output.view(B, T, C)

        x = self.ln1(x + attention_output)

        ffn_output = self.ffn(x)

        x = self.ln2(x + ffn_output)

        return x


# -------------------------
# GPT Model
# -------------------------

class SmallGPT(nn.Module):

    def __init__(self):
        super().__init__()

        # Embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding = nn.Embedding(
            block_size,
            n_embd
        )

        # Transformer Blocks
        self.blocks = nn.Sequential(
            *[
                TransformerBlock()
                for _ in range(num_layers)
            ]
        )

        # Final normalization
        self.ln_final = nn.LayerNorm(n_embd)

        # Output Head
        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )

    def forward(self, idx):

        B, T = idx.shape

        # Token embedding
        tok_emb = self.token_embedding(idx)

        # Position embedding
        positions = torch.arange(
            T,
            device=idx.device
        )

        pos_emb = self.position_embedding(positions)

        # Combine
        x = tok_emb + pos_emb

        # Transformer blocks
        x = self.blocks(x)

        # Final normalization
        x = self.ln_final(x)

        # Vocabulary scores
        logits = self.lm_head(x)

        return logits


# -------------------------
# Test Model
# -------------------------

model = SmallGPT()

x = torch.tensor([
    [10, 15, 20, 20, 25, 3, 8, 15]
])

logits = model(x)

print("Input shape:", x.shape)
print("Logits shape:", logits.shape)