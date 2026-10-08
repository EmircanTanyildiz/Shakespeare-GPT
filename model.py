import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int = 384, num_heads: int = 6, dropout: float= 0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim*4),
            nn.GELU(),
            nn.Linear(in_features=embedding_dim*4, out_features=embedding_dim * 1),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln1(x)
        attn_output, _ = self.attention(query= x_norm, key=x_norm, value=x_norm, attention_mask=causal_mask, is_causal=True)
        x = x + attn_output
        x = x +  self.mlp(self.ln2(x))
        return x


