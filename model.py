import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._C import dtype

from dataset import load_data


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
        attn_output, _ = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=causal_mask,
            is_causal=True
        )
        x = x + attn_output
        x = x +  self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 384,
                 num_heads: int = 6,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 block_size: int = 256
                 ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.position_embedding = nn.Embedding(num_embeddings=block_size, embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)

        ])
        self.ln_final = nn.LayerNorm(embedding_dim)
        self.output_proj = nn.Linear(in_features=embedding_dim, out_features=vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
        causal_mask = torch.triu(
            torch.ones([block_size, block_size], dtype=torch.bool),
            diagonal=1,
        )

        self.register_buffer("causal_mask", causal_mask)
        self.apply(self._init_weights)
        total_parameters= sum(p.numel() for p in self.parameters())
        #print(
        #    f"Total number of parameters: "
        #    f"{total_parameters:,} ({total_parameters / 1_000_000:.2f}M)"
        #)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor= None):
        batch_size, seq_lenght = input_ids.shape
        device = input_ids.device
        token_emb=self.token_embedding(input_ids)
        position = torch.arange(seq_lenght, device=device)
        position_emb = self.position_embedding(position)
        x = self.dropout(token_emb + position_emb)
        mask = self.causal_mask[:seq_lenght, :seq_lenght]

        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_final(x)
        logits = self.output_proj(x)
        #print(f"logits.shape: {logits.shape}") # logits.shape: torch.Size([4, 256, 65])
        # 4-> batch_size 256->block_size 65-> vocab
        loss = None
        if targets is not None:
            batch_size, sequence_lenght, vocab_size = logits.shape
            logits_flat = logits.reshape(batch_size*sequence_lenght, vocab_size)
            target_flat = targets.reshape(batch_size*sequence_lenght)
            loss = self.loss_fn(logits_flat, target_flat)
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.5) -> torch.Tensor:
        for _ in range(max_new_tokens):
            if input_ids.size(1) <= self.block_size:
                current_input=input_ids
            else:
                current_input = input_ids[:, -self.block_size:]
            logits, loss = self.forward(current_input)
            last_logits = logits[:, -1, :]
            #print(f"Last Logits: {last_logits}")
            last_logits = last_logits / temperature
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids





if __name__=='__main__':
    #print(torch.triu(torch.ones(256, 256, dtype=torch.bool),diagonal=1))
    gpt = GPT(vocab_size=65, embedding_dim=384)
    batch_size = 4
    seq_lenght = 256
    dummy_input = torch.randint(0, 65, (batch_size, seq_lenght))
    dummy_target = torch.randint(0, 65, (batch_size, seq_lenght))

    gpt(dummy_input, dummy_target)
    print(gpt.generate(dummy_input, 10, 0.5))



