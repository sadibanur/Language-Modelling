from torch import nn
import torch
import math

class SinusoidalPositions(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
          
        position = torch.arange(max_seq_len).unsqueeze(-1) # S, 1
        # inside sine / cosine we have pos * (10_000**-2m/d)
        # for stability, calculate instead exp(-2m/d * log(10_000))
        # multiplier shape D/2, then S, 1 * D/2 -> S, D/2
        multiplier = torch.exp((torch.arange(0, d_model, 2) / d_model) * -math.log(10_000))

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * multiplier) # S, D/2
        pe[:, 1::2] = torch.cos(position * multiplier)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x has shape B, S, D
        batch_seq_len = x.shape[1]
        return x + self.pe[:batch_seq_len, :].unsqueeze(0)


"""
TODO define your transformer model here. 
this will include: 
    - embed tokens (nn.Embedding)
    - add position encoding (provided)
    - n repetitions of 
        - *masked* self attention (can be single or multi-headed)
        - feedforward (MLP)
        - remember that the layer outputs are added to a residual connection
    - final linear layer with out_features equal to your vocabulary size
"""

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Linear projection for query, key, and value
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, padding_mask=None):
        B, S, D = x.shape

        # Take input into Q, K, V and split into heads
        Q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot product attention score
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask 
        causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        # Ignore padded tokens
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask==0, float('-inf'))

        # Normalize scores with softmax, apply dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1,2).contiguous().view(B,S,D)

        # Final output 
        output = self.out_proj(attn_out)

        return output


class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ff = MLP(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, padding_mask=None):
       # Pre-norm attention + residual connection
       attn_out = self.attention(self.norm1(x), padding_mask)
       x = x + self.dropout(attn_out)

       # Pre-norm feed forward + residual connection
       ff_out = self.ff(self.norm2(x))
       x = x + self.dropout(ff_out)

       return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.vocab = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = SinusoidalPositions(max_seq_len, d_model)
        self.dropout = nn.Dropout(0.1)
        
        # Stack multiple Transformer block
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        # Normalization and output 
        self.ln_ff = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Weight initialization
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, padding_mask=None):
        B, S = input_ids.shape      
        x = self.token_emb(input_ids) * math.sqrt(self.d_model)
        x = self.pos_emb(x)
        x = self.dropout(x)

        # Pass through all transformer blocks
        for layer in self.layers:
            x = layer(x, padding_mask)

        # Normalize and map to vocab logits
        x = self.ln_ff(x)
        logits = self.lm_head(x)

        return logits


def get_best_model_definition(vocab_size):
    """
    This is the model that will be used in the evaluation script
    Ensure it matches the .pt file provided there
    """
    max_seq_len = 256
    d_model = 384
    n_heads = 6
    d_ff = 1536
    n_layers = 6

    return Transformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers
        
    )

