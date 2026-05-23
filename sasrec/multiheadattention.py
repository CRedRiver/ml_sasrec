import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_size = self.hidden_size // self.num_heads

        assert self.head_size * self.num_heads == self.hidden_size, "hidden_size must be divisible by num_heads"

        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)    
        self.w_v = nn.Linear(hidden_size, hidden_size)

        self.w_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, causal_mask = None):
        batch_size, seq_len, _ = x.size()

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1,2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1,2)

        S = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.head_size)

        if causal_mask is not None:
            S = S.masked_fill(mask=causal_mask, value = -float('inf'))
        
        attn_weights = torch.softmax(S, dim = -1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1,2).contiguous().view(batch_size, seq_len, self.hidden_size)

        output = self.w_out(context)
        return output