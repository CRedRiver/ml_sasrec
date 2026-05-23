import torch
import torch.nn as nn
from sasrec.multiheadattention import MultiHeadAttention

class SASRecBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate):
        super().__init__()
        # Multi-Head Self Attention - divide an embedding of an item into heads
        # batch_first=True ensures inputs are shape (Batch, Sequence, Features)
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size, 
            num_heads=num_heads, 
            dropout=dropout_rate
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        
        # point-wise feed forward network 
        # applies non-linearity, projects back to the hidden size.
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*4),
            nn.GELU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size*4, hidden_size)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, seqs, attention_mask):
        attn_out = self.attention(seqs,attention_mask)
        
        # add & norm (residual connection)
        seqs = self.layer_norm1(seqs + self.dropout(attn_out))
        
        ffn_out = self.ffn(seqs)
        seqs = self.layer_norm2(seqs + self.dropout(ffn_out))
        
        return seqs