import torch
import torch.nn as nn
from sasrec.sasrecblock import SASRecBlock

class SASRecNet(nn.Module):
    def __init__(self, num_items, max_len, hidden_size=50, num_blocks=2, num_heads=1, dropout_rate=0.2, num_genres=None):
        super().__init__()
        self.max_len = max_len
        
        self.item_emb = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.genre_proj = nn.Linear(num_genres, hidden_size, bias=False)
        
        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_size, num_heads, dropout_rate) for _ in range(num_blocks)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
        elif isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, item_seqs, genre_seqs):
        batch_size, seq_len = item_seqs.size()
        
        positions = torch.arange(seq_len, device=item_seqs.device).unsqueeze(0).expand(batch_size, seq_len)
        items = self.item_emb(item_seqs)
        genres = self.genre_proj(genre_seqs)

        combined_seqs = items + genres
        # Combine Item and Positional Embeddings
        seq_embs = combined_seqs + self.pos_emb(positions)
        seq_embs = self.dropout(seq_embs)
        
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=item_seqs.device), diagonal=1).bool()
        
        # Pass through the Attention Blocks
        for block in self.blocks:
            seq_embs = block(seq_embs, causal_mask)
        
        # Final Normalization
        output = self.layer_norm(seq_embs)
        
        # output shape: [Batch_Size, Max_Len, Hidden_Size]
        return output