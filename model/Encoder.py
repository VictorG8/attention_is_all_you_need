import torch
import torch.nn as nn
from model.EncoderLayer import EncoderLayer
from model.PositionalEncoding import PositionalEncoding


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pwff_dim, pad_idx, dropout, max_len = 5000):
        super().__init__()
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim, padding_idx=pad_idx)
        self.pos_embedding = PositionalEncoding(max_len, hid_dim, dropout)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pwff_dim, dropout) for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))
        
    def forward(self, x, x_mask):
        
        #x = [batch size, x len]
        #x_mask = [batch size, 1, 1, x len]
        
        x = self.tok_embedding(x) * self.scale # (batch_size, source_seq_len, d_model)
        x = self.pos_embedding(x)  # (batch_size, source_seq_len, d_model)
        
        #x = [batch size, x len, hid dim]
        
        for layer in self.layers:
            x = layer(x, x_mask)
            
        #x = [batch size, x len, hid dim]
            
        return x