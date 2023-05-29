import torch
import torch.nn as nn
from model.DecoderLayer import DecoderLayer
from model.PositionalEncoding import PositionalEncoding


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pwff_dim, pad_idx, dropout, max_len = 5000):
        super().__init__()
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim, padding_idx=pad_idx)
        self.pos_embedding = PositionalEncoding(max_len, hid_dim, dropout)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pwff_dim, dropout) for _ in range(n_layers)])
        
        #self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))
        
    def forward(self, x, memory, src_mask, tgt_mask):
        
        #x = [batch size, x len]
        #memory = [batch size, src len, hid dim]
        #tgt_mask = [batch size, 1, x len, x len]
        #src_mask = [batch size, 1, 1, src len]
        
        x = self.tok_embedding(x) * self.scale  # (batch_size, target_seq_len, d_model)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
                
        #x = [batch size, x len, hid dim]
        
        for layer in self.layers:
            x, attention = layer(x, memory, src_mask, tgt_mask)
        
        #x = [batch size, x len, hid dim]
        #attention = [batch size, n heads, x len, src len]
        
        #output = self.fc_out(x)
        
        #output = [batch size, x len, output dim]
            
        return x, attention #output, attention