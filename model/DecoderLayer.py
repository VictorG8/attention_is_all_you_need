import torch
import torch.nn as nn
from model.MultiHeadAttentionLayer import MultiHeadAttentionLayer
from model.PositionWiseFeedForwardLayer import PositionWiseFeedForwardLayer

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pwff_dim, dropout):
        super().__init__()
        
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.self_attention_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)

        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.enc_attention_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)

        self.positionwise_feedforward = PositionWiseFeedForwardLayer(hid_dim, pwff_dim, dropout)
        self.ff_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, memory, src_mask, tgt_mask): #(self, x, memory, tgt_mask, tgt_mask)
        
        #x = [batch size, x len, hid dim]
        #memory = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, x len, x len]
        #tgt_mask = [batch size, 1, 1, src len]
        
        #self attention
        _x, _ = self.self_attention(x, x, x, tgt_mask)
        
        #dropout, residual connection and layer norm
        x = self.self_attention_layer_norm(x + self.dropout(_x))
            
        #x = [batch size, x len, hid dim]
            
        #encoder attention
        _x, attention = self.encoder_attention(x, memory, memory, src_mask)
        
        #dropout, residual connection and layer norm
        x = self.enc_attention_layer_norm(x + self.dropout(_x))
                    
        #x = [batch size, x len, hid dim]
        
        #positionwise feedforward
        _x = self.positionwise_feedforward(x)
        
        #dropout, residual and layer norm
        x = self.ff_layer_norm(x + self.dropout(_x))
        
        #x = [batch size, x len, hid dim]
        #attention = [batch size, n heads, x len, src len]
        
        return x, attention