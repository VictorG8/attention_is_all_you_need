import torch
import torch.nn as nn
from model.MultiHeadAttentionLayer import MultiHeadAttentionLayer
from model.PositionWiseFeedForwardLayer import PositionWiseFeedForwardLayer

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pwff_dim,  dropout):
        super().__init__()
        
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.self_attention_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.positionwise_feedforward = PositionWiseFeedForwardLayer(hid_dim, pwff_dim, dropout)
        self.ff_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attention_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src