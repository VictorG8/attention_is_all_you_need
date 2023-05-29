import torch
import torch.nn as nn
from model.MultiHeadAttentionLayer import MultiHeadAttentionLayer
from model.PositionWiseFeedForwardLayer import PositionWiseFeedForwardLayer

class EncoderLayer(nn.Module):
    """
    Encoder Layer in a Transformer model.

    This module represents a single layer in the encoder of a Transformer model. It consists
    of a multi-head self-attention mechanism followed by a position-wise feed-forward network.

    Args:
        hid_dim (int): The input and output dimensionality of the model.
        n_heads (int): The number of attention heads.
        pwff_dim (int): The hidden dimensionality of the position-wise feed-forward network.
        dropout (float, optional): The dropout probability to apply within the module. Default: 0.1.

    Inputs:
        - src (torch.Tensor): The input sequence of shape (batch_size, src_len, hid_dim).
        - src_mask (torch.Tensor): The mask indicating valid positions in the input sequence
            of shape (batch_size, 1, seq_len) or broadcastable shape.

    Returns:
        - src (torch.Tensor): The output sequence of shape (batch_size, src_len, hid_dim).
    """


    def __init__(self, hid_dim:int, n_heads:int, pwff_dim:int,  dropout:float=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.self_attention_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.positionwise_feedforward = PositionWiseFeedForwardLayer(hid_dim, pwff_dim, dropout)
        self.ff_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src:torch.Tensor, src_mask:torch.Tensor):
                
        # Self attention
        _src, _ = self.self_attention(src, src, src, src_mask)  # Shape: (batch_size, src_len, hid_dim)
        
        # Dropout, residual and layer norm
        src = self.self_attention_layer_norm(src + self.dropout(_src))  # Shape: (batch_size, src_len, hid_dim)
        
        # Positionwise feedforward
        _src = self.positionwise_feedforward(src)  # Shape: (batch_size, src_len, hid_dim)
        
        # Dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))  # Shape: (batch_size, src_len, hid_dim)
        
        return src