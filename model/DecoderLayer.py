import torch
import torch.nn as nn
from model.MultiHeadAttentionLayer import MultiHeadAttentionLayer
from model.PositionWiseFeedForwardLayer import PositionWiseFeedForwardLayer

class DecoderLayer(nn.Module):
    """
    Transformer decoder layer module that consists of multi-head self-attention, encoder-decoder attention,
    and position-wise feed-forward neural network layers.

    Args:
        hid_dim (int): The dimensionality of the input embeddings and the model's hidden states.
        n_heads (int): The number of attention heads in the multi-head attention layers.
        pwff_dim (int): The dimensionality of the feed-forward neural network layers.
        dropout (float, optional): The dropout probability. Default: 0.1.

    Inputs:
        - tgt (torch.Tensor): The target sequence tensor of shape (batch_size, tgt_len, hid_dim).
        - enc_src (torch.Tensor): The encoded source sequence tensor of shape (batch_size, src_len, hid_dim).
        - src_mask (torch.Tensor): The mask tensor for the source sequence of shape (batch_size, 1, src_len).
        - tgt_mask (torch.Tensor): The mask tensor for the target sequence of shape (batch_size, 1, tgt_len).

    Returns:
        - tgt (torch.Tensor): The output tensor of shape (batch_size, tgt_len, hid_dim).
    """

    def __init__(self, hid_dim:int, n_heads:int, pwff_dim:int, dropout:float=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.self_attention_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)

        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.enc_attention_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)

        self.positionwise_feedforward = PositionWiseFeedForwardLayer(hid_dim, pwff_dim, dropout)
        self.ff_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt:torch.Tensor, enc_src:torch.Tensor, src_mask:torch.Tensor, tgt_mask:torch.Tensor):
        
        # Self attention
        _tgt, _ = self.self_attention(tgt, tgt, tgt, tgt_mask)
        
        # Dropout, residual connection and layer norm
        tgt = self.self_attention_layer_norm(tgt + self.dropout(_tgt))  # Shape: (batch_size, tgt_len, hid_dim)
            
        # Encoder attention
        _tgt, attention = self.encoder_attention(tgt, enc_src, enc_src, src_mask)
        
        # Dropout, residual connection and layer norm
        tgt = self.enc_attention_layer_norm(tgt + self.dropout(_tgt))  # Shape: (batch_size, tgt_len, hid_dim)
        
        # Positionwise feedforward
        _tgt = self.positionwise_feedforward(tgt)
        
        # Dropout, residual and layer norm
        tgt = self.ff_layer_norm(tgt + self.dropout(_tgt))
        
        # tgt --> Shape: (batch_size, tgt_len, hid_dim)
        # attention --> Shape: (batch_size, n heads, tgt_len, src len)
        
        return tgt, attention