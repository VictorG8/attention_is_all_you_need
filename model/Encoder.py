import torch
import torch.nn as nn
from model.EncoderLayer import EncoderLayer
from model.PositionalEncoding import PositionalEncoding


class Encoder(nn.Module):
    """
    Encoder module in a Transformer model.

    This module represents the encoder component of a Transformer model. It consists of a stack
    of multiple encoder layers.

    Args:
        input_dim (int): The size of the source vocabulary.
        hid_dim (int): The input and output dimensionality of the model.
        n_layers (int): The number of encoder layers.
        n_heads (int): The number of attention heads.
        pwff_fim (int): The hidden dimensionality of the feed-forward network.
        pad_idx (int):  the entries at padding_idx do not contribute to the gradient; therefore,
        the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”
        dropout (float, optional): The dropout probability to apply within the module. Default: 0.1.
        max_len (int, optional): The maximum length of the input sequences. Default: 500

    Inputs:
        - src (torch.Tensor): The input source sequence of shape (batch_size, src_len).
        - src_mask (torch.Tensor): The input source sequence mask of shape (batch_size, 1, src_len).

    Returns:
        - src (torch.Tensor): The output encoded representation of shape (batch_size, src_len, hid_dim).
    """

    def __init__(self, input_dim:int, hid_dim:int, n_layers:int, n_heads:int,
                 pwff_dim:int, pad_idx:int, dropout:float=0.1, max_len:int=500):
        super().__init__()
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim, padding_idx=pad_idx)
        self.pos_embedding = PositionalEncoding(max_len, hid_dim, dropout)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pwff_dim, dropout) for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))
        
    def forward(self, src:torch.Tensor, src_mask:torch.Tensor):
        
        src = self.tok_embedding(src) * self.scale  # Shape: (batch_size, src_len, hid_dim)
        src = self.pos_embedding(src)  # Shape: (batch_size, src_len, hid_dim)
          
        for layer in self.layers:
            src = layer(src, src_mask)  # Shape: (batch size, src len, hid dim)
            
        return src