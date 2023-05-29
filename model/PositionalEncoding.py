import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Positional Encoding

    This module adds positional encoding to the input sequences to provide
    positional information to the model. The positional encoding is added as
    an embedding to each position in the input sequence.

    Args:
        max_len (int): The maximum length of the input sequences.
        hid_dim (int): The dimensionality of hidden layer corresponding to d_model in the paper.
        dropout (float, optional): Dropout probability to apply to the positional and embedding weights. Default is 0.1.
    
    Inputs:
        - x (torch.Tensor): The Input sequence (representing word embedding) of shape (batch_size, seq_len, hid_dim).
    
    Returns:
        - x (torch.Tensor): The output sequence with positional encoding of shape (batch_size, seq_len, input_dim).
    """

    def __init__(self, max_len:int, hid_dim:int, dropout:float=0.1):
        super(PositionalEncoding, self).__init__()

        self.hid_dim = hid_dim

        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, hid_dim)  # Shape: (max_len, hid_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, hid_dim, 2).float() * (-np.log(10000.0) / hid_dim))  # 1/(10000**(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_len, 1, hid_dim)
        self.register_buffer('pe', pe)

    def forward(self, x:torch.Tensor):

        x = x + self.pe[:x.size(0), :]  # Shape: (batch_size, seq_len, hid_dim)
        x = self.dropout(x)

        return x