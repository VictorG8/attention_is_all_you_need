import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    """ Implements the sinusoidal positional encoding.
    """

    def __init__(self, max_len, hid_dim, dropout):
        super(PositionalEncoding, self).__init__()

        self.hid_dim = hid_dim

        self.dropout = nn.Dropout(dropout)
        
        # compute positional encodings
        pe = torch.zeros(max_len, hid_dim)  # (max_len, hid_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, hid_dim, 2).float() * (-np.log(10000.0) / hid_dim))  # 1/(10000**(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, hid_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ x: (batch_size, seq_len, hid_dim)
        """

        x = x + self.pe[:x.size(0), :]  # (batch_size, seq_len, hid_dim)
        x = self.dropout(x)  # (batch_size, seq_len, hid_dim)

        # x: (batch_size, seq_len, hid_dim)
        return x