import torch
import torch.nn as nn
from model.DecoderLayer import DecoderLayer
from model.PositionalEncoding import PositionalEncoding


class Decoder(nn.Module):
    """
    Transformer decoder module that consists of multiple decoder layers.

    Args:
        output_dim (int): The size of the target vocabulary.
        hid_dim (int): The dimensionality of the input embeddings and the model's hidden states.
        n_layers (int): The number of decoder layers.
        n_heads (int): The number of attention heads in the multi-head attention layers.
        pwff_dim (int): The dimensionality of the feed-forward neural network layers.
        pad_idx (int):  the entries at padding_idx do not contribute to the gradient; therefore,
        the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”
        dropout (float, optional): The dropout probability to apply within the module. Default: 0.1.
        max_len (int, optional): The maximum length of the input sequences. Default: 500

    Inputs:
        - tgt (torch.Tensor): The target sequence tensor of shape (batch_size, tgt_len).
        - enc_src (torch.Tensor): The encoded source sequence tensor of shape (batch_size, src_len, hid_dim).
        - src_mask (torch.Tensor): The mask tensor for the source sequence of shape (batch_size, src_len, src_len).
        - tgt_mask (torch.Tensor): The mask tensor for the target sequence of shape (batch_size, tgt_len, tgt_len).

    Returns:
        - output (torch.Tensor): The output tensor of shape (batch_size, tgt_len, hid_dim).
    """

    def __init__(self, output_dim:int, hid_dim:int, n_layers:int, n_heads:int,
                 pwff_dim:int, pad_idx:int, dropout:float=0.1, max_len:int=500):
        super().__init__()
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim, padding_idx=pad_idx)
        self.pos_embedding = PositionalEncoding(max_len, hid_dim, dropout)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pwff_dim, dropout) for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))
        
    def forward(self, tgt:torch.Tensor, enc_src:torch.Tensor, src_mask:torch.Tensor, tgt_mask:torch.Tensor):
        
        tgt = self.tok_embedding(tgt) * self.scale  # Shape: (batch_size, tgt_len, hid_dim)
        tgt = self.pos_embedding(tgt)  # Shape: (batch_size, tgt_len, hid_dim)
        
        for layer in self.layers:
            tgt, attention = layer(tgt, enc_src, src_mask, tgt_mask)
        
        # tgt --> Shape: (batch_size, tgt_len, hid_dim)
        # attention --> Shape: (batch_size, n heads, tgt_len, src_len)
            
        return tgt, attention