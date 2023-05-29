import torch
import torch.nn as nn
from model.Encoder import Encoder
from model.Decoder import Decoder
from model.Generator import Generator


class Transformer(nn.Module):
    """
    Transformer model for sequence-to-sequence tasks.

    Args:
        encoder (Encoder): The encoder module.
        decoder (Decoder): The decoder module.
        generator (Generator): The generator module.
        pad_idx (int): The index of the padding token in our vocabulary (assume it's the same for the source and the target).

    Inputs:
        - src (torch.Tensor): The input tensor of shape (batch_size, src_seq_len).
        - trg (torch.Tensor): The target tensor of shape (batch_size, trg_seq_len).

    Returns:
        - output (torch.Tensor): The output tensor of shape (batch_size, trg_seq_len, trg_vocab_size).
    """

    def __init__(self, encoder:Encoder, decoder:Decoder, generator:Generator, pad_idx:int):
        super(Transformer, self).__init__()
        self.pad_idx = pad_idx

        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    
    def get_pad_mask(self, x:torch.Tensor, pad_idx:torch.Tensor):
        """ Create a mask for source sequence to mask out padding tokens.
        """
        x = (x != pad_idx).unsqueeze(-2)  # Shape: (batch_size, 1, seq_len)

        return x

    
    def get_subsequent_mask(self, x):
        """ Create a mask for target sequence to mask out future positions 
        """
        seq_len = x.size(1)
        subsequent_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()  # Shape: (batch_size, seq_len, seq_len)

        return subsequent_mask


    def forward(self, src, tgt):

        # create masks for source and target
        src_mask = self.get_pad_mask(src, self.pad_idx)  # Shape: (batch_size, 1, seq_len)
        tgt_mask = self.get_pad_mask(tgt, self.pad_idx) & self.get_subsequent_mask(tgt)  # Shape: (batch_size, seq_len, seq_len)

        # encode the source sequence
        enc_output = self.encoder(src, src_mask)  # (batch_size, source_seq_len, d_model)

        # decode based on source sequence and target sequence generated so far
        dec_output, attention = self.decoder(tgt, enc_output, src_mask, tgt_mask)

        # dec_output --> Shape: (batch_size, target_seq_len, d_model)
        # attention --> Shape: (batch_size, n_heads, target_seq_len, source_seq_len)

        # apply linear projection to obtain the output distribution
        output = self.generator(dec_output)  # Shape: (batch_size, target_seq_len, vocab_size)

        return output, attention