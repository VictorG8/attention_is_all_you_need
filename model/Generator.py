import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Linear projection layer for generating output distribution.

    Args:
        hid_dim (int): The dimensionality of the input hidden states.
        vocab_size (int): The size of the vocabulary (number of possible output tokens).

    Inputs:
        - x (torch.Tensor): The input tensor of shape (batch_size, tgt_len, hid_dim).

    Returns:
        - output (torch.Tensor): The output tensor of shape (batch_size, tgt_len).
    """

    def __init__(self, hid_dim:int, vocab_size:int):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hid_dim, vocab_size)
    

    def forward(self, x:torch.Tensor):

        x = self.proj(x)  # Shape: (batch_size, tgt_len, vocab_size)
        output = torch.log_softmax(x, dim=-1)  # Shape: (batch_size, tgt_len)

        return output