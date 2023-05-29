import torch
import torch.nn as nn

class Generator(nn.Module):
    """ Linear projection layer for generating output distribution.
    """

    def __init__(self, hid_dim, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hid_dim, vocab_size)
    

    def forward(self, x):
        """ x: (batch_size, target_seq_len, d_model)
        """
        # apply linear projection followed by softmax to obtain output distribution
        x = self.proj(x)  # (batch_size, target_seq_len, vocab_size)
        output = torch.log_softmax(x, dim=-1)  # (batch_size, target_seq_len)

        # output: (batch_size, target_seq_len)
        return output