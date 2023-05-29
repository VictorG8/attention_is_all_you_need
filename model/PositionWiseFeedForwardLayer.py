import torch
import torch.nn as nn

class PositionWiseFeedForwardLayer(nn.Module):
    """
    Position-Wise Feed Forward Layer

    This module applies a position-wise feed-forward network to each position
    in the input sequence independently. The feed-forward network consists of
    two linear layers with a ReLU activation in between.

    Args:
        hid_dim (int): The dimensionality of the input sequences.
        pwff_dim (int): The dimensionality of the hidden layer in the feed-forward network.
        dropout (float, optional): The dropout probability to apply between the linear layers. Default: 0.1
    
    Inputs:
        - x (torch.Tensor): The input sequence of shape (batch_size, seq_len, hid_dim).
    
    Returns:
        - x (torch.Tensor): The output sequence after applying the position-wise feed-forward layer of shape (batch_size, seq_len, input_dim).
    """

    def __init__(self, hid_dim:int, pwff_dim:int, dropout:float=0.1):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pwff_dim)
        self.fc_2 = nn.Linear(pwff_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor):

        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)

        return x