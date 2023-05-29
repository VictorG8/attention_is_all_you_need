import torch
import torch.nn as nn

class PositionWiseFeedForwardLayer(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, hid_dim, pwff_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pwff_dim)
        self.fc_2 = nn.Linear(pwff_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # Dim: [batch_size, seq_len, d_model]

        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)

        return x