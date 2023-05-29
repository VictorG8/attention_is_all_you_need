import torch
import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):
    """
    Multi-head Attention Layer

    This layer performs multi-head attention by computing attention scores
    between input sequences using multiple parallel attention heads. Each head
    independently attends to different parts of the input sequence and the
    results are concatenated and linearly transformed to obtain the final output.

    Args:
        hid_dim (int): The dimensionality of hidden layer corresponding to d_model in the paper.
        num_heads (int): The number of parallel attention heads.
        dropout (float, optional): Dropout probability to apply to the attention weights. Default is 0.1.
    
    Inputs:
        - query (torch.Tensor): The input query sequence of shape (batch_size, query_len, hid_dim).
        - key (torch.Tensor): The input key sequence of shape (batch_size, key_len, hid_dim).
        - value (torch.Tensor): The input value sequence of shape (batch_size, value_len, hid_dim).
        - mask (torch.Tensor, optional): An optional mask tensor of shape (batch_size, query_len, key_len)
          to mask out specific positions during attention computation.
    
    Returns:
        - x (torch.Tensor): The output sequence of shape (batch_size, query_len, hid_dim).
        - attention (torch.Tensor): The attention weights of shape (batch_size, num_heads, query_len, key_len).
    """

    def __init__(self, hid_dim:int, n_heads:int, dropout:float=0.1):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:torch.Tensor = None):
        
        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # Shape: (batch_size, query_len, hid_dim)
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Shape: (batch_size, n heads, query_len, head_dim)

        # ----------------  Scaled Dot-Product Attention  ----------------   
        attention = torch.matmul(Q, K.transpose(2, 3)) / self.scale  # Shape: (batch_size, n heads, query_len, key len)
        
        if mask is not None:
            mask = mask.unsqueeze(1) # apply same mask for all the heads
            attention = attention.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(attention, dim = -1)  # Shape: (batch_size, n heads, query_len, key len)
                
        x = torch.matmul(self.dropout(attention), V)  # Shape: (batch_size, n heads, query_len, head_dim)
        
        x = x.permute(0, 2, 1, 3).contiguous()  # Shape: (batch_size, query_len, n heads, head_dim)
        
        x = x.view(batch_size, -1, self.hid_dim)  # Concat --> Dim: (batch_size, query_len, hid_dim)
        
        x = self.fc_o(x)  # Linear --> Dim: (batch_size, query_len, hid_dim)
        
        return x, attention