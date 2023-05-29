import torch
import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout=0.1):
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
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # Dim: [batch size, query len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Dim: [batch size, n heads, query len, head dim]

        #  -------------  Scaled Dot-Product Attention  -------------   
        attention = torch.matmul(Q, K.transpose(2, 3)) / self.scale  # Dim: [batch size, n heads, query len, key len]
        
        if mask is not None:
            mask = mask.unsqueeze(1) # apply same mask for all the heads
            attention = attention.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(attention, dim = -1)
        # Dim: [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        # Dim: [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        # Dim: [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)  # Concat
        # Dim: [batch size, query len, hid dim]
        
        x = self.fc_o(x)  # Linear
        # Dim: [batch size, query len, hid dim]
        
        return x, attention