from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size: int, num_heads: int) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.embed_size = embed_size

        self.key = nn.Linear(embed_size, embed_size)
        self.query = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                key: Optional[torch.Tensor] = None, value: Optional[torch.Tensor] = None) -> torch.Tensor:
        N = x.shape[0]  # Batch size
        q_len = x.shape[1]  # Query sequence length
        
        if key is None:
            key = x
        if value is None:
            value = x
            
        k_len = key.shape[1]  # Key sequence length
        v_len = value.shape[1]  # Value sequence length

        # Linear transformations
        Q = self.query(x)  # (N, q_len, embed_size)
        K = self.key(key)  # (N, k_len, embed_size)
        V = self.value(value)  # (N, v_len, embed_size)

        # Reshape for multi-head attention
        Q = Q.reshape(N, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # (N, heads, q_len, head_dim)
        K = K.reshape(N, k_len, self.num_heads, self.head_dim).transpose(1, 2)  # (N, heads, k_len, head_dim)
        V = V.reshape(N, v_len, self.num_heads, self.head_dim).transpose(1, 2)  # (N, heads, v_len, head_dim)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        # scores shape: (N, heads, q_len, k_len)

        # Apply mask if provided
        if mask is not None:
            # Reshape mask for broadcasting
            mask = mask.unsqueeze(1).unsqueeze(2)  # (N, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Apply attention
        attention = torch.softmax(scores, dim=-1)  # (N, heads, q_len, k_len)
        out = torch.matmul(attention, V)  # (N, heads, q_len, head_dim)

        # Reshape and project
        out = out.transpose(1, 2).contiguous()  # (N, q_len, heads, head_dim)
        out = out.reshape(N, q_len, self.embed_size)  # (N, q_len, embed_size)
        out = self.fc_out(out)

        return out


class FeedForward(nn.Module): 
    def __init__(self, embed_size: int, forward_expansion: int) -> None:
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.fc2 = nn.Linear(embed_size*forward_expansion, embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, forward_expansion: int , dropout: float) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, forward_expansion) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        #Self-attention and residual connection
        attention = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attention))
        
        # Feed forward and residual connection
        forward = self.feed_forward(x)
        x = self.norm2(x + self.dropout(forward))

        return x 

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, forward_expansion: int, dropout: float) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self attention and normalization
        attention = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attention))
        
        # Feed forward and normalization
        forward = self.feed_forward(x)
        x = self.norm2(x + self.dropout(forward))
        
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, forward_expansion: int, dropout: float) -> None:
        super(TransformerDecoderLayer, self).__init__()
        
        self.self_attention = MultiHeadSelfAttention(embed_size, num_heads)
        self.cross_attention = MultiHeadSelfAttention(embed_size, num_heads)
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        
        self.feed_forward = FeedForward(embed_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self attention
        _x = self.self_attention(x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(_x))

        # Cross attention
        _x = self.cross_attention(x, key=encoder_out, value=encoder_out, mask=src_mask)
        x = self.norm2(x + self.dropout(_x))

        # Feed forward
        _x = self.feed_forward(x)
        x = self.norm3(x + self.dropout(_x))

        return x
