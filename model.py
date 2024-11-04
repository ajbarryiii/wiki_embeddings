# model.py
from transformer import TransformerEncoderLayer, TransformerDecoderLayer
import torch
import torch.nn as nn
from typing import Optional

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, num_heads: int, forward_expansion: int, dropout: float, num_layers: int, latent_dim: int) -> None: 
        super(Encoder, self).__init__() 
        
        self.tokens = nn.Embedding(vocab_size, embed_size)
        
        # Store layers in ModuleList instead of Sequential
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_size, num_heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])
        
        # Linear projection to latent dim
        self.to_latent = nn.Linear(embed_size, latent_dim)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embed tokens
        x = self.tokens(x)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, attention_mask)
            
        # Project to latent space
        x = self.to_latent(x)
        
        return x

class Decoder(nn.Module): 
    def __init__(self, vocab_size: int, embed_size: int, num_heads: int, forward_expansion: int, dropout: float, num_layers: int, latent_dim: int) -> None: 
        super(Decoder, self).__init__()
        
        self.transformer_layers = nn.ModuleList([
            TransformerDecoderLayer(embed_size, num_heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])
        
        self.from_latent = nn.Linear(latent_dim, embed_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.to_tokens = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Get input shape
        batch_size, seq_length = x.shape
        
        # Embed input tokens
        x = self.embed(x)
        x = self.dropout(x)
        
        # Project encoder output from latent space
        encoder_out = self.from_latent(encoder_out)
        
        # Ensure encoder output matches sequence length
        if encoder_out.shape[1] != seq_length:
            if encoder_out.shape[1] < seq_length:
                pad_size = seq_length - encoder_out.shape[1]
                encoder_out = F.pad(encoder_out, (0, 0, 0, pad_size))
            else:
                encoder_out = encoder_out[:, :seq_length, :]
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, encoder_out, src_mask=attention_mask)
        
        # Project to vocabulary
        x = self.to_tokens(x)
        
        # Verify output shape
        assert x.shape == (batch_size, seq_length, self.to_tokens.out_features), \
            f"Expected shape {(batch_size, seq_length, self.to_tokens.out_features)}, got {x.shape}"
        
        return x

# Helper function to create causal mask for self-attention
def create_causal_mask(seq_length: int, device: torch.device) -> torch.Tensor:
    """Creates a causal mask for decoder self-attention"""
    mask = torch.triu(torch.ones((seq_length, seq_length), device=device), diagonal=1).bool()
    return ~mask  # Inverse because we want 1s where attention is allowed
