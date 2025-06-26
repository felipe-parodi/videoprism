import torch
import torch.nn as nn

class MultiHeadAttentionPooling(nn.Module):
    """
    Implements the Multi-Head Attention Pooling (MAP) layer.

    This layer uses a learnable query token to aggregate information from a
    sequence of feature embeddings via cross-attention. This is the trainable
    "head" that sits on top of the frozen VideoPrism backbone.
    """
    def __init__(self, feature_dim: int, num_heads: int, num_classes: int, dropout: float = 0.5):
        """
        Args:
            feature_dim: The dimensionality of the input features from VideoPrism (e.g., 768).
            num_heads: The number of attention heads.
            num_classes: The number of output behavior classes.
            dropout: The dropout rate.
        """
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim))
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        self.layernorm1 = nn.LayerNorm(feature_dim)
        self.layernorm2 = nn.LayerNorm(feature_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MAP layer.

        Args:
            x: Input tensor of shape (batch_size, num_tokens, feature_dim)
               from the VideoPrism encoder.

        Returns:
            Output tensor of shape (batch_size, num_classes) representing
            the logits for each class.
        """
        # Expand query to batch size
        query = self.query.expand(x.shape[0], -1, -1)
        
        # Cross-attention: query attends to the input features
        attn_output, _ = self.attention(query=query, key=x, value=x)
        
        # First residual connection
        x_res1 = self.layernorm1(attn_output + query)
        
        # Feed-forward network
        ffn_output = self.ffn(x_res1)
        
        # Second residual connection
        x_res2 = self.layernorm2(ffn_output + x_res1)
        
        # Classifier
        pooled_output = self.dropout(x_res2.squeeze(1))
        logits = self.classifier(pooled_output)
        
        return logits 