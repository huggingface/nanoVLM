import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, pre_norm: bool = True, return_attn_weights: bool = False):
        """
        A configurable Transformer-style self-attention block supporting Pre-LN and Post-LN.

        Args:
            embed_dim (int): Dimension of the embeddings.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate for attention and FFN.
            pre_norm (bool): If True, uses Pre-LayerNorm. If False, uses Post-LayerNorm.
            return_attn_weights (bool): If True, returns attention weights.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pre_norm = pre_norm
        self.return_attn_weights = return_attn_weights

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # LayerNorm layers
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Feed-forward network (position-wise MLP)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attention_mask=None):
        """
        Forward pass for the self-attention layer.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, embed_dim].
            attention_mask (Tensor or None): Binary mask (1 for valid tokens, 0 for padding).

        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, embed_dim].
        """
        key_padding_mask = None
        if attention_mask is not None:
            # Convert to key_padding_mask for PyTorch MHA: True = ignore
            key_padding_mask = (attention_mask == 0)

        if self.pre_norm:
            # --- Pre-LayerNorm Variant ---
            residual = x
            x_norm = self.attn_norm(x)
            attn_out, attn_weights = self.attn(
                query=x_norm,
                key=x_norm,
                value=x_norm,
                key_padding_mask=key_padding_mask,
                need_weights=self.return_attn_weights
            )
            x = residual + self.dropout(attn_out)

            residual = x
            x_norm = self.ffn_norm(x)
            ffn_out = self.ffn(x_norm)
            x = residual + ffn_out

        else:
            # --- Post-LayerNorm Variant ---
            residual = x
            attn_out, attn_weights = self.attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=key_padding_mask,
                need_weights=self.return_attn_weights
            )
            x = self.attn_norm(residual + self.dropout(attn_out))

            residual = x
            ffn_out = self.ffn(x)
            x = self.ffn_norm(residual + ffn_out)

        if self.return_attn_weights:
            return x, attn_weights
        return x
