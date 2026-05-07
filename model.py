"""Lasso peptide classifier model definition.

Architecture pipeline:
  ESM-2 frozen embeddings (batch, seq_len, embed_dim)
  → 3× Conv1D blocks (Conv1d → BatchNorm1d → ReLU → MaxPool1d(k=2))
  → Multi-Head Attention (residual connection + LayerNorm)
  → Global mean pooling across sequence dimension
  → 2-layer MLP classifier (Dropout → Linear → ReLU → Dropout → Linear → 1)

Padding mask is computed once from zero-sum embedding positions (ESM-2 padding)
and tracked through max-pooling in parallel with the feature stream, so masked
positions never contaminate valid representations.

~736K trainable parameters with the default esm2_t12_35M (embed_dim=480).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LassoPeptideClassifier(nn.Module):
    """CNN + Multi-Head Attention classifier for lasso peptide prediction.

    Args:
        embed_dim: ESM-2 embedding dimension (320/480/640/1280 depending on model).
        cnn_channels: list of output channels for each Conv1D block. Default [128,128,256].
        cnn_kernels: list of kernel sizes (padding = k//2). Default [5,3,3].
        attention_heads: number of heads in MultiheadAttention. Default 4.
        mlp_hidden: hidden dimension in the MLP classifier head. Default 64.
        dropout: dropout rate applied before each Linear layer in the MLP. Default 0.3.
    """

    def __init__(
        self,
        embed_dim=480,
        cnn_channels=None,
        cnn_kernels=None,
        attention_heads=4,
        mlp_hidden=64,
        dropout=0.3,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [128, 128, 256]
        if cnn_kernels is None:
            cnn_kernels = [5, 3, 3]

        # Stacked 1D convolutional blocks with batch norm and max pooling
        in_ch = embed_dim
        conv_blocks = []
        for ch, k in zip(cnn_channels, cnn_kernels):
            conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, ch, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(ch),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),  # halves sequence length each block
                )
            )
            in_ch = ch
        self.conv_blocks = nn.ModuleList(conv_blocks)

        # Multi-head self-attention on the CNN output features
        attn_dim = cnn_channels[-1]
        self.attention = nn.MultiheadAttention(
            attn_dim, num_heads=attention_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(attn_dim)

        # MLP classifier head: attn_dim → mlp_hidden → 1 (single logit output)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(attn_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, seq_len, embed_dim) frozen ESM-2 embeddings.

        Returns:
            (batch, 1) raw logits. Apply torch.sigmoid for probabilities.
        """
        # Build padding mask: positions where the embedding vector is all-zero are padding.
        # ESM-2 with padding="max_length" produces zero embeddings for padded positions.
        padding_mask = (x.abs().sum(dim=-1) == 0)

        # Permute to (batch, channels, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        for conv in self.conv_blocks:
            x = conv(x)
            # Downsample mask in parallel with the feature map (same max pooling)
            mask_1d = padding_mask.float().unsqueeze(1)
            mask_1d = F.max_pool1d(mask_1d, kernel_size=2, stride=2)
            padding_mask = mask_1d.squeeze(1).bool()

        # Back to (batch, seq_len, channels) for attention
        x = x.permute(0, 2, 1)
        attn_out, _ = self.attention(x, x, x, key_padding_mask=padding_mask)
        x = self.attn_norm(x + attn_out)  # residual connection + layer norm

        # Global mean pooling across sequence (ignores fully-masked positions
        # since they were zeroed by the padding mask propagation)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
