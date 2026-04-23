import torch
import torch.nn as nn


class LassoPeptideClassifier(nn.Module):
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

        in_ch = embed_dim
        conv_blocks = []
        for ch, k in zip(cnn_channels, cnn_kernels):
            conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, ch, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(ch),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                )
            )
            in_ch = ch
        self.conv_blocks = nn.ModuleList(conv_blocks)

        attn_dim = cnn_channels[-1]
        self.attention = nn.MultiheadAttention(
            attn_dim, num_heads=attention_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(attn_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(attn_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for conv in self.conv_blocks:
            x = conv(x)
        x = x.permute(0, 2, 1)
        attn_out, _ = self.attention(x, x, x)
        x = self.attn_norm(x + attn_out)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
