import torch
import torch.nn as nn

class LightweightTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers, dropout):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.embedding = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, x):
        B = x.size(0)
        x = self.embedding(x)
        cls_tokens = self.cls.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.encoder(x)
        return x[:, 0]   # CLS token
