import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):

    def __init__(self, patch_h, patch_w, embeding_size):
        super().__init__()
        self.linear = nn.Linear(patch_h * patch_w, embeding_size)

    def forward(self, x):
        B, C, H, W = x.shape  
        x = x.view(B, C, -1) 
        x = self.linear(x)

        return x


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        dim=64,
        depth=5,
        heads=4,
        mlp_dim=8,
        dropout=0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=mlp_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class HsiVisionTransformer(nn.Module):

    def __init__(
        self,
        num_classes,
        input_shape,
        hidden_dim=64,
        num_layers=5,
        num_heads=4,
        mlp_dim=8,
        dropout=0.1,
    ):
        super().__init__()
        _, patch_h, patch_w = input_shape

        self.embed = PatchEmbedding(patch_h, patch_w, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.transformer = TransformerEncoder(
            dim=hidden_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)  

        x = self.transformer(x)
        x = x[:, 0] 
        return self.mlp_head(x)
