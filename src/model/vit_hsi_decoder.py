import torch
import torch.nn as nn
import torch.nn.functional as F

class ViTHsiDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        out_bands: int = 75,
        out_h: int = 9,
        out_w: int = 9,
        embed_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        ViT-style decoder: latent vector -> (out_bands, out_h, out_w)

        Args:
            latent_dim: size of input latent vector (e.g. 256)
            out_bands: number of spectral bands in output (75)
            out_h, out_w: spatial shape (9, 9)
            embed_dim: token embedding dimension inside transformer
            n_layers: number of transformer encoder layers
            n_heads: attention heads
            mlp_dim: feed-forward dimension in transformer blocks
            dropout: dropout probability
        """
        super().__init__()

        self.out_bands = out_bands
        self.out_h = out_h
        self.out_w = out_w
        self.n_patches = out_h * out_w
        self.embed_dim = embed_dim

        # Map latent vector to initial token embeddings: (B, n_patches, embed_dim)
        self.latent_to_tokens = nn.Linear(latent_dim, self.n_patches * embed_dim)

        # Learnable positional embeddings for the output spatial tokens
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, embed_dim))

        # Transformer encoder (ViT-style). Use batch_first for convenience.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final projection from token embedding -> spectral bands
        self.to_spectrum = nn.Linear(embed_dim, out_bands)

        # Small output normalization
        self.norm = nn.LayerNorm(embed_dim)

        # optional small decoder head refinement (conv) -> not required but sometimes helpful
        # We'll keep it optional: a small conv after reshaping (disabled by default).
        self.refine_conv = (
            None  # e.g. nn.Conv2d(out_bands, out_bands, kernel_size=3, padding=1)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.latent_to_tokens.weight)
        if self.latent_to_tokens.bias is not None:
            nn.init.zeros_(self.latent_to_tokens.bias)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.to_spectrum.weight)
        if self.to_spectrum.bias is not None:
            nn.init.zeros_(self.to_spectrum.bias)

    def forward(self, latent: torch.Tensor):
        """
        Args:
            latent: (B, latent_dim)
        Returns:
            out: (B, out_bands, out_h, out_w)
        """
        B = latent.shape[0]
        # map latent -> (B, n_patches * embed_dim)
        x = self.latent_to_tokens(latent)  # (B, n_patches * embed_dim)
        x = x.view(B, self.n_patches, self.embed_dim)  # (B, n_patches, embed_dim)

        # add positional embeddings
        x = x + self.pos_embed  # broadcast (1, n_patches, embed_dim)

        # transformer
        x = self.transformer(x)  # (B, n_patches, embed_dim)

        # optional final normalization
        x = self.norm(x)

        # project tokens to spectra
        spec = self.to_spectrum(x)  # (B, n_patches, out_bands)

        # reshape to spatial map: (B, out_bands, H, W)
        spec = spec.permute(0, 2, 1)  # (B, out_bands, n_patches)
        spec = spec.view(B, self.out_bands, self.out_h, self.out_w)

        # optional refinement conv
        if self.refine_conv is not None:
            spec = self.refine_conv(spec)

        return spec
