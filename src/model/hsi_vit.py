import numpy as np
import torch
import torch.nn as nn


class HsiMultiHeadSelfAttention(nn.Module):
    def __init__(self, d, n_heads=2):
        super(HsiMultiHeadSelfAttention, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.k_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.v_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head : (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head**0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class HsiVitBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(HsiVitBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.hsa = HsiMultiHeadSelfAttention(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x):
        out = x + self.hsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class HsiVisualTransformer(nn.Module):
    def __init__(
        self,
        patch_shape,
        n_splits,
        n_blocks=2,
        hidden_d=8,
        n_heads=2,
        out_d=10,
    ):
        # Super constructor
        super(HsiVisualTransformer, self).__init__()

        # Attributes
        c, h, w = patch_shape

        self.n_splits = n_splits
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        self.split_size = int(c / n_splits)

        # 1) Linear mapper
        self.input_d = n_splits
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        self.register_buffer(
            "positional_embeddings",
            self.__get_positional_embeddings(
                self.split_size * h * w + 1, hidden_d
            ),
            persistent=False,
        )

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [HsiVitBlock(hidden_d, n_heads) for _ in range(n_blocks)]
        )

        # 5) Classification MLPk
        self.mlp = nn.Sequential(nn.Linear(self.hidden_d, out_d), nn.Softmax(dim=-1))

    def forward(self, images):
        # Dividing images into patches
        n, _, _, _ = images.shape
        patches = self.__patchify(images, self.n_splits).to(
            self.positional_embeddings.device
        )

        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Getting the classification token only
        out = out[:, 0]

        return self.mlp(out)  # Map to output dimension, output category distribution

    def __patchify(self, x, n_splits):
        n, c, h, w = x.shape

        assert c % n_splits == 0, "c must be divisible by splits"
        group_channels = c // n_splits

        # Reshape
        x = x.view(n, n_splits, group_channels, h, w)  # [n, splits, c // splits, h, w]
        x = x.permute(0, 3, 4, 2, 1).contiguous()                    # [n, h, w, c // splits, splits]
        x = x.view(n, h * w * group_channels, n_splits)  # [n, h * w * (c / splits), splits]

        return x

    def __get_positional_embeddings(self, sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i][j] = (
                    np.sin(i / (10000 ** (j / d)))
                    if j % 2 == 0
                    else np.cos(i / (10000 ** ((j - 1) / d)))
                )
        return result
