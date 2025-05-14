import torch.nn as nn

from torchvision.models import VisionTransformer


class HsiVisionTransformer(nn.Module):

    def __init__(
        self,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
    ):
        super().__init__()

        self.stem = Conv3DStem()

        self.vit = VisionTransformer(
            image_size=224,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.vit(x)

        return x


class Conv3DStem(nn.Module):
    def __init__(self):
        super(Conv3DStem, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(200, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=68, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)
