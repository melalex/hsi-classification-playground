import torch

from torch import nn


class Ensemble(nn.Module):

    def __init__(self, modules: list[nn.Module]):
        super().__init__()

        self.co_modules = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.stack([it(x) for it in self.co_modules])

        return torch.softmax(torch.sum(y, dim=0), dim=0)
