import torch

from torch import nn


class Ensemble(nn.Module):

    def __init__(self, modules: list[nn.Module]):
        super().__init__()

        self.co_modules = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.stack([it(x) for it in self.co_modules])

        return torch.sum(y, dim=0) / len(self.co_modules)


class MultiViewEnsemble(nn.Module):

    def __init__(self, modules: list[nn.Module], confidence_treshhold=0.5):
        super().__init__()

        self.co_modules = nn.ModuleList(modules)
        self.confidence_treshhold = confidence_treshhold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        result = torch.zeros(batch_size, dtype=int, device=x.device)

        y = torch.stack([it(x) for it in self.co_modules])

        coef, idx = torch.max(y, dim=0)

        positive = torch.sigmoid(coef) > self.confidence_treshhold

        result[positive] = idx[positive] + 1

        return result

    def get_params(self):
        return {"confidence_treshhold": self.confidence_treshhold}
