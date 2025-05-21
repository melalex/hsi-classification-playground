from torch import nn
import torch

class DemocraticModel(nn.Module):

    def __init__(self, models: list[nn.Module], weights: list[float]):
        super().__init__()

        self.models = nn.ModuleList(models)
        self.weights = weights

    def forward(self, x):
        y_pred = self.models(x)
        mode_vals, _ = torch.mode(y_pred, dim=0)
        return mode_vals
