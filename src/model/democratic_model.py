from collections import defaultdict
from torch import nn
import torch


class DemocraticModel(nn.Module):

    def __init__(self, models: list[nn.Module], weights: list[float]):
        super().__init__()

        self.models = nn.ModuleList(models)
        self.weights = weights

    def forward(self, x):
        def calculate_group_confidence(group: list[int]):
            group_size = len(group)
            laplas = (group_size + 0.5) / (group_size + 1)
            group_weight = sum((self.weights[i] for i in group))

            return laplas * (group_weight / group_size)

        groups = defaultdict(list)

        for i, m in enumerate(self.models):
            y_pred = torch.argmax(m(x), dim=1)
            groups[y_pred].append(i)

        confidence = {k: calculate_group_confidence(v) for k, v in groups.items()}

        return max(confidence, key=confidence.get)
